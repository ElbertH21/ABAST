"""A3 diffusion denoiser: a 1D UNet mapping (B,4,128) -> (B,2,128) epsilon.

Input channels = [noisy_residual_dx, noisy_residual_dy, D_dx_norm, D_dy_norm];
output = predicted noise on the 2 residual channels.

DEFAULT: CustomUNet1D — a small standard 1D UNet (timestep embedding injected into
every ResBlock, down/up sampling, skip connections).

Why not diffusers UNet1DModel: its plain DownBlock1D/UpBlock1D block types (built
for RL trajectory planning) do NOT inject the timestep embedding into their conv
blocks, so the denoiser is effectively blind to the noise level. Measured on a
32-sample overfit (identical setup): UNet1DModel floors at ~0.57 MSE and flatlines
after ~100 steps, while CustomUNet1D (11x fewer params) collapses to ~0.03. This is
the "UNet1DModel config fights the task -> write a custom 1D UNet" fallback the brief
authorised; here the fight is timestep conditioning, not the 4-in/2-out shape (the
shape works). UNet1DModel remains selectable via build_model(kind="diffusers") for
comparison, but is not A3's denoiser.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SAMPLE_SIZE = 128
IN_CHANNELS = 4
OUT_CHANNELS = 2


# ----------------------------- diffusers path --------------------------------
def _build_diffusers():
    from diffusers import UNet1DModel
    return UNet1DModel(
        sample_size=SAMPLE_SIZE,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        layers_per_block=1,
        block_out_channels=(64, 128, 256),
        down_block_types=("DownBlock1D", "DownBlock1D", "DownBlock1D"),
        up_block_types=("UpBlock1D", "UpBlock1D", "UpBlock1D"),
    )


# ----------------------------- custom fallback -------------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / max(half - 1, 1)
        )
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


def _gn(c):
    return nn.GroupNorm(8 if c % 8 == 0 else 1, c)


class ResBlock1D(nn.Module):
    def __init__(self, cin, cout, temb_dim):
        super().__init__()
        self.norm1 = _gn(cin)
        self.conv1 = nn.Conv1d(cin, cout, 3, padding=1)
        self.temb = nn.Linear(temb_dim, cout)
        self.norm2 = _gn(cout)
        self.conv2 = nn.Conv1d(cout, cout, 3, padding=1)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(temb)[:, :, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class CustomUNet1D(nn.Module):
    """Small 1D UNet: 128 -> 64 -> 32 -> 64 -> 128 with skips and timestep cond."""

    def __init__(self, in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS,
                 base=64, temb_dim=128):
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 4
        self.temb = nn.Sequential(
            SinusoidalPosEmb(temb_dim), nn.Linear(temb_dim, temb_dim),
            nn.SiLU(), nn.Linear(temb_dim, temb_dim),
        )
        self.in_conv = nn.Conv1d(in_channels, c1, 3, padding=1)
        self.down1 = ResBlock1D(c1, c1, temb_dim)
        self.pool1 = nn.Conv1d(c1, c1, 4, stride=2, padding=1)   # 128 -> 64
        self.down2 = ResBlock1D(c1, c2, temb_dim)
        self.pool2 = nn.Conv1d(c2, c2, 4, stride=2, padding=1)   # 64 -> 32
        self.mid = ResBlock1D(c2, c3, temb_dim)
        self.up2 = nn.ConvTranspose1d(c3, c2, 4, stride=2, padding=1)  # 32 -> 64
        self.dec2 = ResBlock1D(c2 + c2, c2, temb_dim)
        self.up1 = nn.ConvTranspose1d(c2, c1, 4, stride=2, padding=1)  # 64 -> 128
        self.dec1 = ResBlock1D(c1 + c1, c1, temb_dim)
        self.out_norm = _gn(c1)
        self.out_conv = nn.Conv1d(c1, out_channels, 3, padding=1)

    def forward(self, x, t):
        if not torch.is_tensor(t):
            t = torch.tensor([t], device=x.device)
        t = t.to(x.device)                     # scheduler.timesteps are CPU tensors
        if t.ndim == 0:
            t = t[None]
        if t.shape[0] != x.shape[0]:
            t = t.expand(x.shape[0])
        temb = self.temb(t)
        h = self.in_conv(x)
        d1 = self.down1(h, temb)               # (c1,128)
        h = self.pool1(d1)                      # (c1,64)
        d2 = self.down2(h, temb)               # (c2,64)
        h = self.pool2(d2)                      # (c2,32)
        h = self.mid(h, temb)                  # (c3,32)
        h = self.up2(h)                         # (c2,64)
        h = self.dec2(torch.cat([h, d2], 1), temb)
        h = self.up1(h)                         # (c1,128)
        h = self.dec1(torch.cat([h, d1], 1), temb)
        out = self.out_conv(F.silu(self.out_norm(h)))
        return _Output(out)


class _Output:
    """Mimic diffusers' UNet1DOutput.sample so train.py is agnostic."""
    def __init__(self, sample):
        self.sample = sample


# ----------------------------- selector --------------------------------------
def build_model(kind="custom", verbose=True):
    """Return (model, name).

    kind="custom" (default): CustomUNet1D — A3's denoiser (timestep-conditioned).
    kind="diffusers": diffusers UNet1DModel (verified shape), for comparison only;
        note it is timestep-blind and floors ~0.57 MSE (see module docstring).
    """
    if kind == "diffusers":
        m = _build_diffusers()
        m.eval()
        with torch.no_grad():
            y = m(torch.zeros(2, IN_CHANNELS, SAMPLE_SIZE),
                  torch.zeros(2, dtype=torch.long)).sample
        assert tuple(y.shape) == (2, OUT_CHANNELS, SAMPLE_SIZE), \
            f"unexpected output shape {tuple(y.shape)}"
        if verbose:
            print(f"[model] using diffusers UNet1DModel (verified {tuple(y.shape)}, "
                  f"timestep-blind — comparison only)")
        return m, "diffusers-UNet1DModel"
    m = CustomUNet1D()
    if verbose:
        print("[model] using CustomUNet1D (A3 denoiser)")
    return m, "CustomUNet1D"


def count_params(model):
    return sum(p.numel() for p in model.parameters())
