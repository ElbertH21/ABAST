"""
DMTG (attacker A3) environment smoke test.

Verifies the isolated PyTorch/diffusers env actually *executes* on the RTX 5070
(Blackwell, sm_120), not merely that imports succeed. Import-only checks pass even
when the wheel has no sm_120 kernels — the failure only shows up when a real kernel
is launched ("no kernel image is available for execution on the device"). So every
check below runs an actual GPU op.

Run:
    attacker/dmtg/.venv/bin/python attacker/dmtg/smoke_test.py

Exit code 0 iff every check passes.
"""

import sys
import traceback

results = []  # (name, passed: bool, detail: str)


def check(name):
    """Decorator: run a check fn, record PASS/FAIL, never raise out."""
    def wrap(fn):
        try:
            detail = fn()
            results.append((name, True, detail or ""))
        except Exception as e:
            results.append((name, False, f"{type(e).__name__}: {e}"))
            traceback.print_exc()
        return fn
    return wrap


import torch
import torch.nn as nn


print("=" * 68)
print("DMTG environment smoke test")
print("=" * 68)

# ---- Basic version / availability info (printed regardless of pass/fail) ----
print(f"torch version        : {torch.__version__}")
print(f"torch.version.cuda   : {torch.version.cuda}")
print(f"cuda.is_available()  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device name          : {torch.cuda.get_device_name(0)}")
    print(f"device capability    : {torch.cuda.get_device_capability(0)}")
else:
    print("device name          : <no CUDA device>")
    print("device capability    : <no CUDA device>")
print("-" * 68)


@check("CUDA available")
def _cuda_available():
    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    return torch.cuda.get_device_name(0)


@check("sm_120 capability == (12, 0)")
def _capability():
    cap = torch.cuda.get_device_capability(0)
    assert cap == (12, 0), f"expected (12, 0) for Blackwell/sm_120, got {cap}"
    return f"capability {cap}"


@check("Conv1d forward+backward on CUDA")
def _conv_fb():
    # A real kernel launch + autograd. This is what catches a missing-sm_120 wheel.
    dev = torch.device("cuda")
    conv = nn.Conv1d(in_channels=2, out_channels=8, kernel_size=3, padding=1).to(dev)
    x = torch.randn(4, 2, 64, device=dev, requires_grad=True)
    y = conv(x)
    loss = y.pow(2).mean()
    loss.backward()
    torch.cuda.synchronize()
    assert x.grad is not None and torch.isfinite(x.grad).all(), "bad/absent input grad"
    assert conv.weight.grad is not None and torch.isfinite(conv.weight.grad).all(), "bad/absent weight grad"
    return f"out {tuple(y.shape)}, loss {loss.item():.4f}"


@check("UNet1DModel + DDIMScheduler one denoise step on CUDA")
def _diffusion_step():
    from diffusers import UNet1DModel, DDIMScheduler

    dev = torch.device("cuda")
    # Small UNet operating on a (batch, 2, 64) trajectory (2 channels = dx/dy).
    model = UNet1DModel(
        sample_size=64,
        in_channels=2,
        out_channels=2,
        layers_per_block=1,
        block_out_channels=(32, 64),
        down_block_types=("DownBlock1D", "DownBlock1D"),
        up_block_types=("UpBlock1D", "UpBlock1D"),
    ).to(dev)

    scheduler = DDIMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(50)

    sample = torch.randn(3, 2, 64, device=dev)          # dummy noisy trajectory
    t = scheduler.timesteps[0]
    with torch.no_grad():
        noise_pred = model(sample, t).sample             # UNet forward on GPU
        stepped = scheduler.step(noise_pred, t, sample).prev_sample  # one denoise step
    torch.cuda.synchronize()

    assert noise_pred.shape == sample.shape, f"noise_pred shape {noise_pred.shape} != {sample.shape}"
    assert stepped.shape == sample.shape, f"stepped shape {stepped.shape} != {sample.shape}"
    assert torch.isfinite(stepped).all(), "non-finite values after denoise step"
    return f"noise_pred {tuple(noise_pred.shape)} -> prev_sample {tuple(stepped.shape)}"


# ---- Report ----
print("-" * 68)
all_pass = True
for name, passed, detail in results:
    tag = "PASS" if passed else "FAIL"
    all_pass = all_pass and passed
    line = f"[{tag}] {name}"
    if detail:
        line += f"  ({detail})"
    print(line)
print("=" * 68)
print("RESULT:", "ALL PASS" if all_pass else "FAILURE — see above")
sys.exit(0 if all_pass else 1)
