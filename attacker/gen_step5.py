"""Step 5: generate humanized actions from the trained fcn and a freshly trained
BiGRU, then QUANTIFY whether either adds real structure over the straight-line
equidistant input. No converter (that's step 6)."""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/junio/reu-sapiagent/sapiagent")
import tensorflow as tf
from sklearn.model_selection import train_test_split
import settings as stt
from autoencoder_models import bidirectional_autoencoder
from balabit_actions import build_equidistant_inputs

ART = os.path.join(os.path.dirname(__file__), "artifacts")
SEED = stt.RANDOM_STATE  # 11235


def generate(model, X, bs=512):
    """Eager inference in modest batches. Calling model(x) directly avoids the
    compiled predict-function's XLA path, which on this Blackwell card requests a
    >16 GiB conv workspace at large batch and OOM-thrashes."""
    outs = []
    for i in range(0, len(X), bs):
        outs.append(np.asarray(model(X[i:i + bs], training=False)))
    return np.concatenate(outs, axis=0).astype(np.float32)

# ---- data (same as step 4) ----------------------------------------------
targets = np.load(os.path.join(ART, "balabit_human_actions.npy")).astype(np.float32)
inputs = build_equidistant_inputs(targets, 128).astype(np.float32)
print(f"targets {targets.shape}  inputs {inputs.shape}", flush=True)

# ---- 1. generate from the trained fcn -----------------------------------
print("Loading fcn_supervised.keras and generating...", flush=True)
fcn = tf.keras.models.load_model(os.path.join(ART, "fcn_supervised.keras"), compile=False)
fcn_gen = generate(fcn, inputs)
np.save(os.path.join(ART, "fcn_gen.npy"), fcn_gen)  # persist before the long BiGRU run
print(f"fcn_gen {fcn_gen.shape}", flush=True)

# ---- 2. train the BiGRU (same recipe as step 4) -------------------------
tf.keras.utils.set_random_seed(SEED)
Xtr_in, Xva_in, Xtr_tg, Xva_tg = train_test_split(
    inputs, targets, test_size=0.10, random_state=SEED, shuffle=True)
print("Training BiGRU (bidirectional, supervised, mse, 100 epochs)...", flush=True)
_, bigru = bidirectional_autoencoder(128, 2)  # jit_compile=False baked in; loss=stt.LOSS(mse)
hb = bigru.fit(Xtr_in, Xtr_tg, validation_data=(Xva_in, Xva_tg),
               epochs=stt.EPOCHS, batch_size=stt.BATCH_SIZE, shuffle=True, verbose=2)
btr, bva = hb.history["loss"], hb.history["val_loss"]
print(f"BiGRU loss: train {btr[0]:.1f}->{btr[-1]:.1f}  val {bva[0]:.1f}->{bva[-1]:.1f}", flush=True)
bigru.save(os.path.join(ART, "bigru_supervised.keras"))
print("Saved -> bigru_supervised.keras", flush=True)
bigru_gen = generate(bigru, inputs)
print(f"bigru_gen {bigru_gen.shape}", flush=True)

np.save(os.path.join(ART, "bigru_gen.npy"), bigru_gen)

# ---- 3. quantify structure ----------------------------------------------
def path_len(d):    return np.linalg.norm(d, axis=-1).sum(axis=-1)
def net_disp(d):    return np.linalg.norm(d.sum(axis=-2), axis=-1)
def straightness(d):return net_disp(d) / (path_len(d) + 1e-9)
def curvature(d):   # mean |turn angle| between consecutive step vectors (0 for zero vecs)
    v1, v2 = d[..., :-1, :], d[..., 1:, :]
    cross = v1[..., 0]*v2[..., 1] - v1[..., 1]*v2[..., 0]
    dot = (v1*v2).sum(-1)
    return np.abs(np.arctan2(cross, dot)).mean(axis=-1)
def dev_from_input(d, inp):  # mean per-step L2 position deviation from straight-line input
    return np.linalg.norm(np.cumsum(d, -2) - np.cumsum(inp, -2), axis=-1).mean(axis=-1)

# 12 sample actions spanning the (non-degenerate) real path-length distribution
pl = path_len(targets)
order = np.where(pl >= 10)[0]
order = order[np.argsort(pl[order])]
picks = np.array([order[int(r * (len(order) - 1))] for r in np.linspace(0.05, 0.97, 12)])
print(f"sample action indices (by path-length pct): {picks.tolist()}", flush=True)

sources = {"input": inputs[picks], "fcn-gen": fcn_gen[picks],
           "bigru-gen": bigru_gen[picks], "real": targets[picks]}
inp_s = inputs[picks]
rows = {}
for name, d in sources.items():
    dev = dev_from_input(d, inp_s)
    st = straightness(d)
    cv = curvature(d)
    rows[name] = {
        "mean_dev_from_input_px": float(dev.mean()), "dev_std": float(dev.std()),
        "straightness": float(st.mean()), "straightness_std": float(st.std()),
        "curvature_rad_per_step": float(cv.mean()), "curvature_std": float(cv.std()),
    }
table = pd.DataFrame(rows).T[["mean_dev_from_input_px", "dev_std", "straightness",
                              "straightness_std", "curvature_rad_per_step", "curvature_std"]]
table.to_csv(os.path.join(ART, "step5_geometry.csv"))
print("\n=== step5_geometry (mean over 12 samples) ===", flush=True)
print(table.round(3).to_string(), flush=True)

# ---- 4. plot generated vs input vs real ---------------------------------
fig, axes = plt.subplots(3, 4, figsize=(16, 11))
for ax, idx in zip(axes.ravel(), picks):
    for d, c, lbl, lw in [(inputs[idx], "#999999", "input", 1.2),
                          (targets[idx], "black", "real", 1.6),
                          (fcn_gen[idx], "#1f77b4", "fcn-gen", 1.4),
                          (bigru_gen[idx], "#ff7f0e", "bigru-gen", 1.4)]:
        p = np.cumsum(d, axis=0)
        ax.plot(p[:, 0], p[:, 1], "-", color=c, lw=lw, label=lbl,
                ls="--" if lbl == "input" else "-")
    ax.set_title(f"#{idx}  path={pl[idx]:.0f}px", fontsize=8)
    ax.invert_yaxis(); ax.set_aspect("equal", adjustable="datalim")
    ax.tick_params(labelsize=6)
axes.ravel()[0].legend(fontsize=7, loc="best")
fig.suptitle("Step 5: input (straight) vs fcn-gen vs bigru-gen vs real — 12 actions", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(ART, "step5_shapes.png"), dpi=105)
print(f"\nSaved plot -> {ART}/step5_shapes.png", flush=True)

with open(os.path.join(ART, "step5_bigru_history.json"), "w") as f:
    json.dump({"loss": btr, "val_loss": bva}, f, indent=2)
print("STEP5_DONE", flush=True)
