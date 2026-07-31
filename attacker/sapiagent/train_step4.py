"""Step 4: train the supervised fcn autoencoder (equidistant input -> human target).

Verifies the endpoint invariant BEFORE training; if any sampled pair's net
displacement differs, it stops (exits non-zero) instead of training.
No generation here (that's step 5)."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.environ.get("SAPIAGENT_ROOT", "./sapiagent"))  # for settings + models
import tensorflow as tf
from sklearn.model_selection import train_test_split

import settings as stt
from autoencoder_models import fcn_autoencoder
from balabit_actions import build_equidistant_inputs

ART = os.path.join(os.path.dirname(__file__), "artifacts")
SEED = stt.RANDOM_STATE  # 11235
tf.keras.utils.set_random_seed(SEED)

# ---- 1. load target corpus ----------------------------------------------
targets = np.load(os.path.join(ART, "balabit_human_actions.npy")).astype(np.float32)
print(f"Target corpus: {targets.shape} dtype={targets.dtype}")
assert targets.shape[1:] == (128, 2)

# ---- 2. build equidistant inputs (same interleaved representation) --------
inputs = build_equidistant_inputs(targets, length=128).astype(np.float32)
print(f"Equidistant inputs: {inputs.shape} dtype={inputs.dtype}")
predict0_mse = float(np.mean(targets**2))  # irreducible-ish floor reference for the plot

# ---- INVARIANT CHECK (gates training) -----------------------------------
print("\n=== Endpoint invariant check (cumsum start/end must match; between differs) ===")
rng = np.random.default_rng(SEED)
sample_idx = rng.choice(targets.shape[0], size=3, replace=False)
ok = True
for k in sample_idx:
    tgt, inp = targets[k], inputs[k]
    ct, ci = np.cumsum(tgt, axis=0), np.cumsum(inp, axis=0)
    start_match = np.array_equal(np.zeros(2), np.zeros(2))  # both anchored at origin (0,0)
    end_t, end_i = ct[-1], ci[-1]                            # net displacement (= sum of deltas)
    end_match = np.array_equal(np.rint(end_t), np.rint(end_i))
    between_dev = float(np.max(np.linalg.norm(ct - ci, axis=1)))  # path divergence
    print(f"  action #{k:5d}: start(anchor)=(0,0) both={start_match} | "
          f"end target={tuple(np.rint(end_t).astype(int))} input={tuple(np.rint(end_i).astype(int))} "
          f"match={end_match} | max between-path deviation={between_dev:.1f}px")
    ok = ok and end_match and (between_dev > 0.0)
if not ok:
    print("\nENDPOINT INVARIANT FAILED — stopping without training.")
    sys.exit(2)
print("Invariant OK: inputs share start+end with targets, differ only in between.\n")

# ---- 3. normalization: repo trains on raw deltas (no scaler) -> none, symmetric
print("Normalization: NONE (repo autoencoder_training.py uses raw dx/dy deltas); "
      "identical (no) treatment applied to input and target.\n")

# ---- 4. deterministic 10% val split; supervised fit ---------------------
Xtr_in, Xva_in, Xtr_tg, Xva_tg = train_test_split(
    inputs, targets, test_size=0.10, random_state=SEED, shuffle=True)
print(f"Split: train={Xtr_in.shape[0]}  val={Xva_in.shape[0]}  (10% val, seed={SEED})")

LOSS = stt.LOSS  # 'mse'
print(f"Loss used: {LOSS}  (settings.LOSS)")
encoder, model = fcn_autoencoder((stt.FEATURES, stt.DIMENSIONS), fcn_filters=128)

EPOCHS = stt.EPOCHS  # 100
BATCH = stt.BATCH_SIZE  # 32
hist = model.fit(Xtr_in, Xtr_tg,
                 validation_data=(Xva_in, Xva_tg),
                 epochs=EPOCHS, batch_size=BATCH, shuffle=True, verbose=2)

tr = hist.history["loss"]; va = hist.history["val_loss"]
print(f"\n=== RESULT ===")
print(f"epochs run: {len(tr)}")
print(f"train loss: initial={tr[0]:.4f}  final={tr[-1]:.4f}  (min={min(tr):.4f})")
print(f"val   loss: initial={va[0]:.4f}  final={va[-1]:.4f}  (min={min(va):.4f})")
print(f"loss decreased: {tr[-1] < tr[0]}")

# ---- 5. save loss curve + model -----------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(tr, label="train"); plt.plot(va, label="val")
plt.xlabel("epoch"); plt.ylabel(f"{LOSS} loss (px^2)")  # linear/autoscaled: floor is ~6900
plt.axhline(predict0_mse, ls="--", c="gray", lw=1, label=f"predict-0 floor ({predict0_mse:.0f})")
plt.title(f"Step 4 supervised fcn: equidistant->human (loss={LOSS})")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(ART, "step4_loss.png"), dpi=110)
print(f"Saved loss curve -> {ART}/step4_loss.png")

model.save(os.path.join(ART, "fcn_supervised.keras"))  # native .keras (NOT .h5)
print(f"Saved model -> {ART}/fcn_supervised.keras")

with open(os.path.join(ART, "step4_history.json"), "w") as f:
    json.dump({"loss": tr, "val_loss": va, "loss_name": LOSS, "epochs": len(tr),
               "seed": SEED, "batch": BATCH}, f, indent=2)
print("STEP4_DONE")
