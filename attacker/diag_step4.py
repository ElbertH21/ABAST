"""Diagnostic: is the flat MSE an irreducible one-to-many floor, or an
optimization/scale problem? Compare baselines + raw vs standardized training."""
import os, sys
import numpy as np
sys.path.insert(0, "/home/junio/reu-sapiagent/sapiagent")
import tensorflow as tf
from sklearn.model_selection import train_test_split
import settings as stt
from autoencoder_models import fcn_autoencoder
from balabit_actions import build_equidistant_inputs

targets = np.load("artifacts/balabit_human_actions.npy").astype(np.float32)
inputs = build_equidistant_inputs(targets, 128).astype(np.float32)

predict0 = float(np.mean(targets**2))
copyinput = float(np.mean((targets - inputs)**2))
print(f"BASELINE predict-zero MSE (=mean target^2): {predict0:.1f} px^2")
print(f"BASELINE copy-input  MSE (=mean (t-in)^2 ): {copyinput:.1f} px^2")
print(f"  input delta |mean abs|: {np.abs(inputs).mean():.3f} px   "
      f"target delta |mean abs|: {np.abs(targets).mean():.3f} px")

Xtr_in, Xva_in, Xtr_tg, Xva_tg = train_test_split(
    inputs, targets, test_size=0.10, random_state=11235, shuffle=True)

EP = 15
# --- RAW ---
tf.keras.utils.set_random_seed(11235)
_, m_raw = fcn_autoencoder((128, 2), 128)
print("\n--- RAW (no scaling) ---")
h = m_raw.fit(Xtr_in, Xtr_tg, validation_data=(Xva_in, Xva_tg),
              epochs=EP, batch_size=32, verbose=0)
for e in range(EP):
    print(f"  epoch {e+1:2d}: train={h.history['loss'][e]:8.1f}  val={h.history['val_loss'][e]:8.1f}")

# --- STANDARDIZED (fit per-feature std on train targets, apply to both) ---
sd = Xtr_tg.reshape(-1, 2).std(axis=0).astype(np.float32)  # (2,)
print(f"\n--- STANDARDIZED (divide by target std {sd}) ---")
sc = lambda a: a / sd
tf.keras.utils.set_random_seed(11235)
_, m_std = fcn_autoencoder((128, 2), 128)
h2 = m_std.fit(sc(Xtr_in), sc(Xtr_tg), validation_data=(sc(Xva_in), sc(Xva_tg)),
               epochs=EP, batch_size=32, verbose=0)
for e in range(EP):
    print(f"  epoch {e+1:2d}: train_norm={h2.history['loss'][e]:.4f}  val_norm={h2.history['val_loss'][e]:.4f}")
# px^2 val MSE via inverse transform
pred = m_std.predict(sc(Xva_in), verbose=0) * sd
print(f"  standardized model val MSE in px^2 (inverse-transformed): {np.mean((pred - Xva_tg)**2):.1f}")
print("DIAG_DONE")
