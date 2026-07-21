# data/ — obtain / regenerate (gitignored)

This whole folder is **gitignored** (except this README). Nothing here is committed;
each subfolder is either an external dataset you download or attacker output you
regenerate from `attacker/`. Populate it locally.

## Subfolders

- **`balabit/`** — Balabit *Mouse Dynamics Challenge* (real human sessions; A0 human
  imposter baseline + training corpus for A2/A3). Obtain from
  <https://github.com/balabit/Mouse-Dynamics-Challenge>.
  In this working copy the dataset is currently used in place from the top-level
  `Mouse-Dynamics-Challenge/` clone (attacker code references that path directly);
  copy or symlink it into `data/balabit/` if you want everything under `data/`.

- **`windmouse/`** — A1 WindMouse sessions. Regenerate:
  `python attacker/windmouse/save_windmouse_csvs.py`

- **`sapiagent/`** — A2 SapiAgent sessions. Regenerate by running the
  `attacker/sapiagent/` step scripts in order:
  `build_actions_step3.py` → `train_step4.py` → `gen_step5.py` →
  `generate_step7.py` → `handoff_step9.py`.
  (Current working outputs — trained weights, `.npy`/`.pkl` corpora, per-seed
  session CSVs — live under the top-level `a2/` tree; move/copy the session CSVs
  here to consolidate under `data/`.)

- **`dmtg_inspired/`** — A3 DMTG-style diffusion sessions. Regenerate:
  `python attacker/dmtg/train.py` → `python attacker/dmtg/sample.py` →
  `python attacker/dmtg/emit.py`.

## Format

All generated sessions are Balabit-shaped rows (`record timestamp, x, y`, plus
label columns), so the defender notebook's `extract_features` runs on them
unchanged. The defender notebook (Colab) consumes these and writes the bypass
tables into `results/`.
