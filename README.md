# ABAST — Behavioral Authentication Under Attack: AI Mouse Bots, Human Impostors, and User-Specific Vulnerability

## Research Questions
1. How does attack success change as mouse bots become more sophisticated, and how do they compare with real human impostors?
2. Are some users more vulnerable to behavioral imitation than others, and does the strongest attacker depend on the target user?
3. Can additional defense mechanisms improve robustness against synthetic attacks without significantly reducing legitimate-user acceptance?

## Start here
- **Canonical run:** `defender/notebooks/Week8_ABAST_E_M.ipynb`, the leakage-free
  corrected evaluation. All reported numbers come from it.
- **Results:** every reported number traces to a CSV in `results/`.
- **Attackers** (`attacker/`): A1 WindMouse, A2 SapiAgent, A3 DMTG-style diffusion.
  **Defenders and evaluation** (`defender/`).

NSF REU 2026.

```
reu-sapiagent/
├── README.md
├── .gitignore
├── attacker/                 A1–A3 generator code (code only)
│   ├── sapiagent/            A2: FCN/BiGRU autoencoder pipeline + emitter (step scripts)
│   ├── windmouse/            A1: WindMouse session export
│   ├── dmtg/                 A3: DMTG-style 1D UNet + DDIM sampler
│   └── scoring/              notes bridging generated sessions → defender bypass scoring
├── defender/
│   └── notebooks/            reference defender/scoring notebooks (READ-ONLY)
├── data/                     gitignored — see data/README.md to obtain/regenerate
│   ├── balabit/  windmouse/  sapiagent/  dmtg_inspired/
├── results/                  committed CSV artifacts
│   └── figures/              committed PNG figures
├── paper/                    Overleaf mirror (.tex/.bib)
├── poster/
├── docs/
└── archive/                  superseded files kept for provenance
```

## Canonical run

The leakage-free corrected evaluation is
`defender/notebooks/Week8_ABAST_E_M.ipynb`; every reported number comes from it.
The other reference notebook tracked in `defender/notebooks/` is
`Week7_BehavioralAuth.ipynb` (post-floor defender results). All defender notebooks
are READ-ONLY reference — they execute in Colab against Drive + GPU, not locally.

## Data

`data/` is gitignored. See [`data/README.md`](data/README.md) for how to obtain and
regenerate each subfolder. The Balabit source is the
**Mouse Dynamics Challenge**: https://github.com/balabit/Mouse-Dynamics-Challenge

## Regenerating each attacker's data

Each generator writes Balabit-format sessions into its `data/` subfolder:

- **A1 — WindMouse** (`data/windmouse/`): `python attacker/windmouse/save_windmouse_csvs.py`
- **A2 — SapiAgent** (`data/sapiagent/`): run the `attacker/sapiagent/` step scripts in order —
  `build_actions_step3.py` → `train_step4.py` → `gen_step5.py` → `generate_step7.py` → `handoff_step9.py`
  (realism metrics via `realism_step8.py`).
- **A3 — DMTG-style** (`data/dmtg_inspired/`): train then sample —
  `python attacker/dmtg/train.py` → `python attacker/dmtg/sample.py` → `python attacker/dmtg/emit.py`.

A2 and A3 share the pooled-Balabit corpus and the per-step timing sampler; only the
generative mechanism differs. Generated sessions are handed to the defender notebook,
which produces the bypass tables in `results/`.

## Environment

- Attacker / diffusion: local (WSL2, RTX 5070), TensorFlow (A2) and PyTorch (A3), own venvs.
- Defender scoring: Google Colab, TensorFlow. Keep the two environments separate.

## References

- SapiAgent: Antal et al., IEEE Access, 2021 — https://github.com/margitantal68/sapiagent
- WindMouse: ben.land
- DMTG: Liu et al., arXiv:2410.18233
- Balabit Mouse Dynamics Challenge: https://github.com/balabit/Mouse-Dynamics-Challenge
