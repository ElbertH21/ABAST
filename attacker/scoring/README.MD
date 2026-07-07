# Attacker-side bypass scoring

Bridges the defender's trained T0–T3 models to attacker bypass rates.
Scores WindMouse (A1) and SapiAgent (A2) attack sessions against every
tier at the FAR<=0.10 threshold, all-tiers (supersedes the earlier
T3-only WindMouse evaluation).

- `attacker_bypass_scoring.ipynb` — the scoring notebook (to be added after the grid run)
- output: `attack_results_all_tiers.csv` — per (user, tier, model, seed) bypass rates,
  the data feeding Mahnoor's vulnerability classifier and anomaly-detection tasks