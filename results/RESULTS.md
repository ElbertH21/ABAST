
## Step 8 — realism comparison
- [realism_metrics.csv](realism_metrics.csv), [realism_per_session.csv](realism_per_session.csv), [realism_distributions.png](realism_distributions.png)
- Finding: Genuine/WindMouse/SapiAgent pause-rate = 0.563 ± 0.27 / 0.852 ± 0.012 / 0.853 ± 0.0086: the two synthetic attackers share the empirical dt sampler so their pause rates match each other and sit far above genuine, confirming acceptance is a timing/threshold-gap effect, not mimicry — while curvature/jerk/velocity separate the three by geometry; WindMouse OOB 0.02% vs SapiAgent 2.87% rules out a clipping confound.

## Step 8 — realism comparison (4-source: +DMTG/A3)
- [realism_metrics.csv](realism_metrics.csv), [realism_per_session.csv](realism_per_session.csv), [realism_curvature_filtered.csv](realism_curvature_filtered.csv), [realism_distributions.png](realism_distributions.png)
- Finding: Four-source, one extract_features. Pause-rate genuine/WM/SA/DMTG = 0.563 ± 0.27 / 0.852 ± 0.012 / 0.853 ± 0.0086 / 0.853 ± 0.0095: all three synthetics share the empirical dt sampler so their pause rates match and sit far above genuine — acceptance is a timing/threshold-gap effect, not mimicry. Geometry separates them: velocity genuine/WM/SA/DMTG = 1.84e+03 ± 1.3e+03 / 369 ± 38 / 221 ± 32 / 528 ± 67. Displacement-filtered curvature (dist>=2px) genuine/WM/SA/DMTG = 0.103 / 0.0346 / 0.488 / 0.135: the raw curvature makes SapiAgent look human, but filtering the near-stationary steps shows SapiAgent ~4.8x genuine while DMTG stays close — correcting the 'SapiAgent matches human curvature' claim.
