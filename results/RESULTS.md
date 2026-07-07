
## Step 8 — realism comparison
- [realism_metrics.csv](realism_metrics.csv), [realism_per_session.csv](realism_per_session.csv), [realism_distributions.png](realism_distributions.png)
- Finding: Genuine/WindMouse/SapiAgent pause-rate = 0.563 ± 0.27 / 0.852 ± 0.012 / 0.853 ± 0.0086: the two synthetic attackers share the empirical dt sampler so their pause rates match each other and sit far above genuine, confirming acceptance is a timing/threshold-gap effect, not mimicry — while curvature/jerk/velocity separate the three by geometry; WindMouse OOB 0.02% vs SapiAgent 2.87% rules out a clipping confound.
