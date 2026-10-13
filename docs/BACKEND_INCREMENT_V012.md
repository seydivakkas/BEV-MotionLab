# BEV interval uncertainty v0.1.2 — 2026-10-11

## Code boundary
- Opt-in CLI `analyze --uncertainty-trials 200 --pixel-std-px 0.5 --landmark-std-px 0.5 --timestamp-std-s 0.0` emits `speed_uncertainty.csv` (included in SHA-256 manifest).
- Each interval reuses **one jointly perturbed** calibration fit for its two endpoints (correlated landmarks). Both contact points, and their timestamps, are independently sampled. Seeds are stable across Windows/Linux process hash randomization.
- Invalid GAP intervals never acquire a speed estimate. Insufficient Monte Carlo support fails *before* creating output. At present robust-RANSAC + Monte Carlo is explicitly refused: resampling all landmarks with DLT is not equivalent to robust fitting.
- Interval bounds are 2.5th and 97.5th percentiles of a *conditional synthetic error model*. They are **not** statistically calibrated real-world confidence intervals.

## Open before backend v1
1. Joint camera pose/lens uncertainty and independently validated undistortion; camera-pose drift and hardware-time provenance.
2. Model footpoint identity ambiguity, long occlusions, off-plane motion and systematic biases rather than treating narrow MC bands as accuracy.
3. Lock parameter/seed stress scenarios, resource ceilings and stability under near-horizon homographies.
4. Stable offline API, packaging/licensing and comprehensive synthetic E2E.
5. **Frontend after backend; real video/ground-survey data LAST**.

Status: **DRAFT / HOLD**.