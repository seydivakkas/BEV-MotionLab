# BEV-MotionLab — backend v0.1.7 per-interval height-risk gate

## Implemented

- `src/bevmotionlab/interval_validity.py`: every valid speed interval is paired exactly with its start/end pixel observations, applies the conditional conservative `speed_bias_bound`, and writes lower/upper speed envelopes.
- Opt-in `analyze --camera3d-model-json MODEL --contact-max-height-m H --max-height-speed-bias-kmh B`. Both new thresholds must be present and finite, and an independently holdout-audited 3D camera model is mandatory.
- If **any VALID speed interval** exceeds the declared `B` risk budget, **the run fails before creating an output directory**. Invalid/gap intervals remain explicitly `NOT_EVALUATED_INVALID_SPEED`; no false numerical estimate is assigned.
- Upon conditional acceptance, `speed_height_risk.csv`, `height_bias_audit.json` and SHA-256 manifest are created. Existing `speed_timeseries.csv` and CLI defaults are unchanged. Four new integration/negative-control tests.

## Boundaries / HOLD

A monocular image does not measure target footpoint height. `H` is a *user-declared assumption*; bound excludes systematic lens miscalibration, frame-specific pose drift, lens model failures and model mismatch. It is **not** a 95% confidence interval or ground truth. Independent real field measurements remain LAST, after backend and frontend. Until full camera/ROI, invalid-contact, height-drift, resource and package acceptance, backend v1 stays HOLD.