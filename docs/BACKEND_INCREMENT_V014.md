# BEV-MotionLab — Backend increment v0.1.4: fixed-reference camera-motion audit

**Status:** Engineering/synthetic implementation only; backend v1 acceptance **HOLD**. No real video or surveyed measurements used.

## Implemented

- `pose_audit.audit_pose`: strict independent observation of at least 4 known fixed landmarks in undistorted pixel coordinates, with non-collinear configuration check.
- Compares `camera_id`, `lens_calibration_id`, exact `resolution_px`, and pixel convention to the originally documented calibration. Missing, duplicated, unknown, out-of-bounds or non-finite fiducials are rejected.
- Explicit maximum and RMSE displacement thresholds in **pixels**; no physical/metre thresholds implied.
- Opt-in `analyze --pose-audit-json path.json --pose-max-rmse-px 2 --pose-max-individual-px 4`; if blocked, no analysis outputs are created. Existing analyses without pose evidence keep prior behavior and write `pose_audit.json: NOT_CHECKED`.
- SHA-256 input audit is added to `run_config.json`, with `pose_audit.json` included in `manifest.json`.
- Synthetic unit and CLI failure-path regression tests ensure calibrated data cannot bypass a supplied failed pose audit.

Example pose audit input:

```json
{"camera_id":"synthetic_cam","lens_calibration_id":"synthetic_identity","resolution_px":[800,600],"pixel_convention":"undistorted","points":[{"id":"A","image_uv_px":[200,200]}, {"id":"B","image_uv_px":[600,200]}, {"id":"C","image_uv_px":[200,450]}, {"id":"D","image_uv_px":[600,450]}]}
```

## Critical limitations

- This checks **observed image movement of immobile reference markers**. It does **not** recover 3D camera pose, detect movement between audit snapshots, validate the undistortion model, or prove a speed measurement is physically accurate.
- Synthetic fixture marker coordinates do not count as an independent real-world camera audit. Passing is `NUMERICALLY_STABLE`, not safe/deployable.
- The audit is opt-in, preserving the legacy CLI. For an eventual production-style workflow, a policy must require it and tie snapshots to each analyzed period.
- Distortion model provenance, ground-height parallax, varying contact points, realistic camera drift, lens bias and real-data acceptance remain open. **Real data last.**
