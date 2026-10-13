# Backend gate ledger — v0.1.1 (2026-10-11)

## Change set and evidence

- New `quality.py`: deterministic, reviewable numerical gate for inlier support, fit RMSE and independent holdout in image pixels and world metres. Missing holdout reports `BLOCKED` rather than fabricating confidence.
- `calibrate` records numerical gate status in its JSON output. `analyze --require-quality-gate` fails before creating outputs if the gate is blocked. This switch is opt-in to retain older exploratory fixtures; it is mandatory for future strict-acceptance demonstrations.
- New `video.validate_detection_timestamps`: enforces exact frame index and timestamp-domain correspondence to `ffprobe` frame PTS with an explicitly configured offset. Multiple tracks on the same video frame must agree on its timestamp. There is no FPS-based guessing, timestamp replacement, camera sync, or video decoding.
- `analyze --ffprobe-json PTS.json [--pts-tolerance-s 0.005 --pts-offset-s 0]` writes `timestamp_audit.json` and includes PTS evidence hash in `run_config.json`. Mismatch stops before output creation.
- New gate regression tests include holdout blocked/pass, invalid PTS, duplicate frame times, strict output stop.

## Reproduce

```powershell
py -m pip install -e .
py -m unittest discover -s tests -v
py -m bevmotionlab analyze --calibration examples/synthetic_calibration.json --detections examples/synthetic_tracks.csv --camera-id synthetic_cam --require-quality-gate --out out/bev-strict
```

A real FFprobe JSON with frame entries may be supplied via `--ffprobe-json`. The bundled CSV is synthetic. This is numerical validity, **not field-calibrated speed accuracy**.

## Still BLOCKED before backend acceptance

1. Verify physical camera/lens provenance, timestamp source domain, distortion and camera-pose drift handling.
2. Attach uncertainty bands to every valid speed interval, including correlated calibration sensitivity; mark failure/non-identifiable geometry.
3. Define independent synthetic challenge profiles for occlusion, footpoint ambiguity, out-of-plane motion and speed discontinuities with locked thresholds.
4. Provide stable local backend service contract after domain/core checks, packaging/license review and reproducible CI artifacts.
5. Real video and surveyed-ground-speed acceptance is strictly **the final stage**, after frontend.

**Gate:** backend-in-progress; frontend HOLD; real-data HOLD.