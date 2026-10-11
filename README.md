# BEV-MotionLab

**An offline-first camera-geometry laboratory for ground-plane projection, bird's-eye-view trajectories and uncertainty-aware physical speed estimation.**

> **Status (feature/backend-v0.1): runnable offline backend research build, not a validated real-camera speed-measurement system.** Synthetic CLI and unit tests are implemented. Field accuracy, final backend acceptance and frontend remain **HOLD**. See [`docs/BACKEND_V010.md`](docs/BACKEND_V010.md).

## 1. What problem does the project solve?

Motion in image pixels is not physical motion in metres. A perspective camera introduces depth-dependent pixel scale: a displacement of 20 pixels near the horizon is not equivalent to 20 pixels in the foreground. BEV-MotionLab projects ground-contact points from a **calibrated, approximately planar ground surface** into a metric world frame and estimates speed from **real timestamps**, while explicitly communicating when the result is unreliable.

The engineering questions are:

1. Under which calibration and scene assumptions can pixel tracks be converted into metric trajectories?
2. How can incorrect point correspondences, poor geometric conditioning and camera changes be detected rather than silently accepted?
3. How much do contact-point error, homography uncertainty and timestamp jitter distort speed?
4. How should calibration-fit and independent holdout measurements be separated?
5. When should a speed measurement be **withheld** instead of outputting misleading precision?

## 2. Scope and non-goals

**In scope:** planar projective geometry, normalized DLT, robust homography fitting, world-plane projection, bottom-centre contact approximation, track timestamp handling, metre/second and km/h conversion, uncertainty intervals, calibrated/uncalibrated status flags, deterministic synthetic experiments, BEV/plot reporting, Windows-first offline CLI.

**Not in v1.0:** automatic camera intrinsic calibration, universal single-image metric reconstruction, recovering arbitrary 3D motion, vehicle detector training, monocular absolute scale without measured references, real-time traffic enforcement, certified legal evidence, traffic-control decisions or support for moving/zooming cameras without recalibration.

**Core assumption:** a fixed camera observes targets moving on one measured planar ground surface. A calibrated mapping is valid only in its defined spatial region and camera configuration. A linearly moving object above the plane (e.g. a flying drone) violates this model.

## 3. Scientific and geometric foundation

For a planar scene, a point in undistorted image coordinates and its metric ground-plane coordinates are related by a homography:

```text
lambda * [u, v, 1]^T = H_world_to_image * [X_m, Y_m, 1]^T
[X_m, Y_m, 1]^T ~ inverse(H_world_to_image) * [u, v, 1]^T
```

- Fit `H_world_to_image` from measured world-point ↔ image-point correspondences and invert for pixel-to-world use. State the transform direction in every API.
- The coordinate frame must define origin, x/y orientation, metres, image resolution and whether pixels were lens-undistorted.
- Use **normalized DLT** with at least four non-collinear **inlier** correspondences. Four supplied points with one outlier cannot provide four correct inliers; a robust estimator needs sufficient additional points.
- For OpenCV `findHomography(world_xy, image_uv, RANSAC, threshold_px)`, the RANSAC residual threshold is expressed in **destination image pixels**. Passing `image_uv -> world_xy` with `threshold_px=3` mistakenly interprets 3 as **world units**. This project deliberately fits world → image for pixel-based RANSAC thresholds.
- A camera lens-distortion correction must be done consistently on both calibration points and detection contact points; never combine raw distorted pixels with an undistorted homography.
- Homography `H` is scale-equivalent (`H` and `cH` describe the same mapping). Raw determinant or condition-number thresholds without normalization/scaling context are **not** universally meaningful quality criteria.

### Speed equation

For world-plane contact points `P(t) = [X_m(t), Y_m(t)]` and positive elapsed time:

```text
speed_mps = norm(P(t2) - P(t1)) / (t2_s - t1_s)
speed_kmh = 3.6 * speed_mps
```

Instantaneous two-frame differences amplify location noise. Smoothing and finite-difference window length are explicit, versioned configuration choices that trade bias against variance. A physical output is rejected when timestamp, projection or calibration quality is insufficient.

## 4. Target system architecture

```text
Measured calibration points (world metres ↔ image pixels)
                         |
                         v
        geometry validation + lens convention
                         |
                         v
   normalized DLT / pixel-domain RANSAC / holdout
                         |
                         v
       H + calibration provenance + validity
                         |
                         +--------------------------+
                         |                          |
Timestamped bboxes/points (CSV)                 Quality audit
                         |
                         v
   contact-point extraction (with parallax flag)
                         |
                         v
    undistortion (only if calibrated/configured)
                         |
                         v
        pixel-to-world ground projection
                         |
                         v
      timestamp-aware temporal aggregation
                         |
                         v
      speed estimate + uncertainty + flags
                         |
                +--------+--------+
                |                 |
         trajectory data    BEV / speed plots
```

This separation allows the geometry module to be validated independently from detections, video codecs, tracking or UI tools.

## 5. Proposed module and source contracts

| Target module | Responsibility | Input → Output |
|---|---|---|
| `bevmotionlab/io/schema.py` | Validate schema, units, metadata | JSON/CSV → typed records or diagnostic failure |
| `bevmotionlab/geometry/normalize.py` | Hartley normalization and degeneracy checks | Point pairs → normalized systems |
| `bevmotionlab/geometry/homography.py` | DLT, RANSAC and inverse transform | World/image pairs → H, inlier mask, diagnostics |
| `bevmotionlab/geometry/projection.py` | Perspective divide and domain validation | H, image points → world XY + validity |
| `bevmotionlab/geometry/quality.py` | Fit and independent holdout audit | H, split samples → calibration report |
| `bevmotionlab/detection/contact.py` | Bottom-centre contact approximation | Bounding box → `[u,v]` + uncertainty flag |
| `bevmotionlab/video/timestamps.py` | Optional FFprobe PTS ingestion | Video PTS → ordered time base |
| `bevmotionlab/speed/estimator.py` | Finite/central differences, smoothing | World-time series → speed/time series |
| `bevmotionlab/speed/uncertainty.py` | Pixel, calibration and timing uncertainty | Inputs/distributions → interval and failure flags |
| `bevmotionlab/validation/holdout.py` | Calibration independent evaluation | Holdout landmarks → error statistics |
| `bevmotionlab/viz/bev.py` | Ground plane trajectories and plots | Metric tracks → PNG/SVG |
| `bevmotionlab/cli.py` | Offline orchestration, config and reporting | CLI arguments → reproducible artifact directory |

Names and paths above are **design contracts**, not a claim of implemented files.

## 6. Data and API contracts

### 6.1 Calibration definition: `calibration.json`

```json
{
  "schema_version": "1.0-design",
  "camera_id": "cam_001",
  "resolution_px": [1920, 1080],
  "coordinate_frame": {"units": "m", "origin": "measured_marker_A", "x_axis": "road_forward", "y_axis": "road_left"},
  "pixel_convention": "undistorted",
  "lens_calibration_id": "lens_001",
  "correspondences": [
    {"id": "A", "world_xy_m": [0.0, 0.0], "image_uv_px": [440.0, 880.0], "split": "fit"},
    {"id": "B", "world_xy_m": [10.0, 0.0], "image_uv_px": [1300.0, 900.0], "split": "fit"},
    {"id": "C", "world_xy_m": [10.0, 6.0], "image_uv_px": [1150.0, 550.0], "split": "fit"},
    {"id": "D", "world_xy_m": [0.0, 6.0], "image_uv_px": [550.0, 540.0], "split": "fit"},
    {"id": "E", "world_xy_m": [5.0, 3.0], "image_uv_px": [910.0, 700.0], "split": "holdout"}
  ],
  "valid_roi_uv_px": [[300, 400], [1600, 1000]],
  "measurement_provenance": "surveyed_ground_markers",
  "timestamp_source": "video_pts",
  "calibrated_at_utc": "2026-10-11T12:00:00Z"
}
```

**Illustrative, invented fixture geometry:** these numbers show schema structure, **not** a physically validated calibration. Real calibration requires actual measured world/image correspondences and a camera-specific fit. `lens_calibration_id` must refer to matching, available correction parameters, otherwise reject a declared-undistorted claim.

Calibration quality uses at least 4 non-collinear fit inliers, additional holdout points for independent evidence, 2D coverage of the intended ROI and constrained pixel residuals. Report fit/holdout errors independently.

### 6.2 Tracked detection inputs

```csv
frame,timestamp_s,track_id,x1_px,y1_px,x2_px,y2_px,confidence
0,0.000,7,800,450,900,700,0.96
1,0.033,7,806,451,907,703,0.96
```

- `timestamp_s` is the acquisition/presentation timestamp, not simply `frame / nominal_fps` when VFR is possible.
- A `track_id` is a **source tracker identity**, whose switches can corrupt speed; discontinuities must be flagged rather than smoothed across blindly.
- Box coordinates are validated against resolution and box orientation; bottom-centre is only a **proxy** for actual ground contact and can be wrong under occlusion or truncation.
- If per-track world positions are supplied instead of boxes, record the provenance and bypass contact extraction explicitly.
- `timestamp_s` must be strictly increasing **within a track**; gaps/irregular frame periods are preserved. No hidden frame interpolation.

### 6.3 Outputs

```text
runs/<run-id>/
├── calibration_report.json    # fit/holdout, degeneracy, model metadata
├── projected_tracks.csv      # X_m, Y_m, validity, uncertainty and source track
├── speed_timeseries.csv      # mps, kmh, interval, quality and method
├── flags.jsonl               # stale/mismatch/OOB/parallax/gap/invalid PTS
├── experiment_config.json    # seed, split, method, software versions
├── bev_trajectory.svg        # map of valid metric positions
└── manifest.json             # input provenance / SHA256
```

Possible output statuses: `VALID`, `LOW_CONFIDENCE`, `OUTSIDE_CALIBRATED_ROI`, `INVALID_TIMESTAMP`, `INVALID_HOMOGRAPHY`, `CONTACT_UNCERTAIN`, `STALE_CALIBRATION`, `CAMERA_MISMATCH`. An invalid speed is **null with an explicit cause**, never silently zero.

## 7. Calibration safety and quality gates

Before speed estimation, all of the following gates must be recorded:

1. **Geometric validity:** at least four well-spread non-collinear inliers; reject repeated points, near-degenerate configurations and near-zero homogeneous denominators.
2. **Residual units:** pixel reprojection residual for `world → image`, holdout world RMSE in metres for inverse projection; do not mix units under one label.
3. **Independent evidence:** holdout markers not used to fit or tune RANSAC parameters; include spatial error by ROI/subregion and image-to-world residuals.
4. **Camera validity:** camera ID, image resolution, focal setting, camera pose, distortion convention and calibration fingerprint agree with observations.
5. **Temporal validity:** meaningful monotonic presentation timestamps, positive time deltas, acceptable jitter and proper recording of missing frames.
6. **Contact validity:** box visibility, ground support assumption, truncation or occlusion conditions explicitly identified.
7. **Numeric stability:** report point spread, singular-value diagnostics and uncertainty; no context-free universal matrix condition threshold.

Reject or warn according to an explicit severity policy. A camera moved after calibration must be recalibrated even if the previous calibration is recent; age alone does not prove invalidity.

## 8. Uncertainty estimation

Use a transparent, reproducible uncertainty model separating at least:

- Contact-point error (e.g. calibrated pixel-noise covariance; bbox foot approximation).
- Calibration-point measurement error (fit resampling/Monte Carlo; coherent recalculation of H from sampled point sets rather than blindly perturbing arbitrary matrix entries).
- Timestamp jitter/quantization and irregular sampling.
- Scene-model bias: off-plane point, rolling shutter, lens calibration mismatch, camera shake and false `track_id` continuity.

For each valid speed interval report median/point estimate, interval bounds, simulation seed, trials and assumptions. Pure Monte Carlo dispersion **does not** capture unknown systematic camera-model bias. No coverage guarantee without evidence. Physical precision should match the model's uncertainty.

## 9. Evaluation scenarios and metrics

| Scenario ID | Controlled case | Required observation |
|---|---|---|
| BEV-01 | Noise-free synthetic planar H | Geometric parameter recovery and inverse round-trip |
| BEV-02 | At least 5 fit candidates including an outlier | RANSAC finds ≥4 valid inliers; outlier excluded |
| BEV-03 | Different spatial landmark spreads | Ill-conditioning/degeneracy rejected with reason |
| BEV-04 | 10 m in 1 s synthetic physical motion | Nominal 36 km/h before perturbations |
| BEV-05 | Irregular VFR timestamps | Speed uses PTS; nominal FPS shortcut is not accepted |
| BEV-06 | Camera shift / changed resolution / distortion mismatch | Calibration invalidity surfaced |
| BEV-07 | Object off-plane / box foot truncated | Error flag and failure rather than fabricated measurement |
| BEV-08 | Holdout markers from unseen ROI | Fit/holdout leakage prevented; error profile reported |
| BEV-09 | Pixel jitter, varying dt, track-ID discontinuity | Confidence interval and jump handling traced |
| BEV-10 | Exact same synthetic configuration and seed | Byte-stable normalized reports where feasible |

Report `fit_reprojection_rmse_px`, `holdout_reprojection_rmse_px`, `holdout_world_rmse_m`, `speed_mae_kmh`, `speed_rmse_kmh`, `speed_bias_kmh`, interval empirical coverage, retained-track coverage and rejected-case fraction. **Target thresholds must be justified by ground-marker precision, camera resolution, ROI and use case**; a single universal `0.5 m` or `2 px` rule is not defensible for every setup.

## 10. Planned experiment protocol

- Publish a deterministic ground-plane simulator and fixed synthetic calibration prior to optimizations.
- Fit H on development correspondences; keep independent holdout locations and motion traces locked.
- Compare: (A) raw pixel displacement (dimensionally non-physical baseline), (B) correctly calibrated unsmoothed BEV, (C) temporally smoothed BEV, (D) uncertainty-gated BEV. Do not label A as km/h.
- Sweep camera viewpoint, ROI location, pixel noise, landmark placement, outliers, timestamps, track gaps and off-plane height.
- Display actual bad examples and invalid results. Show a trade-off curve of accepted measurement fraction versus speed error rather than claiming success after rejection.
- In the **final acceptance phase only**, measure ground markers and physical motion with an independently established reference method and documented rights/provenance.

## 11. Development roadmap

| Phase | System milestone | Exit criterion |
|---|---|---|
| B0 | Freeze I/O, world frame and camera assumptions | Schemas, ROI and coordinate conventions agreed |
| B1 | Normalized DLT, robust fit and projection | All transform directions/units explicit |
| B2 | Bounding-box contact and timestamp ingestion | Missing/irregular data flagged and preserved |
| B3 | Speed methods and metric outputs | m/s ↔ km/h and timing implementation coherent |
| B4 | Uncertainty and quality gates | Invalid geometry/calibration cannot produce false precision |
| B5 | Synthetic sweeps and visual reporting | Fit vs holdout, errors vs acceptance coverage documented |
| B6 | Windows/Linux packaging, CLI and CI | Reproducible local install and test matrix |
| B7 | Final independent physical validation | Quality decision supported by independently measured reference |

**Implementation ordering:** complete modules and interfaces first, then their integration and experiments. Release acceptance/independent real-data verification comes last. Essential small unit checks still belong alongside each implementation so incompatible modules do not accumulate silently.


### Architecture freeze — ADR-001 (2026-10-11)

**Delivery sequence is locked:** **(0) GitHub README and architecture freeze → (1) complete and harden the offline backend → (2) build and polish the frontend → (3) complete end-to-end synthetic/system acceptance → (4) perform independent real-data evaluation LAST.** No production/field claims before stage 4. A change to this sequence or the v1 system boundary requires an explicit, documented design revision; a README does not claim the target code already exists.

**Frozen cross-project implementation decisions:**

| Decision | Contract |
|---|---|
| Operating environment | Windows 10/11 first; Linux CI/secondary execution where practical; local CPU operation, no mandatory cloud, subscription, paid API or hosted inference |
| Backend | Python 3.11+ target for new modules; pure domain/application layer stays callable from CLI without an HTTP server; explicit typed schemas and versioned run artifacts |
| API adapter | Optional local FastAPI/Pydantic `v1` adapter only **after the core backend is functional**; bind to `127.0.0.1` by default, never expose the service to the LAN by default |
| Frontend | **Planned, not implemented:** React + TypeScript + Vite, locally bundled static assets, no CDN/telemetry, responsive clear dashboard; Turkish-first labels with traceable units, errors and provenance |
| Integration | Frontend consumes only the versioned local API; simulation/math never runs as an unreviewed duplicate implementation in browser code; shared CLI/API core |
| Storage | Local file-backed runs (`JSON`/`CSV`/`SVG`/`PNG`/`HTML` as appropriate), explicit schema versions, safe output paths and reproducibility manifest; no required database or network service |
| UI truthfulness | Never invent a result, fallback silently, or display a planned feature as operational; quality warnings and `not available` states must be visible |
| Testing cadence | Small unit/contract/integration checks accompany backend and UI development; broad system benchmarks and release acceptance happen after backend and frontend completion; **external real data only in the final evaluation stage** |

**Backend completion gate (before frontend development):** all in-scope domain modules, edge-case and lifecycle handling, reproducible synthetic input generation/replay, CLI, stable typed I/O, audit logging, error codes, API contract design, resource/safety bounds and representative automated integration checks are implemented. A green smoke test alone does not satisfy this gate.

**Frontend completion gate (before final acceptance):** complete navigation and workflows, understandable visualizations and units, responsive layout, form validation, explicit loading/empty/error/unreliable states, keyboard/accessibility checks, result drill-down and file export, full offline operation with local API, and correct display of backend-reported evidence. The UI must never determine scientific validity independently of backend flags.

**Final acceptance gate:** perform integrated synthetic replay, locked comparisons, failure/stress and reproducibility tests on Windows, plus cross-platform checks when relevant. **Only after these steps** introduce independently sourced, permissioned real observations and evaluate quality without silently tuning to the final test set. For the QoD simulator, a real trace is *offline replay evidence*, not proof of live operator QoD gains.

### BEV-MotionLab frontend contract (planned)

The local interface will provide a **calibration workspace** (pixel/world pairs, validity region and fit/holdout split), **projection workspace** (contact points and ground-plane trajectories), **speed workspace** (time-aligned speed curves, uncertainty and quality flags), and a **report workspace** (fit error, rejected intervals and reproducibility export). It must visually distinguish measured, estimated and unavailable values. Calibration and speed computations remain backend responsibilities; visual marks alone cannot certify a camera.

The final real-data gate uses measured ground references and timestamped real tracks only after the synthetic backend/UI workflow is complete. Real-video importing must be permissioned and optional, not required for local development.

## 12. Proposed repository structure

```text
BEV-MotionLab/
├── README.md
├── pyproject.toml               # planned, not yet present
├── src/bevmotionlab/
│   ├── io/
│   ├── geometry/
│   ├── detection/
│   ├── video/
│   ├── speed/
│   ├── validation/
│   ├── viz/
│   └── cli.py
├── tests/
├── frontend/            # planned React/TypeScript/Vite offline UI
├── examples/
├── data/synthetic/
├── experiments/
├── docs/
└── .github/workflows/ci.yml
```

## 13. Working backend v0.1 CLI (synthetic-only)

```powershell
py -m pip install -e .
py -m unittest discover -s tests -v
py -m bevmotionlab calibrate --calibration examples/synthetic_calibration.json --out runs/calibration.json
py -m bevmotionlab analyze --calibration examples/synthetic_calibration.json --detections examples/synthetic_tracks.csv --camera-id synthetic_cam --out runs/demo
```

These commands have run against **synthetic** inputs. Their existence does **not** establish actual calibrated-camera accuracy. Raw video reading, full uncertainty per interval, lens correction and independently measured field validation remain future work.

## 14. Technical stack and deployment policy

**Proposed:** Python ≥3.11, NumPy, OpenCV for homography/RANSAC (optional if a NumPy-only reference implementation is supplied), SciPy for optional smoothing, Matplotlib/SVG for visual reports, FFprobe as an optional locally installed dependency for PTS extraction. Exact versions are selected and pinned during implementation. No GPU, login, paid API or cloud endpoint required. Operate on Windows 10/11 first; validate Linux in CI.

Prevent malformed file paths, uncontrolled processing sizes and untrusted deserialization. Do not embed video files or personal data in example fixtures without explicit rights. Publish only redistributable imagery and calibration records.

**Licensing:** repository license and dependency redistribution rights are **not selected/verified** at README stage. No open-source license is assumed until a `LICENSE` file is explicitly approved.

## 15. References

- [OpenCV homography estimation](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) — estimator direction and pixel-domain residual convention
- [OpenCV perspective transformations](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html)
- [FFmpeg / FFprobe documentation](https://ffmpeg.org/ffprobe.html)

---

**Claim discipline:** this is a metric-ground-plane measurement research tool, not a universal single-view 3D measurement system. Architecture is specified; implementation, validation and accuracy evidence are pending.