# Offline backend loopback API v1.0 (2026-10-11)

## Transport contract

- `GET /v1/health` and `POST /v1/execute` bind to **127.0.0.1 only**, never a public interface.
- Both require a secret bearer token (>=24 characters). For an actual launch, generate a high-entropy token using `secrets.token_urlsafe(32)` and pass it to `server(token=...)`. Do not embed it in source, query parameters or Git history.
- `Origin` requests are rejected to avoid cross-origin browser use. JSON media type is required on POST; payload size <=256 KiB. No arbitrary file paths are accepted in the request body and no remote service or cloud API is used.
- Envelope: `{"schema_version":"1.0","operation":"<operation>","input":{...}}`; successful response: `{"schema_version":"1.0","result":{...}}`.
- Unsupported contracts return an input-validation error. The service does not return internal stack traces, paths or bearer credentials.
- Library use: `from bevmotionlab.api import server; http=server(token=<GENERATED_TOKEN>, port=0); http.serve_forever()`. The caller owns process lifetime and shutdown. This is a *backend-only* transport; no frontend is being shipped.

## Security and acceptance limitations

- This is a local research API. No remote exposure, no TLS termination, no account management, no browser CORS access or public deployment is implemented.
- Authentication prevents untrusted local callers without the token, but cannot defend against a compromised local user/process or malware on the same machine.
- Request budgets do not yet include a global CPU deadline or rate limiting. Performance and denial-of-service stress are an **OPEN backend gate**.
- The domain operations are synthetic/mathematical prototypes. Neither HTTP success nor unit CI establishes field validity.
- Backend v1 **HOLD** until domain-specific worst-case/regression gates, packaging rights and full synthetic E2E are complete. Frontend follows backend; independent real data is **last**.

### Domain routes: calibrate, project_speed
`input` must carry `fit_pixel_px`, `fit_world_m`, `holdout_pixel_px`, `holdout_world_m` with matching N×2 coordinate lists. `project_speed` also takes `trajectory_pixel_px` and `timestamps_s`; numeric quality gate must pass first. This endpoint *only* verifies synthetic/numerical correspondences. It does not attest lens/camera geometry or physical speed accuracy.