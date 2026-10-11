"""Causal interval speed estimates, explicit invalid results and uncertainty."""
from __future__ import annotations
import math
import numpy as np
from .geometry import fit_dlt, project


def estimate_speed(times_s, world_xy_m, *, max_gap_s=2., max_speed_kmh=None):
    t=np.asarray(times_s,dtype=float)
    p=np.asarray(world_xy_m,dtype=float)
    if t.ndim!=1 or p.shape!=(len(t),2) or not np.isfinite(t).all() or not np.isfinite(p).all():
        raise ValueError('finite timestamps and matching Nx2 positions required')
    if len(t)<2: return []
    if np.any(np.diff(t)<=0): raise ValueError('per-track timestamps must increase strictly')
    if not math.isfinite(max_gap_s) or max_gap_s<=0: raise ValueError('max_gap_s must be positive')
    if max_speed_kmh is not None and (not math.isfinite(max_speed_kmh) or max_speed_kmh<=0):
        raise ValueError('invalid max_speed_kmh')
    out=[]
    for i in range(1,len(t)):
        dt=float(t[i]-t[i-1]); speed=float(np.linalg.norm(p[i]-p[i-1])/dt*3.6)
        reason='GAP' if dt>max_gap_s else ('SPEED_OUTLIER' if max_speed_kmh is not None and speed>max_speed_kmh else None)
        out.append({'start_s':float(t[i-1]),'end_s':float(t[i]),'dt_s':dt,
                    'speed_kmh':None if reason else speed,'status':reason or 'VALID'})
    return out


def monte_carlo_interval(*, fit_pixel_px, fit_world_m, trajectory_pixel_px, times_s,
                         pixel_std=0.5, landmark_std_px=0.5, timestamp_std_s=0.,
                         seed=42, trials=200):
    """Refit entire H for every perturbed landmark draw; cannot model off-plane bias."""
    if trials<10 or trials>5000 or any(not np.isfinite(v) or v<0 for v in (pixel_std,landmark_std_px,timestamp_std_s)):
        raise ValueError('invalid Monte Carlo inputs')
    rng=np.random.default_rng(seed)
    fit_uv=np.asarray(fit_pixel_px,float); fit_xy=np.asarray(fit_world_m,float)
    trajectory=np.asarray(trajectory_pixel_px,float); ts=np.asarray(times_s,float)
    if trajectory.shape!=(2,2) or ts.shape!=(2,) or not np.isfinite(trajectory).all() or not np.isfinite(ts).all() or ts[1]<=ts[0]:
        raise ValueError('need two valid trajectory samples at increasing times')
    samples=[]; failures=0
    for _ in range(trials):
        try:
            H=fit_dlt(fit_uv+rng.normal(0,landmark_std_px,fit_uv.shape),fit_xy)
            px=trajectory+rng.normal(0,pixel_std,trajectory.shape)
            t=ts+rng.normal(0,timestamp_std_s,2)
            if t[1]<=t[0]: raise ValueError('time inversion')
            w=project(H,px)
            samples.append(float(np.linalg.norm(w[1]-w[0])/(t[1]-t[0])*3.6))
        except (ValueError,np.linalg.LinAlgError): failures+=1
    if len(samples)<max(10,trials//2):
        raise ValueError('insufficient Monte Carlo valid samples')
    return {'median_kmh':float(np.median(samples)), 'p025_kmh':float(np.quantile(samples,.025)),
            'p975_kmh':float(np.quantile(samples,.975)), 'valid_trials':len(samples),
            'failed_trials':failures,'seed':seed, 'systematic_bias_included':False}