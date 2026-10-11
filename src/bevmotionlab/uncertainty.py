"""Interval-specific uncertainty with shared homography perturbation.

Conditional coverage only: this model excludes systematic camera drift,
nonplanarity, incorrect footpoint semantics and unknown real-world noise.
"""
from __future__ import annotations
import hashlib
import math
import numpy as np
from .speed import monte_carlo_interval


def interval_uncertainties(sequences, fit_uv, fit_world, *, trials=200, seed=42,
                           pixel_std_px=.5, landmark_std_px=.5, timestamp_std_s=0.,
                           max_gap_s=2.):
    if not isinstance(trials, int) or not 10 <= trials <= 5000:
        raise ValueError('UNCERTAINTY_TRIALS_INVALID')
    for label,value in [('pixel_std_px',pixel_std_px),('landmark_std_px',landmark_std_px),
                        ('timestamp_std_s',timestamp_std_s),('max_gap_s',max_gap_s)]:
        if not math.isfinite(value) or value < 0 or (label=='max_gap_s' and value==0):
            raise ValueError('UNCERTAINTY_PARAMETER_INVALID: '+label)
    results=[]
    for (track_id,segment), seq in sorted(sequences.items()):
        for i in range(1,len(seq)):
            a,b=seq[i-1],seq[i]
            dt=float(b[0]-a[0])
            row={'track_id':track_id,'segment':segment,'start_s':a[0], 'end_s':b[0],
                 'status':'VALID','median_kmh':None,'p025_kmh':None,
                 'p975_kmh':None,'valid_trials':0,'failed_trials':0}
            if dt<=0 or dt>max_gap_s:
                row['status']='GAP' if dt>max_gap_s else 'INVALID_TIMESTAMP'
            else:
                # Hash is stable across Python hash-randomization and OS processes.
                token=f'{seed}|{track_id}|{segment}|{i}|{a[0]:.9f}|{b[0]:.9f}'
                local_seed=int.from_bytes(hashlib.sha256(token.encode()).digest()[:8],'big')
                try:
                    outcome=monte_carlo_interval(fit_pixel_px=fit_uv, fit_world_m=fit_world,
                        trajectory_pixel_px=np.asarray([a[2],b[2]]),times_s=[a[0],b[0]],
                        pixel_std=pixel_std_px,landmark_std_px=landmark_std_px,
                        timestamp_std_s=timestamp_std_s,seed=local_seed,trials=trials)
                    for key in ('median_kmh','p025_kmh','p975_kmh','valid_trials','failed_trials'):
                        row[key]=outcome[key]
                except (ValueError,np.linalg.LinAlgError):
                    row['status']='UNCERTAINTY_NOT_IDENTIFIABLE'
            results.append(row)
    return results