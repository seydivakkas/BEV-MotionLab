"""Fail-closed conditional off-plane bias validation for *every* speed interval.

A declared bound on contact point height is an assumption, not an observation.
This audit is paired by (track, segment, start_s, end_s), not by row order.
It does not replace unknown systematic camera/lens pose uncertainty.
"""
from __future__ import annotations

import math
from .height_risk import speed_bias_bound


def audit_speed_intervals(model, history, rows, *, maximum_height_m,
                          max_bias_kmh):
    if type(max_bias_kmh) not in (float, int) or not math.isfinite(max_bias_kmh) or max_bias_kmh < 0:
        raise ValueError('INTERVAL_VALIDITY_INVALID_BIAS_THRESHOLD')
    expected = {}
    for (track, segment), sequence in history.items():
        for before, after in zip(sequence, sequence[1:]):
            first, second = before[0], after[0]
            key = (track, segment, float(first), float(second))
            if key in expected:
                raise ValueError('INTERVAL_VALIDITY_DUPLICATE_HISTORY')
            expected[key] = (before[2], after[2])
    if len(expected) != len(rows):
        raise ValueError('INTERVAL_VALIDITY_UNMATCHED_INTERVALS')
    seen = set()
    audit = []
    for row in rows:
        key = (row['track_id'],row['segment'],float(row['start_s']),float(row['end_s']))
        if key not in expected or key in seen:
            raise ValueError('INTERVAL_VALIDITY_UNMATCHED_INTERVALS')
        seen.add(key)
        if row['status'] != 'VALID':
            audit.append({'track_id':row['track_id'],'segment':row['segment'],
                          'start_s':row['start_s'],'end_s':row['end_s'],
                          'status':'NOT_EVALUATED_INVALID_SPEED','speed_bias_upper_bound_kmh':None,
                          'speed_lower_bound_kmh':None,'speed_upper_bound_kmh':None})
            continue
        start_px,end_px=expected[key]
        result=speed_bias_bound(model,start_px,end_px,elapsed_s=row['dt_s'],
                                maximum_height_m=maximum_height_m)
        upper=result['speed_bias_upper_bound_kmh']
        central=row['speed_kmh']
        if central is None or not math.isfinite(float(central)) or central<0:
            raise ValueError('INTERVAL_VALIDITY_INVALID_SPEED_VALUE')
        audit.append({'track_id':row['track_id'],'segment':row['segment'],
                      'start_s':row['start_s'],'end_s':row['end_s'],
                      'status':'BOUND_ACCEPTED' if upper<=max_bias_kmh else 'BLOCKED_HEIGHT_BIAS',
                      'speed_bias_upper_bound_kmh':upper,
                      'speed_lower_bound_kmh':max(0.,float(central)-upper),
                      'speed_upper_bound_kmh':float(central)+upper})
    if seen != set(expected):
        raise ValueError('INTERVAL_VALIDITY_UNMATCHED_INTERVALS')
    blocked=[r for r in audit if r['status']=='BLOCKED_HEIGHT_BIAS']
    return {'schema_version':'1.0','status':'BLOCKED' if blocked else 'CONDITIONAL_PASS',
            'accepted':not blocked,'checked_valid_intervals':sum(r['status']!='NOT_EVALUATED_INVALID_SPEED' for r in audit),
            'blocked_intervals':len(blocked),'maximum_height_m':float(maximum_height_m),
            'max_bias_kmh':float(max_bias_kmh), 'rows':audit,
            'limitations':'Assumed height envelope; no measured target height or systemic lens/pose error; not a real-world accuracy guarantee.'}