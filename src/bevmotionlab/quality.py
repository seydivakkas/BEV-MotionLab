"""Fail-closed *numerical* calibration gate; never a substitute for field validation."""
from __future__ import annotations
import math


def assess_calibration(report: dict, *, max_fit_rmse_px: float = 2.,
                       max_holdout_rmse_m: float = .5,
                       max_holdout_reprojection_px: float = 4.,
                       min_inlier_fraction: float = .75) -> dict:
    thresholds = (max_fit_rmse_px, max_holdout_rmse_m,
                  max_holdout_reprojection_px, min_inlier_fraction)
    if any(not math.isfinite(x) or x <= 0 for x in thresholds) or min_inlier_fraction > 1:
        raise ValueError('calibration thresholds must be positive finite; inlier fraction <= 1')
    failures = []
    count, inliers = report['fit_count'], report['inlier_count']
    fit_error = report['fit_rmse_px']
    if count < 4 or inliers < 4 or inliers / count < min_inlier_fraction:
        failures.append('INSUFFICIENT_INLIER_SUPPORT')
    if not math.isfinite(fit_error) or fit_error > max_fit_rmse_px:
        failures.append('FIT_REPROJECTION_TOO_HIGH')
    holdout = report.get('holdout')
    if holdout is None or holdout.get('n', 0) == 0:
        failures.append('NO_INDEPENDENT_HOLDOUT')
    else:
        if not math.isfinite(holdout['world_rmse_m']) or holdout['world_rmse_m'] > max_holdout_rmse_m:
            failures.append('HOLDOUT_WORLD_ERROR_TOO_HIGH')
        if (not math.isfinite(holdout['reprojection_rmse_px'])
                or holdout['reprojection_rmse_px'] > max_holdout_reprojection_px):
            failures.append('HOLDOUT_PIXEL_ERROR_TOO_HIGH')
    return {'status': 'NUMERICAL_PASS' if not failures else 'BLOCKED',
            'passed': not failures, 'reasons': failures,
            'thresholds': {'max_fit_rmse_px': max_fit_rmse_px,
                           'max_holdout_rmse_m': max_holdout_rmse_m,
                           'max_holdout_reprojection_px': max_holdout_reprojection_px,
                           'min_inlier_fraction': min_inlier_fraction},
            'interpretation': 'Numerical checks only. Not a physical speed or field-calibration certificate.'}