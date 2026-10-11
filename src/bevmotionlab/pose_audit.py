"""Fail-closed stationary-landmark camera pose audit in undistorted pixel space.

This detects image displacement of specified immobile reference landmarks; it
cannot prove a camera remained fixed between observations or that the ground is
planar. It is deliberately not an estimator of camera pose in 3D.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
import json
import numpy as np


def audit_pose(calibration: Mapping, observation: Mapping, *,
               max_rmse_px: float = 2.0, max_individual_px: float = 4.0,
               min_points: int = 4) -> dict:
    """Compare independent fixed landmarks by ID, rejecting missing evidence.

    Observation schema: camera_id, lens_calibration_id, resolution_px,
    pixel_convention='undistorted', points=[{id,image_uv_px}]. Every point must
    be a unique known calibration landmark. No interpolation or ID guessing.
    """
    for name, value in [('max_rmse_px', max_rmse_px),
                        ('max_individual_px', max_individual_px)]:
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'POSE_AUDIT_INVALID_THRESHOLD: {name}')
    if type(min_points) is not int or min_points < 4:
        raise ValueError('POSE_AUDIT_INVALID_MIN_POINTS')
    if not isinstance(calibration, Mapping) or not isinstance(observation, Mapping):
        raise ValueError('POSE_AUDIT_INVALID_DOCUMENT')
    for field in ('camera_id', 'lens_calibration_id', 'resolution_px', 'pixel_convention'):
        if field not in calibration or field not in observation or calibration[field] != observation[field]:
            raise ValueError(f'POSE_AUDIT_PROVENANCE_MISMATCH: {field}')
    if calibration['pixel_convention'] != 'undistorted':
        raise ValueError('POSE_AUDIT_REQUIRES_UNDISTORTED_PIXELS')
    reference = {}
    for marker in calibration.get('correspondences', []):
        key = marker.get('id')
        if not isinstance(key, str) or not key or key in reference:
            raise ValueError('POSE_AUDIT_INVALID_CALIBRATION_IDS')
        xy = np.asarray(marker.get('image_uv_px'), dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError('POSE_AUDIT_INVALID_CALIBRATION_POINTS')
        reference[key] = xy
    points = observation.get('points')
    if not isinstance(points, list):
        raise ValueError('POSE_AUDIT_MISSING_POINTS')
    seen = set()
    ref, current = [], []
    width, height = calibration['resolution_px']
    for point in points:
        if not isinstance(point, Mapping):
            raise ValueError('POSE_AUDIT_INVALID_POINT')
        key = point.get('id')
        if not isinstance(key, str) or key in seen or key not in reference:
            raise ValueError('POSE_AUDIT_UNKNOWN_OR_DUPLICATE_ID')
        seen.add(key)
        val = np.asarray(point.get('image_uv_px'), dtype=float)
        if val.shape != (2,) or not np.isfinite(val).all() or not (0 <= val[0] <= width and 0 <= val[1] <= height):
            raise ValueError('POSE_AUDIT_INVALID_IMAGE_POSITION')
        ref.append(reference[key]); current.append(val)
    if len(ref) < min_points:
        raise ValueError('POSE_AUDIT_INSUFFICIENT_STATIONARY_LANDMARKS')
    ref = np.asarray(ref)
    # Collinear points cannot constrain changes transverse to that line.
    sv = np.linalg.svd(ref - ref.mean(axis=0), compute_uv=False)
    if sv[0] <= 0 or sv[1] / sv[0] < 1e-5:
        raise ValueError('POSE_AUDIT_DEGENERATE_REFERENCE_GEOMETRY')
    errors = np.linalg.norm(np.asarray(current) - ref, axis=1)
    rmse = float(np.sqrt(np.mean(errors**2)))
    maximum = float(np.max(errors))
    reasons=[]
    if rmse > max_rmse_px: reasons.append('POSE_RMSE_EXCEEDS_LIMIT')
    if maximum > max_individual_px: reasons.append('POSE_LANDMARK_DISPLACEMENT_EXCEEDS_LIMIT')
    return {'schema_version': '1.0', 'status': 'NUMERICALLY_STABLE' if not reasons else 'BLOCKED',
            'passed': not reasons, 'reasons': reasons, 'matched_stationary_points':len(ref),
            'rmse_px':rmse, 'max_displacement_px':maximum,
            'thresholds':{'max_rmse_px':float(max_rmse_px),'max_individual_px':float(max_individual_px)},
            'scope':'Observational fiducial displacement only; not 3D pose or field accuracy certification.'}


def audit_pose_file(calibration: dict, path, **kwargs):
    return audit_pose(calibration, json.loads(Path(path).read_text(encoding='utf-8')), **kwargs)
