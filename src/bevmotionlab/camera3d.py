"""Opt-in pinhole/Brown-Conrady camera-to-ground safety model.

Camera transform is X_camera = R_cw @ (X_world - C_world).
Ground plane is Z_world=0, metres. Raw pixels are undistorted numerically
before ray-plane intersection. This is a *geometric consistency checker*, not
an estimator of human feet or a claim that every detected point lies on Z=0.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
import numpy as np


COEFFICIENTS = ('k1', 'k2', 'p1', 'p2', 'k3')


def _finite_array(value, shape, name):
    arr = np.asarray(value, dtype=float)
    if arr.shape != shape or not np.isfinite(arr).all():
        raise ValueError(f'CAMERA3D_INVALID_{name.upper()}')
    return arr


def validate_camera(model):
    """Require measured intrinsics, extrinsics and explicit lens provenance."""
    if not isinstance(model, Mapping) or model.get('schema_version') != '1.0':
        raise ValueError('CAMERA3D_INVALID_SCHEMA')
    if not isinstance(model.get('lens_calibration_id'), str) or not model['lens_calibration_id']:
        raise ValueError('CAMERA3D_MISSING_LENS_PROVENANCE')
    if model.get('world_units') != 'm' or model.get('plane_z_m') != 0:
        raise ValueError('CAMERA3D_GROUND_PLANE_REQUIRED')
    if model.get('pixel_convention') not in ('undistorted', 'raw_distorted'):
        raise ValueError('CAMERA3D_PIXEL_CONVENTION_REQUIRED')
    K = _finite_array(model.get('K'), (3, 3), 'intrinsics')
    if not np.allclose(K[2], [0,0,1], atol=1e-12) or K[0,0] <= 0 or K[1,1] <= 0 or abs(K[1,0]) > 1e-12:
        raise ValueError('CAMERA3D_INVALID_INTRINSICS')
    R = _finite_array(model.get('R_cw'), (3,3), 'rotation')
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-7) or abs(np.linalg.det(R)-1) > 1e-7:
        raise ValueError('CAMERA3D_INVALID_ROTATION')
    C = _finite_array(model.get('camera_center_world_m'), (3,), 'center')
    if C[2] <= 0:
        raise ValueError('CAMERA3D_CAMERA_MUST_BE_ABOVE_GROUND')
    dist = model.get('distortion')
    if not isinstance(dist, Mapping) or set(dist) != set(COEFFICIENTS):
        raise ValueError('CAMERA3D_INVALID_DISTORTION')
    params = _finite_array([dist[k] for k in COEFFICIENTS], (5,), 'distortion')
    if model['pixel_convention'] == 'undistorted' and not np.allclose(params, 0):
        # Model may *contain* lens distortion, but do not silently apply it to
        # pixels explicitly marked as already undistorted.
        pass
    return K, R, C, params


def _distort(xy, d):
    x, y = xy
    k1,k2,p1,p2,k3 = d
    r2=x*x+y*y
    radial = 1+k1*r2+k2*r2*r2+k3*r2*r2*r2
    return np.array([x*radial+2*p1*x*y+p2*(r2+2*x*x),
                     y*radial+p1*(r2+2*y*y)+2*p2*x*y], dtype=float)


def _undistort_normalized(raw, coefficients):
    candidate=np.array(raw, dtype=float)
    for _ in range(40):
        estimated=_distort(candidate, coefficients)
        residual=np.asarray(raw)-estimated
        if np.linalg.norm(residual) <= 1e-11:
            return candidate
        # Newton step in 2D; finite differences are stable at measured pixel
        # magnitudes and reject non-invertible extreme radial models.
        delta=1e-6
        J=np.column_stack([(_distort(candidate+np.eye(2)[i]*delta,coefficients)-
                            _distort(candidate-np.eye(2)[i]*delta,coefficients))/(2*delta)
                           for i in range(2)])
        try:step=np.linalg.solve(J,residual)
        except np.linalg.LinAlgError as e:raise ValueError('CAMERA3D_NONINVERTIBLE_LENS') from e
        candidate+=step
        if not np.isfinite(candidate).all() or np.linalg.norm(candidate)>5:
            raise ValueError('CAMERA3D_LENS_INVERSION_DIVERGED')
    raise ValueError('CAMERA3D_LENS_INVERSION_NOT_CONVERGED')


def intersect_ground(model, pixels):
    """Return N-by-2 metric ground points; reject rays behind or on horizon."""
    K,R,C,d=validate_camera(model)
    uv=np.asarray(pixels,dtype=float)
    if uv.ndim!=2 or uv.shape[1]!=2 or not np.isfinite(uv).all():
        raise ValueError('CAMERA3D_INVALID_PIXELS')
    points=[]
    for u,v in uv:
        ray=np.linalg.solve(K,np.array([u,v,1.]))
        if model['pixel_convention']=='raw_distorted':
            ray[:2]=_undistort_normalized(ray[:2],d)
        world_ray=R.T @ ray
        if world_ray[2]>=-1e-9:
            raise ValueError('CAMERA3D_RAY_DOES_NOT_HIT_GROUND_FORWARD')
        travel=-C[2]/world_ray[2]
        if travel<=0 or not math.isfinite(travel):
            raise ValueError('CAMERA3D_INVALID_RAY_INTERSECTION')
        xy=C[:2]+travel*world_ray[:2]
        if not np.isfinite(xy).all() or np.linalg.norm(xy)>1e6:
            raise ValueError('CAMERA3D_UNBOUNDED_PROJECTION')
        points.append(xy)
    return np.asarray(points).reshape(-1,2)


def audit_ground_model(model, *, pixel_points, planar_world_points,
                       max_residual_m=0.15, min_points=4):
    """Independent metric ground marks must agree with 3-D ray-plane geometry."""
    if isinstance(max_residual_m,bool) or not isinstance(max_residual_m,(int,float)) or not math.isfinite(max_residual_m) or max_residual_m<=0:
        raise ValueError('CAMERA3D_INVALID_THRESHOLD')
    uv=np.asarray(pixel_points,dtype=float)
    world=np.asarray(planar_world_points,dtype=float)
    if uv.ndim!=2 or uv.shape[1]!=2 or world.shape!=uv.shape or len(uv)<min_points or min_points<4:
        raise ValueError('CAMERA3D_INSUFFICIENT_CORRESPONDENCES')
    if not np.isfinite(world).all() or np.linalg.matrix_rank(world-world.mean(axis=0))<2:
        raise ValueError('CAMERA3D_DEGENERATE_WORLD_MARKERS')
    measured=intersect_ground(model,uv)
    errors=np.linalg.norm(measured-world,axis=1)
    rms=float(np.sqrt(np.mean(errors**2)))
    maximum=float(np.max(errors))
    return {'schema_version':'1.0','status':'GEOMETRY_CONSISTENT' if maximum<=max_residual_m else 'BLOCKED',
            'passed':maximum<=max_residual_m,'rmse_m':rms,'max_residual_m':maximum,
            'threshold_m':float(max_residual_m),'count':len(uv),
            'limitation':'Reference markers assumed on flat Z=0; out-of-plane target feet remain unknown.'}