"""Conditional 3D contact-height error bound, never a field accuracy claim."""
import math
import numpy as np
from .camera3d import validate_camera, _undistort_normalized

def _world_ray(model, pixel):
    K,R,C,d=validate_camera(model)
    uv=np.asarray(pixel,dtype=float)
    if uv.shape!=(2,) or not np.isfinite(uv).all():raise ValueError('HEIGHT_RISK_INVALID_PIXEL')
    ray=np.linalg.solve(K,np.array([*uv,1.]))
    if model['pixel_convention']=='raw_distorted':
        ray[:2]=_undistort_normalized(ray[:2],d)
    world=R.T@ray
    if not np.isfinite(world).all() or world[2]>=-1e-9:
        raise ValueError('HEIGHT_RISK_RAY_NO_FORWARD_GROUND_INTERSECTION')
    return world,C

def ground_bias_bound(model,pixel,*,maximum_height_m):
    if type(maximum_height_m) not in (int,float) or not math.isfinite(maximum_height_m) or maximum_height_m<0:
        raise ValueError('HEIGHT_RISK_INVALID_MAX_HEIGHT')
    ray,C=_world_ray(model,pixel)
    if maximum_height_m>=C[2]:raise ValueError('HEIGHT_RISK_HEIGHT_REACHES_CAMERA')
    error=float(maximum_height_m*np.linalg.norm(ray[:2])/abs(ray[2]))
    if not math.isfinite(error) or error>1e6:raise ValueError('HEIGHT_RISK_UNBOUNDED')
    return {'schema_version':'1.0','maximum_height_m':float(maximum_height_m),
            'xy_position_bias_upper_bound_m':error,'physical_height_measured':False}

def speed_bias_bound(model,pixel_start,pixel_end,*,elapsed_s,maximum_height_m):
    if type(elapsed_s) not in (int,float) or not math.isfinite(elapsed_s) or elapsed_s<=0:
        raise ValueError('HEIGHT_RISK_INVALID_INTERVAL')
    a=ground_bias_bound(model,pixel_start,maximum_height_m=maximum_height_m)
    b=ground_bias_bound(model,pixel_end,maximum_height_m=maximum_height_m)
    speed=3.6*(a['xy_position_bias_upper_bound_m']+b['xy_position_bias_upper_bound_m'])/elapsed_s
    if not math.isfinite(speed) or speed>1e8:raise ValueError('HEIGHT_RISK_SPEED_BOUND_TOO_LARGE')
    return {'schema_version':'1.0','maximum_height_m':float(maximum_height_m),
            'elapsed_s':float(elapsed_s),'speed_bias_upper_bound_kmh':float(speed),
            'start_position_bias_upper_m':a['xy_position_bias_upper_bound_m'],
            'end_position_bias_upper_m':b['xy_position_bias_upper_bound_m'],
            'status':'CONDITIONAL_BOUND_ONLY','real_data_validated':False}
