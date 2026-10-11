"""Version 1.0 in-process numerical API; no arbitrary file access."""
from __future__ import annotations
import numpy as np
from .geometry import fit_dlt, project, holdout_report
from .quality import assess_calibration
from .speed import estimate_speed
from .loopback import create_server


def _calibrate(data):
    fit_px=data['fit_pixel_px'];fit_m=data['fit_world_m']
    holds_px=data['holdout_pixel_px'];holds_m=data['holdout_world_m']
    if len(fit_px)>500 or len(holds_px)>500 or len(holds_px)==0:
        raise ValueError('calibration cardinality outside supported range')
    H=fit_dlt(fit_px,fit_m)
    holdout=holdout_report(H,holds_px,holds_m)
    fit_report=holdout_report(H,fit_px,fit_m)
    gate=assess_calibration({'fit_count':len(fit_px),'inlier_count':len(fit_px),
                              'fit_rmse_px':fit_report['reprojection_rmse_px'],'holdout':holdout})
    return H,gate


def dispatch(request):
    if not isinstance(request,dict) or set(request)!= {'schema_version','operation','input'}:
        raise ValueError('exact JSON API envelope required')
    if request['schema_version']!='1.0' or request['operation'] not in ('calibrate','project_speed'):
        raise ValueError('unsupported API operation/version')
    inp=request['input']
    if not isinstance(inp,dict): raise ValueError('input must be an object')
    required={'fit_pixel_px','fit_world_m','holdout_pixel_px','holdout_world_m'}
    allowed=required|({'trajectory_pixel_px','timestamps_s','max_gap_s'} if request['operation']=='project_speed' else set())
    if not required.issubset(inp) or set(inp)-allowed:raise ValueError('unexpected or absent fields')
    H,gate=_calibrate(inp)
    result={'quality_gate':gate,'H_pixel_to_world':H.tolist()}
    if request['operation']=='project_speed':
        if not gate['passed']:raise ValueError('calibration must pass independent numerical holdout')
        px=np.asarray(inp['trajectory_pixel_px'],float)
        if px.ndim!=2 or px.shape[1:]!=(2,) or not 2<=len(px)<=5000:
            raise ValueError('trajectory must be 2..5000 points')
        world=project(H,px)
        speeds=estimate_speed(inp['timestamps_s'],world,max_gap_s=inp.get('max_gap_s',2.))
        result.update({'projected_world_xy_m':world.tolist(),'speed_intervals':speeds})
    return result


def server(*,token,port=0):
    return create_server(dispatch,token=token,port=port)