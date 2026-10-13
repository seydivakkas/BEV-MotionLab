"""Version 1.0 in-process numerical API; no arbitrary file access."""
from __future__ import annotations
import numpy as np
from .geometry import fit_dlt, project, holdout_report
from .quality import assess_calibration
from .speed import estimate_speed
from .loopback import create_server
from .motion_module import BEVMotionModule


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



def _project_tracks(input_data):
    """Stateless bounded API call; calibration/data provenance are required.

    v1.0 original calibrate/project_speed operations are unchanged. This
    extension explicitly avoids file paths and untrusted side effects.
    """
    if not isinstance(input_data,dict) or set(input_data)-{'calibration','samples','limits','ransac','seed'} or not {'calibration','samples'}.issubset(input_data):
        raise ValueError('INVALID_BEVMOTION_INPUT_SCHEMA')
    calibration=input_data['calibration']
    if not isinstance(calibration,dict):raise ValueError('CALIBRATION_NOT_OBJECT')
    allowed_cal={'fit_pixel_px','fit_world_m','holdout_pixel_px','holdout_world_m',
                 'camera_id','lens_calibration_id','measurement_provenance',
                 'pixel_convention','pixel_roi'}
    required_cal=allowed_cal-{'pixel_roi'}
    if required_cal-set(calibration) or set(calibration)-allowed_cal:
        raise ValueError('INVALID_CALIBRATION_SCHEMA')
    samples=input_data['samples']
    if not isinstance(samples,list) or not 1<=len(samples)<=2000:
        raise ValueError('SAMPLE_CARDINALITY_LIMIT')
    limits=input_data.get('limits',{})
    allow={'max_major_axis_m','max_gap_s','max_speed_kmh','max_speed_sigma_kmh'}
    if not isinstance(limits,dict) or set(limits)-allow:
        raise ValueError('INVALID_PROJECT_TRACK_LIMITS')
    ransac=input_data.get('ransac',False)
    if type(ransac) is not bool:raise ValueError('INVALID_RANSAC_FLAG')
    seed=input_data.get('seed',42)
    if type(seed) is not int or not 0<=seed<=2147483647:raise ValueError('INVALID_SEED')
    module=BEVMotionModule.from_correspondences(**calibration,**limits,
                                                ransac=ransac,seed=seed)
    rows=[module.process_sample(item) for item in samples]
    return {'schema_version':'1.0','operation':'project_tracks',
            'calibration':{'camera_id':module.evidence.camera_id,
                           'lens_calibration_id':module.evidence.lens_calibration_id,
                           'pixel_roi':module.evidence.pixel_roi,
                           'quality_gate':module.evidence.numerical_quality_gate,
                           'H_pixel_to_world':module.H.tolist()},
            'records':rows,
            'counts':{'samples':len(rows),'conditional_valid_speed_intervals':sum(bool(r.get('speed_valid')) for r in rows),
                      'blocked_or_initializing_samples':sum(not bool(r.get('speed_valid')) for r in rows)},
            'claim':'SYNTHETIC_FIRST_CONDITIONAL_NUMERICAL_NOT_FIELD_VALIDATED',
            'limits':'Pixel Jacobian first-order and declared contact; excludes unknown systematic camera/lens/contact bias. GT/real field accuracy not claimed.'}


def dispatch(request):
    if not isinstance(request,dict) or set(request)!= {'schema_version','operation','input'}:
        raise ValueError('exact JSON API envelope required')
    if request['schema_version']!='1.0' or request['operation'] not in ('calibrate','project_speed','project_tracks'):
        raise ValueError('unsupported API operation/version')
    inp=request['input']
    if request['operation']=='project_tracks':
        return _project_tracks(inp)
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
