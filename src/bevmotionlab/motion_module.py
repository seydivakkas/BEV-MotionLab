"""Fail-closed BEV ground-contact trajectory engine (synthetic-first).

Image coordinates MUST already be undistorted and ground-contact-labelled.
The analytical covariance is conditional on pixel uncertainty and the fitted
homography; it excludes systematic camera, lens, survey, and contact bias.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any
import numpy as np

from .geometry import fit_dlt, fit_ransac, holdout_report
from .quality import assess_calibration
from .covariance_projection import (project_pixel_covariance_to_bev,
                                    CHI2_2_95)


def compute_homography_dlt(src_pts, dst_pts):
    """Estimate pixel->metre homography using existing normalized SVD DLT."""
    return fit_dlt(src_pts, dst_pts)


def extract_error_ellipse_parameters(cov_m):
    """95% *semi*-major/minor axes in metres for a 2D Gaussian covariance."""
    p = np.asarray(cov_m, dtype=float)
    if (p.shape != (2, 2) or not np.isfinite(p).all()
            or not np.allclose(p, p.T, rtol=1e-9, atol=1e-10)):
        raise ValueError('INVALID_METRIC_COVARIANCE')
    values, vectors = np.linalg.eigh(p)
    if values[0] < -1e-10:
        raise ValueError('METRIC_COVARIANCE_NOT_PSD')
    major, minor = np.sqrt(CHI2_2_95 * np.maximum(values[::-1], 0))
    angle = float(np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1])))
    return {'semi_major_m': float(major), 'semi_minor_m': float(minor),
            'orientation_deg': angle, 'chi2_2d_95': float(CHI2_2_95),
            'ellipse_kind': '95pct_confidence_region_conditional_gaussian'}


def _safe_cov(data, code):
    value = np.asarray(data, dtype=float)
    if (value.shape != (2, 2) or not np.isfinite(value).all()
            or not np.allclose(value, value.T, atol=1e-9, rtol=1e-9)
            or np.linalg.eigvalsh(value).min() < -1e-9):
        raise ValueError(code)
    return (value + value.T) / 2


def calculate_physical_speed(before, after, *, cross_covariance_m2=None,
                             max_gap_s=0.5, max_speed_kmh=240.,
                             max_speed_sigma_kmh=35., delta_time_std_s=0.):
    """Causal 2-point speed with explicitly conditional uncertainty.

    Cov(delta position)=P_before+P_after-C12-C12.T if cross-frame block
    supplied. Without C12, the independence approximation is labelled and
    MUST NOT be presented as a verified 95% coverage bound.
    """
    if (not np.isfinite([max_gap_s, max_speed_kmh, max_speed_sigma_kmh,
                         delta_time_std_s]).all()
            or min(max_gap_s, max_speed_kmh, max_speed_sigma_kmh) <= 0
            or delta_time_std_s < 0):
        raise ValueError('INVALID_SPEED_LIMITS')
    for state in (before, after):
        if any(k not in state for k in ('timestamp_s', 'x_m', 'y_m', 'covariance_m2')):
            raise ValueError('INCOMPLETE_SPEED_SAMPLE')
    t0, t1 = float(before['timestamp_s']), float(after['timestamp_s'])
    xy0 = np.array([before['x_m'], before['y_m']], dtype=float)
    xy1 = np.array([after['x_m'], after['y_m']], dtype=float)
    if not np.isfinite([t0, t1, *xy0, *xy1]).all() or t1 <= t0:
        raise ValueError('NONMONOTONIC_OR_NONFINITE_TIME_POSITION')
    dt = t1-t0
    base = {'start_s': t0, 'end_s': t1, 'dt_s': dt,
            'speed_mps': None, 'speed_kmh': None, 'velocity_mps': None,
            'speed_sigma_kmh': None, 'speed_interval95_kmh': None,
            'speed_valid': False, 'systematic_errors_included': False,
            'cross_frame_correlation_accounted': cross_covariance_m2 is not None}
    if dt > max_gap_s:
        return {**base, 'status': 'GAP', 'reason': 'TIME_GAP_EXCEEDS_LIMIT'}
    p0 = _safe_cov(before['covariance_m2'], 'INVALID_PREVIOUS_COVARIANCE')
    p1 = _safe_cov(after['covariance_m2'], 'INVALID_CURRENT_COVARIANCE')
    p_delta = p0+p1
    if cross_covariance_m2 is not None:
        C = np.asarray(cross_covariance_m2, float)
        if C.shape != (2, 2) or not np.isfinite(C).all():
            raise ValueError('INVALID_CROSS_FRAME_COVARIANCE')
        p_delta -= C + C.T
    p_delta = (p_delta+p_delta.T)/2
    if np.linalg.eigvalsh(p_delta).min() < -1e-8:
        raise ValueError('INVALID_DIFFERENCE_COVARIANCE')
    delta = xy1-xy0
    distance = float(np.linalg.norm(delta))
    # Norm is nondifferentiable at zero; no bogus tiny sigma reported.
    if distance < 1e-10:
        return {**base, 'status': 'UNCERTAIN_PROJECTION',
                'reason': 'ZERO_DISPLACEMENT_DIRECTION_UNDEFINED'}
    velocity = delta / dt
    v_mps = distance/dt
    grad = (delta/distance)/dt
    variance = max(0., float(grad @ p_delta @ grad))
    # Include timestamp error in propagation of 1 / dt (first order).
    variance += (v_mps / dt * delta_time_std_s)**2
    sigma_kmh = 3.6 * math.sqrt(variance)
    speed_kmh = 3.6 * v_mps
    if not np.isfinite([sigma_kmh, speed_kmh]).all():
        raise ValueError('NONFINITE_SPEED_UNCERTAINTY')
    if speed_kmh > max_speed_kmh:
        return {**base, 'status': 'SPEED_OUTLIER', 'reason': 'EXCEEDS_SPEED_LIMIT'}
    if sigma_kmh > max_speed_sigma_kmh:
        return {**base, 'status': 'UNCERTAIN_PROJECTION',
                'reason': 'SPEED_SIGMA_EXCEEDS_LIMIT', 'speed_sigma_kmh': sigma_kmh}
    return {**base, 'status': 'VALID_CONDITIONAL', 'speed_valid': True,
            'speed_mps': float(v_mps), 'speed_kmh': float(speed_kmh),
            'velocity_mps': velocity.tolist(),
            'heading_deg': float(math.degrees(math.atan2(delta[1], delta[0]))),
            'speed_sigma_kmh': sigma_kmh,
            'speed_interval95_kmh': [max(0., speed_kmh-1.95996398454*sigma_kmh),
                                    speed_kmh+1.95996398454*sigma_kmh],
            'conditional_assumptions': 'time and pixel noise first-order Gaussian; cross-frame independence assumed' if cross_covariance_m2 is None else 'supplied cross-time covariance; pixel Gaussian delta method',
            'not_a_field_accuracy_certification': True}


def project_cross_frame_covariance_to_bev(u0, v0, u1, v1, cross_cov_uv, H):
    """Propagate Cov(pixel_previous, pixel_current) with J0 * C01 * J1.T.

    C01 is a rectangular 2x2 cross-covariance, not required to be symmetric
    or positive semidefinite by itself. The *joint* covariance should be PSD.
    """
    c=np.asarray(cross_cov_uv,float)
    if c.shape!=(2,2) or not np.isfinite(c).all():
        raise ValueError('INVALID_PIXEL_CROSS_COVARIANCE')
    h=np.asarray(H,float)
    # Each Jacobian comes from the same existing validated projection kernel.
    J0=np.asarray(project_pixel_covariance_to_bev(u0,v0,np.zeros((2,2)),h,
             max_major_axis_m=1e12)['jacobian_m_per_px'])
    J1=np.asarray(project_pixel_covariance_to_bev(u1,v1,np.zeros((2,2)),h,
             max_major_axis_m=1e12)['jacobian_m_per_px'])
    return (J0@c@J1.T).tolist()


def _convex_hull(points):
    """Deterministic calibrated-domain hull, not a rectangular extrapolation."""
    pts=sorted({(float(x),float(y)) for x,y in points})
    def cross(o,a,b):return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    def half(seq):
        result=[]
        for p in seq:
            while len(result)>=2 and cross(result[-2],result[-1],p)<=0:result.pop()
            result.append(p)
        return result
    hull=half(pts)[:-1]+half(reversed(pts))[:-1]
    if len(hull)<3:raise ValueError('DEGENERATE_CALIBRATION_DOMAIN')
    return tuple(hull)


def _inside_convex_polygon(x,y,hull):
    for a,b in zip(hull,(*hull[1:],hull[0])):
        cross=(b[0]-a[0])*(y-a[1])-(b[1]-a[1])*(x-a[0])
        if cross < -1e-8:return False
    return True


@dataclass(frozen=True)
class CalibrationEvidence:
    camera_id: str
    lens_calibration_id: str
    measurement_provenance: str
    numerical_quality_gate: dict
    pixel_roi: tuple[tuple[float, float], tuple[float, float]]
    calibration_hull_uv_px: tuple[tuple[float, float], ...]


class BEVMotionModule:
    """Track-specific, sequential pixel->metre+conditional speed state machine.

    Built from independently validated undistorted calibration. Source input
    contains explicit ground contact assertion; bbox bottoms are NOT assumed.
    """
    def __init__(self, H, evidence: CalibrationEvidence, *, max_major_axis_m=3.,
                 max_gap_s=.5, max_speed_kmh=240., max_speed_sigma_kmh=35.,
                 min_horizon_relative=1e-9, max_tracks=256):
        h = np.asarray(H, float)
        if (h.shape != (3, 3) or not np.isfinite(h).all() or
                np.linalg.matrix_rank(h) < 3 or not evidence.numerical_quality_gate.get('passed')):
            raise ValueError('UNVERIFIED_OR_INVALID_HOMOGRAPHY')
        if any(not np.isfinite(v) or v <= 0 for v in (max_major_axis_m,max_gap_s,max_speed_kmh,max_speed_sigma_kmh,min_horizon_relative)):
            raise ValueError('INVALID_BEVMOTION_LIMIT')
        if not isinstance(max_tracks, int) or not 1 <= max_tracks <= 10000:
            raise ValueError('INVALID_TRACK_LIMIT')
        self.H = h.copy()
        self.evidence = evidence
        self.max_major_axis_m = float(max_major_axis_m)
        self.max_gap_s = float(max_gap_s)
        self.max_speed_kmh = float(max_speed_kmh)
        self.max_speed_sigma_kmh = float(max_speed_sigma_kmh)
        self.min_horizon_relative = float(min_horizon_relative)
        self.max_tracks = max_tracks
        (x0,y0),(x1,y1)=evidence.pixel_roi
        ref=np.array([(x0+x1)*.5,(y0+y1)*.5,1.])
        ref_w=float(self.H[2] @ ref)
        if abs(ref_w) < min_horizon_relative*np.linalg.norm(self.H[2]):
            raise ValueError('UNSTABLE_HORIZON_IN_CALIBRATED_ROI')
        self._ref_sign = math.copysign(1.,ref_w)
        self._history: dict[str, dict] = {}

    @classmethod
    def from_correspondences(cls, *, fit_pixel_px, fit_world_m,
                             holdout_pixel_px, holdout_world_m,
                             camera_id, lens_calibration_id, measurement_provenance,
                             pixel_convention='undistorted', ransac=False,
                             ransac_threshold_px=2.0, ransac_trials=500,
                             seed=42, pixel_roi=None, **kwargs):
        if pixel_convention != 'undistorted':
            raise ValueError('UNDISTORTED_PIXELS_REQUIRED')
        for name,s in [('camera_id',camera_id),('lens_calibration_id',lens_calibration_id),('measurement_provenance',measurement_provenance)]:
            if not isinstance(s,str) or not s.strip():
                raise ValueError('MISSING_'+name.upper())
        a=np.asarray(fit_pixel_px,float);b=np.asarray(fit_world_m,float)
        c=np.asarray(holdout_pixel_px,float);d=np.asarray(holdout_world_m,float)
        if (a.ndim!=2 or a.shape[1:]!=(2,) or b.shape!=a.shape
                or c.ndim!=2 or c.shape[1:]!=(2,) or d.shape!=c.shape
                or not 4<=len(a)<=500 or not 4<=len(c)<=500
                or not all(np.isfinite(x).all() for x in (a,b,c,d))):
            raise ValueError('INVALID_CALIBRATION_CARDINALITY_OR_COORDINATES')
        if ransac:
            fit=fit_ransac(a,b,threshold_px=ransac_threshold_px,seed=seed,trials=ransac_trials)
            H=fit.H;inlier_count=fit.inlier_count;fit_rmse=fit.fit_rmse_px
        else:
            H=fit_dlt(a,b);inlier_count=len(a)
            fit_rmse=holdout_report(H,a,b)['reprojection_rmse_px']
        held=holdout_report(H,c,d)
        gate=assess_calibration({'fit_count':len(a),'inlier_count':inlier_count,
                                 'fit_rmse_px':fit_rmse,'holdout':held})
        if not gate['passed']:
            raise ValueError('CALIBRATION_QUALITY_BLOCKED:'+','.join(gate['reasons']))
        if pixel_roi is None:
            allpts=np.concatenate([a,c],axis=0)
            lo=allpts.min(axis=0);hi=allpts.max(axis=0)
        else:
            roi=np.asarray(pixel_roi,float)
            if roi.shape!=(2,2): raise ValueError('INVALID_PIXEL_ROI')
            lo,hi=roi
        if not np.isfinite([*lo,*hi]).all() or np.any(hi<=lo):
            raise ValueError('INVALID_PIXEL_ROI')
        if np.any(a<lo-1e-6) or np.any(a>hi+1e-6) or np.any(c<lo-1e-6) or np.any(c>hi+1e-6):
            raise ValueError('CALIBRATION_POINTS_OUTSIDE_ROI')
        evidence=CalibrationEvidence(camera_id,lens_calibration_id,measurement_provenance,gate,
                                     (tuple(map(float,lo)),tuple(map(float,hi))),
                                     _convex_hull(np.concatenate([a,c],axis=0)))
        return cls(H,evidence,**kwargs)

    def reset_track(self,track_id):
        self._history.pop(str(track_id),None)

    def process_sample(self, sample: dict[str,Any]) -> dict[str,Any]:
        if not isinstance(sample,dict):raise ValueError('SAMPLE_NOT_OBJECT')
        required={'track_id','timestamp_s','u','v','cov_uv','contact_is_ground','segment','camera_id','lens_calibration_id'}
        allowed=required|{'occluded','identity_discontinuity','cross_covariance_m2','delta_time_std_s'}
        if required-set(sample) or set(sample)-allowed:
            raise ValueError('INVALID_SAMPLE_CONTRACT')
        tid=sample['track_id']
        if not isinstance(tid,(str,int)) or isinstance(tid,bool) or not 1<=len(str(tid))<=128:
            raise ValueError('INVALID_TRACK_ID')
        tid=str(tid)
        segment=sample['segment']
        if not isinstance(segment,(str,int)) or isinstance(segment,bool) or not 1<=len(str(segment))<=128:
            raise ValueError('INVALID_SEGMENT')
        segment=str(segment)
        t=float(sample['timestamp_s']);u=float(sample['u']);v=float(sample['v'])
        if not np.isfinite([t,u,v]).all():raise ValueError('NONFINITE_TIME_OR_PIXEL')
        _safe_cov(sample['cov_uv'],'INVALID_PIXEL_COVARIANCE')
        if sample['camera_id'] != self.evidence.camera_id or sample['lens_calibration_id'] != self.evidence.lens_calibration_id:
            self.reset_track(tid)
            return {'track_id':tid,'timestamp_s':t,'status':'UNCERTAIN_PROJECTION','reason':'CAMERA_LENS_MISMATCH','speed_valid':False}
        old=self._history.get(tid)
        if old is not None and t <= old['timestamp_s']:
            raise ValueError('NONMONOTONIC_TRACK_TIMESTAMP')
        if sample.get('occluded',False) or sample.get('identity_discontinuity',False) or sample['contact_is_ground'] is not True:
            self.reset_track(tid)
            reason=('OCCLUSION' if sample.get('occluded',False) else
                    'IDENTITY_DISCONTINUITY' if sample.get('identity_discontinuity',False) else 'UNCERTAIN_GROUND_CONTACT')
            return {'track_id':tid,'timestamp_s':t,'status':'UNCERTAIN_PROJECTION','reason':reason,'speed_valid':False}
        (x0,y0),(x1,y1)=self.evidence.pixel_roi
        if not (x0<=u<=x1 and y0<=v<=y1 and
                _inside_convex_polygon(u,v,self.evidence.calibration_hull_uv_px)):
            self.reset_track(tid)
            return {'track_id':tid,'timestamp_s':t,'status':'UNCERTAIN_PROJECTION','reason':'OUTSIDE_CALIBRATED_ROI','speed_valid':False}
        w=float(self.H[2] @ np.array([u,v,1.]))
        if (w*self._ref_sign <= 0 or
                abs(w)<self.min_horizon_relative*np.linalg.norm(self.H[2])):
            self.reset_track(tid)
            return {'track_id':tid,'timestamp_s':t,'status':'UNCERTAIN_PROJECTION','reason':'NEAR_OR_BEHIND_HORIZON','speed_valid':False}
        projection=project_pixel_covariance_to_bev(u,v,sample['cov_uv'],self.H,
                                                    max_major_axis_m=self.max_major_axis_m,
                                                    horizon_epsilon=self.min_horizon_relative)
        if projection['status']!='VALID':
            self.reset_track(tid)
            return {'track_id':tid,'timestamp_s':t,'status':'UNCERTAIN_PROJECTION',
                    'reason':projection.get('reason','ELLIPSE_EXCEEDS_LIMIT'),
                    'projection':projection,'speed_valid':False}
        position={'timestamp_s':t,'x_m':projection['x_m'],'y_m':projection['y_m'],
                  'covariance_m2':projection['covariance_m2'],'segment':segment}
        result={'track_id':tid,'timestamp_s':t,'segment':segment,
                'position':projection,'velocity_mps':None,'acceleration_mps2':None, 'acceleration_confidence':'NOT_ESTABLISHED',
                'speed':None,'speed_valid':False,
                'calibration_status':'INDEPENDENT_NUMERICAL_PASS_NOT_FIELD_VALIDATED'}
        if old is None or old['segment']!=segment:
            result.update({'status':'INITIALIZED','reason':'FIRST_VALID_CONTACT_OR_NEW_SEGMENT'})
        else:
            interval=calculate_physical_speed(old,position,
                     cross_covariance_m2=sample.get('cross_covariance_m2'),
                     delta_time_std_s=float(sample.get('delta_time_std_s',0.)),
                     max_gap_s=self.max_gap_s,max_speed_kmh=self.max_speed_kmh,
                     max_speed_sigma_kmh=self.max_speed_sigma_kmh)
            result['speed']=interval
            result['status']=interval['status']
            result['speed_valid']=interval['speed_valid']
            if interval['speed_valid']:
                result['velocity_mps']=interval['velocity_mps']
                prev_v=old.get('velocity_mps')
                if prev_v is not None:
                    # Difference of interval velocity estimates; correlated
                    # acceleration confidence not estimated from two intervals.
                    result['acceleration_mps2']=((np.asarray(interval['velocity_mps'])-
                                                   np.asarray(prev_v))/(t-old['timestamp_s'])).tolist()
        if tid not in self._history and len(self._history)>=self.max_tracks:
            raise ValueError('TRACK_CAPACITY_EXCEEDED')
        position['velocity_mps']=result['velocity_mps']
        self._history[tid]=position
        return result
