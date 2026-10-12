"""Homography Jacobian uncertainty propagation for undistorted input pixels.
No systematic calibration/lens bias is included in first-order delta covariance.
"""
from __future__ import annotations
import math
import numpy as np

CHI2_2_95=5.991464547


def project_pixel_covariance_to_bev(u,v,cov_uv,H,*,max_major_axis_m=10.,horizon_epsilon=1e-9):
    h=np.asarray(H,dtype=float);p=np.asarray(cov_uv,float)
    if h.shape!=(3,3) or p.shape!=(2,2) or not np.isfinite(h).all() or not np.isfinite(p).all():raise ValueError('INVALID_H_OR_COVARIANCE')
    if not np.allclose(p,p.T,atol=1e-10) or np.linalg.eigvalsh(p).min()<-1e-10:raise ValueError('NOT_PSD')
    if not np.isfinite([u,v,max_major_axis_m]).all() or max_major_axis_m<=0:raise ValueError('INVALID_POINT_OR_LIMIT')
    w=float(h[2,0]*u+h[2,1]*v+h[2,2]);scale=float(np.linalg.norm(h[2,:]))
    if abs(w)<horizon_epsilon*max(scale,1e-12):
        return {'status':'UNCERTAIN_PROJECTION','reason':'NEAR_HORIZON','speed_valid':False}
    X=(h[0,0]*u+h[0,1]*v+h[0,2])/w
    Y=(h[1,0]*u+h[1,1]*v+h[1,2])/w
    J=np.array([[(h[0,0]-X*h[2,0])/w,(h[0,1]-X*h[2,1])/w],
                [(h[1,0]-Y*h[2,0])/w,(h[1,1]-Y*h[2,1])/w]])
    C=J@p@J.T;C=(C+C.T)/2
    eig,axes=np.linalg.eigh(C);eig=np.maximum(eig,0.0)
    major=float(math.sqrt(CHI2_2_95*eig[1]));minor=float(math.sqrt(CHI2_2_95*eig[0]))
    result={'x_m':float(X),'y_m':float(Y),'covariance_m2':C.tolist(),
            'jacobian_m_per_px':J.tolist(),'ellipse95':{'major_m':major,'minor_m':minor,
                    'angle_deg':float(np.rad2deg(math.atan2(axes[1,1],axes[0,1])))},
            'note':'Conditional first-order pixel noise only; not systematic camera/ground contact error.'}
    status='VALID' if major<=max_major_axis_m else 'UNCERTAIN_PROJECTION'
    return {'status':status,'speed_valid':status=='VALID',**result}
