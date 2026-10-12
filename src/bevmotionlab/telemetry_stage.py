"""BEV JSONL stage: camera-calibrated pixel tracks -> metre-space trajectories.
Default example H is a SYNTHETIC scale mapping only, never a real calibration.
"""
from __future__ import annotations
import json,sys,argparse
import numpy as np
from .covariance_projection import project_pixel_covariance_to_bev


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--homography-json',help='Explicit measured pixel-to-metre H; otherwise DEMO synthetic scale')
    parser.add_argument('--max-ellipse-m',type=float,default=8.)
    a=parser.parse_args(argv)
    H=np.asarray(json.load(open(a.homography_json,encoding='utf-8'))['H_pixel_to_world'],float) if a.homography_json else np.diag([.06,.06,1.])
    positions={}
    for line in sys.stdin:
        if not line.strip():continue
        try:
            event=json.loads(line)
            if event.get('type')!='tracks':
                print(json.dumps(event),flush=True);continue
            now=float(event['timestamp_s']);results=[]
            for tr in event['tracks']:
                tid=tr['track_id']
                projected=project_pixel_covariance_to_bev(tr['u'],tr['v'],tr['pixel_covariance'],H,max_major_axis_m=a.max_ellipse_m)
                valid=projected['status']=='VALID' and tr['speed_valid']
                current=np.array([projected.get('x_m',float('nan')),projected.get('y_m',float('nan'))])
                velocity=None;V=None
                if valid and tid in positions:
                    old_t,old_xy,old_cov=positions[tid]
                    dt=now-old_t
                    if 0<dt<=.35:
                        velocity=((current-old_xy)/dt).tolist()
                        # Assumes independence; cross-frame filter covariance is unknown.
                        V=((np.asarray(projected['covariance_m2'])+old_cov)/dt**2).tolist()
                if valid:
                    positions[tid]=(now,current,np.asarray(projected['covariance_m2']))
                else:
                    positions.pop(tid,None)
                results.append({'track_id':tid,'projection':projected,'velocity_mps':velocity,
                                'velocity_covariance':V,
                                'speed_kmh':float(3.6*np.linalg.norm(velocity)) if velocity is not None else None,
                                'valid':bool(valid and velocity is not None),
                                'status':'VALID' if valid and velocity is not None else 'UNCERTAIN_PROJECTION_OR_SPEED'})
            print(json.dumps({'schema_version':'1.0','type':'bev','stage':'bev','timestamp_s':now,
                              'tracks':results,'source':event['source'],'calibration':'SYNTHETIC_SCALE' if not a.homography_json else 'USER_SUPPLIED_UNVERIFIED'},allow_nan=False),flush=True)
        except Exception as e:
            print(json.dumps({'schema_version':'1.0','type':'error','stage':'bev','error':type(e).__name__,'detail':str(e)}),flush=True)

if __name__=='__main__':main()
