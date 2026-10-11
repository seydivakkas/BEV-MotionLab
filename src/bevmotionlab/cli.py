"""Offline CLI for validated homography, tracking-foot projection and speed."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from . import __version__
from .geometry import fit_dlt, fit_ransac, holdout_report, project
from .schema import Detection, load_calibration
from .speed import estimate_speed
from .visualize import render_bev_svg


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')


def _manifest(output, paths):
    data={'schema_version':'1.0','software_version':__version__,'files':{}}
    for p in paths:
        p=Path(p); data['files'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    _write_json(Path(output)/'manifest.json',data)


def calibrate(calibration_path, *, robust=False, seed=42):
    data, fit, hold = load_calibration(calibration_path)
    uv=np.array([v[0] for v in fit]); xy=np.array([v[1] for v in fit])
    fitted=fit_ransac(uv,xy,seed=seed) if robust else None
    H=fitted.H if fitted is not None else fit_dlt(uv,xy)
    report={'schema_version':'1.0','camera_id':data['camera_id'],
            'fit_count':len(fit),'inlier_count':fitted.inlier_count if fitted else len(fit),
            'fit_rmse_px':fitted.fit_rmse_px if fitted else holdout_report(H,uv,xy)['reprojection_rmse_px'],
            'holdout':holdout_report(H,[v[0] for v in hold],[v[1] for v in hold]) if hold else None,
            'holdout_status':'MEASURED' if hold else 'NOT_AVAILABLE',
            'H_pixel_to_world':H.tolist(), 'method':'ransac' if robust else 'dlt',
            'seed':seed, 'model':'planar-only', 'claim':'synthetic/engineering only; no field accuracy validation'}
    return H, report, data


def analyze(calibration_path, detections_path, output, *, camera_id, robust=False, seed=42, max_gap_s=2.):
    out=Path(output)
    if not camera_id: raise ValueError('camera_id required')
    if out.resolve() in (Path(calibration_path).resolve(),Path(detections_path).resolve()):
        raise ValueError('output path must not replace input')
    H,report,cal=calibrate(calibration_path,robust=robust,seed=seed)
    if camera_id!=cal['camera_id']: raise ValueError('CAMERA_MISMATCH: calibration does not match observations')
    if cal.get('pixel_convention')!='undistorted': raise ValueError('lens model unsupported')
    history={}; projected=[]; issues=[]; segment_for_track={}; last_seen={}
    with Path(detections_path).open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            d=Detection.from_csv(row,cal['resolution_px'])
            key=(d.track_id,segment_for_track.get(d.track_id,0))
            prior_time=last_seen.get(d.track_id)
            if prior_time is not None and d.timestamp_s<=prior_time:
                raise ValueError(f'INVALID_TIMESTAMP track_id={d.track_id}')
            last_seen[d.track_id]=d.timestamp_s
            uv=d.foot_px
            roi=cal.get('valid_roi_uv_px')
            status='VALID'
            if roi is not None and not (roi[0][0]<=uv[0]<=roi[1][0] and roi[0][1]<=uv[1]<=roi[1][1]):
                status='OUTSIDE_CALIBRATED_ROI'
            elif d.box[3]>=cal['resolution_px'][1]-1:
                status='CONTACT_UNCERTAIN'
            try:
                xy=project(H,[uv])[0] if status=='VALID' else None
            except ValueError:
                status='INVALID_HOMOGRAPHY';xy=None
            rec={'frame':d.frame,'timestamp_s':d.timestamp_s,'track_id':d.track_id,'x_m':None if xy is None else float(xy[0]),
                 'y_m':None if xy is None else float(xy[1]),'confidence':d.confidence,'status':status}
            projected.append(rec)
            if xy is not None:
                history.setdefault(key,[]).append((d.timestamp_s,xy))
            else:
                segment_for_track[d.track_id]=segment_for_track.get(d.track_id,0)+1
            if status!='VALID': issues.append({'track_id':d.track_id,'time_s':d.timestamp_s,'reason':status})
    rows=[]
    for (tid,segment), seq in sorted(history.items()):
        results=estimate_speed([v[0] for v in seq],[v[1] for v in seq],max_gap_s=max_gap_s)
        rows.extend({'track_id':tid,'segment':segment,**r} for r in results)
    out.mkdir(parents=True,exist_ok=True)
    files=[]
    for name, data, keys in [('projected_tracks.csv',projected,['frame','timestamp_s','track_id','x_m','y_m','confidence','status']),
                             ('speed_timeseries.csv',rows,['track_id','segment','start_s','end_s','dt_s','speed_kmh','status'])]:
        file=out/name
        with file.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data)
        files.append(file)
    for name,obj in [('calibration_report.json',report),('flags.json',issues),
                     ('run_config.json',{'schema_version':'1.0','camera_id':camera_id,'seed':seed,'robust':robust,'max_gap_s':max_gap_s,'calibration_sha256':hashlib.sha256(Path(calibration_path).read_bytes()).hexdigest(),'detections_sha256':hashlib.sha256(Path(detections_path).read_bytes()).hexdigest()})]:
        path=out/name;_write_json(path,obj);files.append(path)
    try:
        svg=render_bev_svg(out/'projected_tracks.csv',out/'bev_trajectory.svg')
        files.append(svg)
    except ValueError:
        issues.append({'reason':'NO_VALID_POINTS_FOR_BEV_VIEW'})
        _write_json(out/'flags.json',issues)
    _manifest(out,files)
    return {'calibration':report,'observations':len(projected),'speed_intervals':len(rows),'flags':len(issues),'output':str(out)}


def main(argv=None):
    parser=argparse.ArgumentParser(description='BEV-MotionLab offline geometry backend v0.1')
    sub=parser.add_subparsers(dest='command',required=True)
    c=sub.add_parser('calibrate');c.add_argument('--calibration',required=True);c.add_argument('--out',required=True);c.add_argument('--robust',action='store_true');c.add_argument('--seed',type=int,default=42)
    a=sub.add_parser('analyze');a.add_argument('--calibration',required=True);a.add_argument('--detections',required=True);a.add_argument('--camera-id',required=True);a.add_argument('--out',required=True);a.add_argument('--robust',action='store_true');a.add_argument('--seed',type=int,default=42);a.add_argument('--max-gap-s',type=float,default=2.)
    args=parser.parse_args(argv)
    if args.command=='calibrate':
        _, report, _=calibrate(args.calibration,robust=args.robust,seed=args.seed)
        path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True);_write_json(path,report)
        print(json.dumps({'calibration_report':str(path),'fit_rmse_px':report['fit_rmse_px']}))
    else:
        print(json.dumps(analyze(args.calibration,args.detections,args.out,camera_id=args.camera_id,robust=args.robust,seed=args.seed,max_gap_s=args.max_gap_s)))

if __name__=='__main__':main()