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
from .quality import assess_calibration
from .pose_audit import audit_pose_file
from .camera3d import audit_ground_model
from .video import validate_pts, validate_detection_timestamps
from .schema import Detection, load_calibration
from .speed import estimate_speed
from .uncertainty import interval_uncertainties
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
    report['quality_gate'] = assess_calibration(report)
    return H, report, data


def analyze(calibration_path, detections_path, output, *, camera_id, robust=False, seed=42, max_gap_s=2.,
            require_quality_gate=False, ffprobe_json_path=None, pts_tolerance_s=.005, pts_offset_s=0.,
            uncertainty_trials=0, pixel_std_px=.5, landmark_std_px=.5, timestamp_std_s=0., pose_audit_json_path=None, pose_max_rmse_px=2., pose_max_individual_px=4., camera3d_model_json_path=None, camera3d_max_residual_m=.15):
    out=Path(output)
    if not camera_id: raise ValueError('camera_id required')
    if type(uncertainty_trials) is not int or uncertainty_trials < 0 or (uncertainty_trials and uncertainty_trials < 10) or uncertainty_trials > 5000:
        raise ValueError('UNCERTAINTY_TRIALS_INVALID')
    if robust and uncertainty_trials:
        raise ValueError('ROBUST_UNCERTAINTY_UNSUPPORTED: no conditional robust-refit model')
    if out.resolve() in (Path(calibration_path).resolve(),Path(detections_path).resolve()):
        raise ValueError('output path must not replace input')
    H,report,cal=calibrate(calibration_path,robust=robust,seed=seed)
    if camera_id!=cal['camera_id']: raise ValueError('CAMERA_MISMATCH: calibration does not match observations')
    if require_quality_gate and not report['quality_gate']['passed']:
        raise ValueError('CALIBRATION_QUALITY_BLOCKED: '+','.join(report['quality_gate']['reasons']))
    if cal.get('pixel_convention')!='undistorted': raise ValueError('lens model unsupported')
    pose_audit = None
    if pose_audit_json_path is not None:
        pose_audit = audit_pose_file(cal, pose_audit_json_path,
                                    max_rmse_px=pose_max_rmse_px,
                                    max_individual_px=pose_max_individual_px)
        if not pose_audit['passed']:
            raise ValueError('POSE_AUDIT_BLOCKED: '+','.join(pose_audit['reasons']))
    camera3d_audit = None
    if camera3d_model_json_path is not None:
        independent = [p for p in cal['correspondences'] if p['split']=='holdout']
        if len(independent) < 4:
            raise ValueError('CAMERA3D_INSUFFICIENT_INDEPENDENT_HOLDOUT')
        model = json.loads(Path(camera3d_model_json_path).read_text(encoding='utf-8'))
        if model['lens_calibration_id'] != cal['lens_calibration_id']:
            raise ValueError('CAMERA3D_LENS_PROVENANCE_MISMATCH')
        # Pre-undistorted calibration coordinates are never re-undistorted.
        if model.get('pixel_convention') != 'undistorted':
            raise ValueError('CAMERA3D_MODEL_MUST_ACCEPT_CALIBRATION_PIXEL_CONVENTION')
        camera3d_audit = audit_ground_model(
            model, pixel_points=[p['image_uv_px'] for p in independent],
            planar_world_points=[p['world_xy_m'] for p in independent],
            max_residual_m=camera3d_max_residual_m)
        if not camera3d_audit['passed']:
            raise ValueError('CAMERA3D_GEOMETRY_BLOCKED')
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
                history.setdefault(key,[]).append((d.timestamp_s,xy,uv))
            else:
                segment_for_track[d.track_id]=segment_for_track.get(d.track_id,0)+1
            if status!='VALID': issues.append({'track_id':d.track_id,'time_s':d.timestamp_s,'reason':status})
    pts_audit = None
    if ffprobe_json_path is not None:
        pts_data = validate_pts(json.loads(Path(ffprobe_json_path).read_text(encoding='utf-8')))
        pts_audit = validate_detection_timestamps(projected, pts_data, tolerance_s=pts_tolerance_s, offset_s=pts_offset_s)
    rows=[]
    for (tid,segment), seq in sorted(history.items()):
        results=estimate_speed([v[0] for v in seq],[v[1] for v in seq],max_gap_s=max_gap_s)
        rows.extend({'track_id':tid,'segment':segment,**r} for r in results)
    uncertainty_rows=[]
    if uncertainty_trials:
        fit=[p for p in cal['correspondences'] if p['split']=='fit']
        uncertainty_rows=interval_uncertainties(history,
            [p['image_uv_px'] for p in fit], [p['world_xy_m'] for p in fit],
            trials=uncertainty_trials,seed=seed,pixel_std_px=pixel_std_px,
            landmark_std_px=landmark_std_px,timestamp_std_s=timestamp_std_s,
            max_gap_s=max_gap_s)
        if len(uncertainty_rows)!=len(rows) or any(
            item['status']=='UNCERTAINTY_NOT_IDENTIFIABLE' for item in uncertainty_rows):
            raise ValueError('UNCERTAINTY_GATE_BLOCKED: invalid/missing interval uncertainty')
    out.mkdir(parents=True,exist_ok=True)
    files=[]
    for name, data, keys in [('projected_tracks.csv',projected,['frame','timestamp_s','track_id','x_m','y_m','confidence','status']),
                             ('speed_timeseries.csv',rows,['track_id','segment','start_s','end_s','dt_s','speed_kmh','status'])]:
        file=out/name
        with file.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data)
        files.append(file)
    if uncertainty_trials:
        filename=out/'speed_uncertainty.csv'
        with filename.open('w',newline='',encoding='utf-8') as f:
            fields=['track_id','segment','start_s','end_s','status','median_kmh','p025_kmh','p975_kmh','valid_trials','failed_trials']
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(uncertainty_rows)
        files.append(filename)
    for name,obj in [('calibration_report.json',report),('pose_audit.json',pose_audit or {'status':'NOT_CHECKED','reason':'independent fixed-landmark evidence not supplied'}),
                     ('camera3d_audit.json',camera3d_audit or {'status':'NOT_CHECKED','reason':'camera 3D extrinsics and independent ground holdouts not supplied'}),('flags.json',issues),
                     ('timestamp_audit.json',pts_audit or {'status':'NOT_CHECKED','reason':'ffprobe PTS not supplied'}),
                     ('run_config.json',{'schema_version':'1.0','camera_id':camera_id,'seed':seed,'robust':robust,'max_gap_s':max_gap_s,'calibration_sha256':hashlib.sha256(Path(calibration_path).read_bytes()).hexdigest(),'detections_sha256':hashlib.sha256(Path(detections_path).read_bytes()).hexdigest(),
                       'ffprobe_sha256':hashlib.sha256(Path(ffprobe_json_path).read_bytes()).hexdigest() if ffprobe_json_path else None,
                       'pose_audit_sha256':hashlib.sha256(Path(pose_audit_json_path).read_bytes()).hexdigest() if pose_audit_json_path else None,
                       'camera3d_model_sha256':hashlib.sha256(Path(camera3d_model_json_path).read_bytes()).hexdigest() if camera3d_model_json_path else None,
                        'require_quality_gate':require_quality_gate,
                       'uncertainty_model':{'trials':uncertainty_trials,'pixel_std_px':pixel_std_px,
                                            'landmark_std_px':landmark_std_px,'timestamp_std_s':timestamp_std_s,
                                            'systematic_bias_included':False} if uncertainty_trials else None})]:
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
    a=sub.add_parser('analyze');a.add_argument('--calibration',required=True);a.add_argument('--detections',required=True);a.add_argument('--camera-id',required=True);a.add_argument('--out',required=True);a.add_argument('--robust',action='store_true');a.add_argument('--seed',type=int,default=42);a.add_argument('--max-gap-s',type=float,default=2.);a.add_argument('--require-quality-gate',action='store_true');a.add_argument('--ffprobe-json');a.add_argument('--pts-tolerance-s',type=float,default=.005);a.add_argument('--pts-offset-s',type=float,default=0.);a.add_argument('--uncertainty-trials',type=int,default=0);a.add_argument('--pixel-std-px',type=float,default=.5);a.add_argument('--landmark-std-px',type=float,default=.5);a.add_argument('--timestamp-std-s',type=float,default=0.);a.add_argument('--pose-audit-json');a.add_argument('--pose-max-rmse-px',type=float,default=2.);a.add_argument('--pose-max-individual-px',type=float,default=4.);a.add_argument('--camera3d-model-json');a.add_argument('--camera3d-max-residual-m',type=float,default=.15)
    args=parser.parse_args(argv)
    if args.command=='calibrate':
        _, report, _=calibrate(args.calibration,robust=args.robust,seed=args.seed)
        path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True);_write_json(path,report)
        print(json.dumps({'calibration_report':str(path),'fit_rmse_px':report['fit_rmse_px']}))
    else:
        print(json.dumps(analyze(args.calibration,args.detections,args.out,camera_id=args.camera_id,robust=args.robust,seed=args.seed,max_gap_s=args.max_gap_s,require_quality_gate=args.require_quality_gate,ffprobe_json_path=args.ffprobe_json,pts_tolerance_s=args.pts_tolerance_s,pts_offset_s=args.pts_offset_s,uncertainty_trials=args.uncertainty_trials,pixel_std_px=args.pixel_std_px,landmark_std_px=args.landmark_std_px,timestamp_std_s=args.timestamp_std_s, pose_audit_json_path=args.pose_audit_json,pose_max_rmse_px=args.pose_max_rmse_px,pose_max_individual_px=args.pose_max_individual_px,camera3d_model_json_path=args.camera3d_model_json,camera3d_max_residual_m=args.camera3d_max_residual_m)))

if __name__=='__main__':main()