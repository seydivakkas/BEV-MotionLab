"""Synthetic camera rays, physical geometry, distortion and fail-closed checks."""
import unittest
import numpy as np
from bevmotionlab.camera3d import intersect_ground, audit_ground_model


def model():
    return {'schema_version':'1.0','world_units':'m','plane_z_m':0,
            'lens_calibration_id':'synthetic-identity', 'pixel_convention':'undistorted',
            'K':[[120,0,200],[0,120,160],[0,0,1]],
            'R_cw':[[1,0,0],[0,-1,0],[0,0,-1]],
            'camera_center_world_m':[0,0,10],
            'distortion':{'k1':0.,'k2':0.,'p1':0.,'p2':0.,'k3':0.}}

class Camera3DTests(unittest.TestCase):
    def test_ground_intersection_and_report(self):
        m=model(); px=[[188,172],[212,172],[188,148],[212,148]]
        expected=np.array([[-1,-1],[1,-1],[-1,1],[1,1]])
        np.testing.assert_allclose(intersect_ground(m,px),expected,atol=1e-12)
        self.assertTrue(audit_ground_model(m,pixel_points=px,planar_world_points=expected)['passed'])
        self.assertFalse(audit_ground_model(m,pixel_points=px,planar_world_points=expected+np.array([.35,0]))['passed'])
    def test_lens_distortion_is_applied_only_to_raw_pixels(self):
        m=model();m['pixel_convention']='raw_distorted';m['distortion']['k1']=.12
        ideal=np.array([.25,-.15]);r2=float(ideal@ideal)
        distorted=ideal*(1+.12*r2)
        uv=[[200+120*distorted[0],160+120*distorted[1]]]
        np.testing.assert_allclose(intersect_ground(m,uv)[0],[2.5,1.5],rtol=1e-9,atol=1e-9)
    def test_camera_provenance_and_invalid_geometry(self):
        m=model();m.pop('lens_calibration_id')
        with self.assertRaisesRegex(ValueError,'PROVENANCE'):intersect_ground(m,[[200,160]])
        m=model();m['R_cw'][2]=[0,0,1]
        with self.assertRaisesRegex(ValueError,'ROTATION'):intersect_ground(m,[[200,160]])
        m=model();m['R_cw']=[[1,0,0],[0,1,0],[0,0,1]]
        with self.assertRaisesRegex(ValueError,'RAY_DOES_NOT_HIT_GROUND'):intersect_ground(m,[[200,160]])
    def test_degenerate_markers_rejected(self):
        with self.assertRaisesRegex(ValueError,'DEGENERATE_WORLD'):
            audit_ground_model(model(),pixel_points=[[180,140],[190,150],[200,160],[210,170]],
                               planar_world_points=[[0,0],[1,1],[2,2],[3,3]])

if __name__=='__main__':unittest.main()

class Camera3DCliIntegrationTests(unittest.TestCase):
    def test_strict_independent_holdout_writes_audit_and_detects_mismatch(self):
        import json, tempfile
        from pathlib import Path
        from bevmotionlab.cli import analyze
        m=model()
        # Synthetic intrinsics and extrinsics are consistent with the generated
        # world landmarks; this does NOT calibrate a real lens or camera.
        fit_uv=[[176,136],[224,136],[176,184],[224,184]]
        hold_uv=[[188,148],[212,148],[188,172],[212,172]]
        def world(p):return intersect_ground(m,[p])[0].tolist()
        cal={'schema_version':'1.0','camera_id':'synthetic_cam','resolution_px':[800,600],
             'coordinate_frame':{'units':'m'},'pixel_convention':'undistorted',
             'lens_calibration_id':'synthetic-identity',
             'measurement_provenance':'synthetic-only',
             'calibrated_at_utc':'2026-10-11T12:00:00Z',
             'correspondences':[{'id':str(i),'image_uv_px':p,'world_xy_m':world(p),
                                 'split':'fit' if i<4 else 'holdout'}
                                for i,p in enumerate(fit_uv+hold_uv)]}
        with tempfile.TemporaryDirectory() as td:
            directory=Path(td); calibration=directory/'cal.json';camera=directory/'model.json'
            calibration.write_text(json.dumps(cal),encoding='utf8');camera.write_text(json.dumps(m),encoding='utf8')
            detections=Path(__file__).resolve().parents[1]/'examples/synthetic_tracks.csv'
            kwargs=dict(camera_id='synthetic_cam',camera3d_model_json_path=camera)
            analyze(calibration,detections,directory/'ok',**kwargs)
            audit=json.loads((directory/'ok/camera3d_audit.json').read_text())
            self.assertTrue(audit['passed'])
            self.assertIn('camera3d_audit.json',json.loads((directory/'ok/manifest.json').read_text())['files'])
            cal['correspondences'][-1]['world_xy_m'][0]+=2
            calibration.write_text(json.dumps(cal),encoding='utf8')
            with self.assertRaisesRegex(ValueError,'CAMERA3D_GEOMETRY_BLOCKED'):
                analyze(calibration,detections,directory/'fail',**kwargs)
            self.assertFalse((directory/'fail').exists())