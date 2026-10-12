import csv
import json
import tempfile
import unittest
from pathlib import Path
from bevmotionlab.interval_validity import audit_speed_intervals
from bevmotionlab.cli import analyze
from bevmotionlab.camera3d import intersect_ground
from test_camera3d import model

class IntervalValidityTests(unittest.TestCase):
    def setup_inputs(self, directory):
        camera = model()
        fits=[[176,136],[224,136],[176,184],[224,184]]
        holds=[[188,148],[212,148],[188,172],[212,172]]
        cal={'schema_version':'1.0','camera_id':'synthetic_cam',
             'resolution_px':[800,600], 'coordinate_frame':{'units':'m'},
             'pixel_convention':'undistorted','lens_calibration_id':'synthetic-identity',
             'measurement_provenance':'synthetic-only','calibrated_at_utc':'2026-10-11T12:00:00Z',
             'correspondences':[{'id':str(i),'image_uv_px':xy,
                                 'world_xy_m':intersect_ground(camera,[xy])[0].tolist(),
                                 'split':'fit' if i<4 else 'holdout'}
                                for i,xy in enumerate(fits+holds)]}
        a=directory/'cal.json'; b=directory/'cam.json'; a.write_text(json.dumps(cal));b.write_text(json.dumps(camera))
        detections=Path(__file__).resolve().parents[1]/'examples/synthetic_tracks.csv'
        return a,b,detections

    def test_full_cli_emits_one_audit_per_interval_and_manifest(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);cal,cam,det=self.setup_inputs(root)
            destination=root/'ok'
            report=analyze(cal,det,destination,camera_id='synthetic_cam',camera3d_model_json_path=cam,
                           contact_max_height_m=.5,max_height_speed_bias_kmh=100)
            with (destination/'speed_timeseries.csv').open() as f:
                rows=list(csv.DictReader(f))
            with (destination/'speed_height_risk.csv').open() as f:
                risks=list(csv.DictReader(f))
            self.assertEqual(len(rows),len(risks))
            self.assertGreater(len(risks),0)
            self.assertTrue(all(r['status']=='BOUND_ACCEPTED' for r in risks if r['speed_bias_upper_bound_kmh']))
            self.assertEqual(json.loads((destination/'height_bias_audit.json').read_text())['status'],'CONDITIONAL_PASS')
            self.assertIn('speed_height_risk.csv',json.loads((destination/'manifest.json').read_text())['files'])
    def test_exceeding_budget_cannot_write_partial_output(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);cal,cam,det=self.setup_inputs(root)
            dst=root/'blocked'
            with self.assertRaisesRegex(ValueError,'HEIGHT_BIAS_BLOCKED'):
                analyze(cal,det,dst,camera_id='synthetic_cam',camera3d_model_json_path=cam,
                        contact_max_height_m=1.,max_height_speed_bias_kmh=0)
            self.assertFalse(dst.exists())
    def test_all_intervals_must_match_exact_history(self):
        history={('t',0):[(0.,[0.,0.],[200.,160.]),(1.,[1.,0.],[320.,160.])]}
        rows=[{'track_id':'t','segment':0,'start_s':0.,'end_s':1.,'dt_s':1.,'speed_kmh':3.6,'status':'VALID'}]
        self.assertTrue(audit_speed_intervals(model(),history,rows,maximum_height_m=1.,max_bias_kmh=4)['accepted'])
        rows[0]['end_s']=2.
        with self.assertRaisesRegex(ValueError,'UNMATCHED_INTERVALS'):
            audit_speed_intervals(model(),history,rows,maximum_height_m=1.,max_bias_kmh=4)
    def test_reject_unsupported_partial_inputs_and_negative_budget(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);cal,cam,det=self.setup_inputs(root)
            with self.assertRaisesRegex(ValueError,'BOTH_PARAMETERS_REQUIRED'):
                analyze(cal,det,root/'bad',camera_id='synthetic_cam',contact_max_height_m=.1)
            with self.assertRaisesRegex(ValueError,'REQUIRES_AUDITED_3D_CAMERA'):
                analyze(cal,det,root/'bad',camera_id='synthetic_cam',contact_max_height_m=.1,max_height_speed_bias_kmh=10)
            with self.assertRaisesRegex(ValueError,'INVALID_BIAS_THRESHOLD'):
                analyze(cal,det,root/'bad',camera_id='synthetic_cam',camera3d_model_json_path=cam,
                        contact_max_height_m=.1,max_height_speed_bias_kmh=-1)
if __name__=='__main__': unittest.main()