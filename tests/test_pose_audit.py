"""Independent synthetic fiducial pose-audit contract tests."""
import json
import tempfile
import unittest
from pathlib import Path

from bevmotionlab.pose_audit import audit_pose
from bevmotionlab.cli import analyze

ROOT = Path(__file__).resolve().parents[1]


class PoseAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cal = json.loads((ROOT/'examples/synthetic_calibration.json').read_text())

    def observation(self, dx=0., dy=0.):
        c=self.cal
        return {'camera_id': c['camera_id'], 'resolution_px':c['resolution_px'],
                'lens_calibration_id':c['lens_calibration_id'],
                'pixel_convention':'undistorted',
                'points':[{'id':p['id'], 'image_uv_px':[p['image_uv_px'][0]+dx,p['image_uv_px'][1]+dy]}
                          for p in c['correspondences']]}

    def test_stable_and_shifted_camera_proxy(self):
        self.assertTrue(audit_pose(self.cal, self.observation())['passed'])
        shifted=audit_pose(self.cal,self.observation(dx=8.))
        self.assertEqual(shifted['status'],'BLOCKED')
        self.assertIn('POSE_RMSE_EXCEEDS_LIMIT',shifted['reasons'])

    def test_unknown_duplicate_and_missing_fiducials_fail_closed(self):
        obj=self.observation();obj['points'][0]['id']='unknown'
        with self.assertRaisesRegex(ValueError,'UNKNOWN_OR_DUPLICATE'):
            audit_pose(self.cal,obj)
        obj=self.observation();obj['points']=obj['points'][:3]
        with self.assertRaisesRegex(ValueError,'INSUFFICIENT'):
            audit_pose(self.cal,obj)
        obj=self.observation();obj['points'][1]['id']=obj['points'][0]['id']
        with self.assertRaisesRegex(ValueError,'UNKNOWN_OR_DUPLICATE'):
            audit_pose(self.cal,obj)

    def test_provenance_mismatch_blocks(self):
        obj=self.observation();obj['lens_calibration_id']='altered'
        with self.assertRaisesRegex(ValueError,'PROVENANCE'):
            audit_pose(self.cal,obj)

    def test_cli_no_artifacts_if_drift_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); inp=root/'pose.json';out=root/'blocked'
            inp.write_text(json.dumps(self.observation(dx=8.)))
            kwargs={'calibration_path': ROOT/'examples/synthetic_calibration.json',
                    'detections_path': ROOT/'examples/synthetic_tracks.csv',
                    'camera_id':'synthetic_cam','pose_audit_json_path':inp}
            with self.assertRaisesRegex(ValueError,'POSE_AUDIT_BLOCKED'):
                analyze(**kwargs,output=out)
            self.assertFalse(out.exists())
            inp.write_text(json.dumps(self.observation()))
            good=analyze(**kwargs,output=root/'ok')
            self.assertEqual(json.loads((root/'ok/pose_audit.json').read_text())['status'],'NUMERICALLY_STABLE')
            manifest=json.loads((root/'ok/manifest.json').read_text())
            self.assertIn('pose_audit.json',manifest['files'])


if __name__=='__main__':unittest.main()
