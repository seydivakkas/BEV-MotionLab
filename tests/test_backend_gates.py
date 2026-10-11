import json
import tempfile
import unittest
from pathlib import Path
from bevmotionlab.quality import assess_calibration
from bevmotionlab.video import validate_detection_timestamps, validate_pts
from bevmotionlab.cli import analyze

ROOT = Path(__file__).resolve().parents[1]


class GateTests(unittest.TestCase):
    def test_calibration_gate_requires_holdout(self):
        report = dict(fit_count=5, inlier_count=5, fit_rmse_px=.1, holdout=None)
        a = assess_calibration(report)
        self.assertFalse(a['passed'])
        self.assertIn('NO_INDEPENDENT_HOLDOUT', a['reasons'])
        report['holdout'] = {'n':2, 'world_rmse_m':.25, 'reprojection_rmse_px':1.1}
        self.assertTrue(assess_calibration(report)['passed'])
        report['holdout']['world_rmse_m']=.9
        self.assertIn('HOLDOUT_WORLD_ERROR_TOO_HIGH', assess_calibration(report)['reasons'])

    def test_pts_frame_alignment_and_duplicates(self):
        pts = validate_pts({'frames':[{'best_effort_timestamp_time':x} for x in ('0','.1','.21')]})
        rows = [{'frame':1,'timestamp_s':.1}, {'frame':1,'timestamp_s':.1}, {'frame':2,'timestamp_s':.21}]
        self.assertEqual(validate_detection_timestamps(rows,pts)['distinct_frames'],2)
        with self.assertRaisesRegex(ValueError,'PTS_MISMATCH'):
            validate_detection_timestamps([{'frame':2,'timestamp_s':.2}],pts)
        with self.assertRaisesRegex(ValueError,'PTS_CONFLICTING'):
            validate_detection_timestamps([{'frame':1,'timestamp_s':.1},{'frame':1,'timestamp_s':.1000001}],pts)
        with self.assertRaisesRegex(ValueError,'PTS_FRAME'):
            validate_detection_timestamps([{'frame':8,'timestamp_s':.1}],pts)

    def test_cli_rejects_misaligned_pts_without_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);pts=tmp/'pts.json'
            pts.write_text(json.dumps({'frames':[{'best_effort_timestamp_time':str(t)} for t in (0,.5,1.,1.5,2.)]}))
            kw={'calibration_path':ROOT/'examples/synthetic_calibration.json',
                'detections_path':ROOT/'examples/synthetic_tracks.csv','camera_id':'synthetic_cam'}
            a=analyze(**kw,output=tmp/'good',ffprobe_json_path=pts,require_quality_gate=True)
            self.assertEqual(a['speed_intervals'],4)
            self.assertEqual(json.loads((tmp/'good'/'timestamp_audit.json').read_text())['status'],'MATCHED')
            pts.write_text(json.dumps({'frames':[{'best_effort_timestamp_time':str(t)} for t in (0,.5,1.1,1.5,2.)]}))
            with self.assertRaisesRegex(ValueError,'PTS_MISMATCH'):
                analyze(**kw,output=tmp/'bad',ffprobe_json_path=pts)
            self.assertFalse((tmp/'bad').exists())


if __name__ == '__main__':unittest.main()