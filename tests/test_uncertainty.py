import csv
import json
import tempfile
import unittest
from pathlib import Path
from bevmotionlab.cli import analyze
from bevmotionlab.uncertainty import interval_uncertainties

ROOT=Path(__file__).resolve().parents[1]
KW=dict(calibration_path=ROOT/'examples/synthetic_calibration.json',
        detections_path=ROOT/'examples/synthetic_tracks.csv',camera_id='synthetic_cam')

class UncertaintyTests(unittest.TestCase):
    def test_stable_interval_uncertainty_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            analyze(**KW,output=p/'a',require_quality_gate=True,uncertainty_trials=60,seed=7)
            analyze(**KW,output=p/'b',require_quality_gate=True,uncertainty_trials=60,seed=7)
            a=(p/'a/speed_uncertainty.csv').read_bytes()
            self.assertEqual(a,(p/'b/speed_uncertainty.csv').read_bytes())
            rows=list(csv.DictReader(a.decode().splitlines()))
            self.assertEqual(len(rows),4)
            self.assertTrue(all(r['status']=='VALID' and float(r['p025_kmh'])<=float(r['p975_kmh']) for r in rows))
            self.assertIn('speed_uncertainty.csv',json.loads((p/'a/manifest.json').read_text())['files'])
            self.assertFalse(json.loads((p/'a/run_config.json').read_text())['uncertainty_model']['systematic_bias_included'])

    def test_fail_closed_unsupported_robust_mc(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            with self.assertRaisesRegex(ValueError,'ROBUST_UNCERTAINTY_UNSUPPORTED'):
                analyze(**KW,output=p/'invalid',robust=True,uncertainty_trials=20)
            self.assertFalse((p/'invalid').exists())

    def test_gap_invalid_and_no_interval_guessed(self):
        fit_uv=[(0,0),(2,0),(0,2),(2,2)]; fit_xy=fit_uv
        values=interval_uncertainties({('T',0):[(0.,(0.,0.),(0.,0.)),(8.,(1.,1.),(1.,1.))]},fit_uv,fit_xy,trials=20)
        self.assertEqual(values[0]['status'],'GAP')
        self.assertIsNone(values[0]['median_kmh'])
        with self.assertRaisesRegex(ValueError,'UNCERTAINTY_TRIALS_INVALID'):
            interval_uncertainties({},fit_uv,fit_xy,trials=3)

if __name__=='__main__':unittest.main()