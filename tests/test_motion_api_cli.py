"""Contract and artifact-integrity tests independent of HTTP/browser drivers."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from bevmotionlab.api import dispatch
from bevmotionlab.motion_cli import run


class BEVAPITests(unittest.TestCase):
    def setUp(self):
        here=Path(__file__).resolve().parents[1]
        self.calibration=json.loads((here/'examples/motion_calibration.json').read_text())
        self.samples=[json.loads(line) for line in (here/'examples/motion_samples.jsonl').read_text().splitlines()]

    def test_track_projection_api_backward_compatibility(self):
        req={'schema_version':'1.0','operation':'project_tracks','input':{
            'calibration':self.calibration,'samples':self.samples}}
        result=dispatch(req)
        self.assertEqual(result['counts']['samples'],8)
        self.assertEqual(result['calibration']['quality_gate']['status'],'NUMERICAL_PASS')
        self.assertEqual(result['records'][3]['status'],'UNCERTAIN_PROJECTION')
        self.assertTrue(result['records'][2]['speed_valid'])
        self.assertEqual(result['records'][4]['status'],'INITIALIZED')
        with self.assertRaises(ValueError):
            dispatch({**req,'schema_version':'2.0'})
        with self.assertRaises(ValueError):
            dispatch({'schema_version':'1.0','operation':'project_tracks',
                      'input':{**req['input'],'arbitrary_path':'/tmp/file'}})

    def test_api_rejects_camera_metadata_missing_and_samples_abuse(self):
        req={'schema_version':'1.0','operation':'project_tracks','input':{
            'calibration':self.calibration,'samples':self.samples}}
        bad={**self.calibration};bad.pop('lens_calibration_id')
        with self.assertRaises(ValueError):
            dispatch({**req,'input':{**req['input'],'calibration':bad}})
        with self.assertRaises(ValueError):
            dispatch({**req,'input':{**req['input'],'samples':self.samples*260}})
        with self.assertRaises(ValueError):
            dispatch({**req,'input':{**req['input'],'limits':{'max_major_axis_m':-1}}})

    def test_cli_manifest_hashes_and_refusal_to_overwrite(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'result'
            report=run(root/'examples/motion_calibration.json',root/'examples/motion_samples.jsonl',out)
            self.assertEqual(set(report['sha256']),{'motion_report.json','motion_records.jsonl'})
            for name,digest in report['sha256'].items():
                self.assertEqual(hashlib.sha256((out/name).read_bytes()).hexdigest(),digest)
            self.assertFalse(report['real_data_accepted'])
            with self.assertRaises(FileExistsError):
                run(root/'examples/motion_calibration.json',root/'examples/motion_samples.jsonl',out)

    def test_cli_blocks_before_output_on_bad_ground_point(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            td=Path(td)
            records=list(self.samples)
            records[1]={**records[1], 'cov_uv':[[1,3],[3,1]]}
            bad=td/'bad.jsonl';bad.write_text(''.join(json.dumps(x)+'\n' for x in records))
            out=td/'out'
            with self.assertRaises(ValueError):
                run(root/'examples/motion_calibration.json',bad,out)
            self.assertFalse(out.exists())

if __name__=='__main__':unittest.main()
