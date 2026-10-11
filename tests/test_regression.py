import csv
import json
import tempfile
import unittest
from pathlib import Path
from bevmotionlab.cli import analyze
from bevmotionlab.speed import estimate_speed

class RegressionTests(unittest.TestCase):
    def test_invalid_contact_does_not_get_connected_across_gap(self):
        points=[[100,200],[300,200],[550,400],[350,400]]
        pairs=[{'id':str(i),'image_uv_px':p,'world_xy_m':[p[0]/10,p[1]/10],'split':'fit'} for i,p in enumerate(points)]
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);cal=p/'cal.json';tracks=p/'tracks.csv'
            cal.write_text(json.dumps({'schema_version':'1.0','camera_id':'c1','resolution_px':[800,600],'coordinate_frame':{'units':'m'},'pixel_convention':'undistorted','lens_calibration_id':'undistorted-preset','calibrated_at_utc':'2026-10-11T12:00:00Z','measurement_provenance':'synthetic','correspondences':pairs,'valid_roi_uv_px':[[50,50],[600,500]]}))
            with tracks.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['frame','timestamp_s','track_id','x1_px','x2_px','y1_px','y2_px','confidence']);w.writeheader()
                for i,(t,x) in enumerate([(0,150),(.1,170),(.2,730),(.3,200),(.4,220)]):
                    w.writerow({'frame':i,'timestamp_s':t,'track_id':'7','x1_px':x-10,'x2_px':x+10,'y1_px':120,'y2_px':210,'confidence':1.})
            result=analyze(cal,tracks,p/'out',camera_id='c1')
            self.assertEqual(result['flags'],1)
            self.assertEqual(result['speed_intervals'],2)
            self.assertTrue((p/'out'/'bev_trajectory.svg').exists())

    def test_invalid_temporal_inputs(self):
        with self.assertRaises(ValueError):estimate_speed([1,0],[[0,0],[1,0]])
        with self.assertRaises(ValueError):estimate_speed([0,1],[[0,0],[float('nan'),0]])