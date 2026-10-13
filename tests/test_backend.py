import csv
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from bevmotionlab.geometry import fit_dlt, fit_ransac, project, holdout_report
from bevmotionlab.speed import estimate_speed, monte_carlo_interval
from bevmotionlab.cli import analyze, calibrate

class BackendTests(unittest.TestCase):
    def setUp(self):
        self.H=np.array([[.012, .001,-5],[-.002,.017,-8],[.0001,.0002,1.]])
        self.uv=np.array([[50,50],[600,50],[50,500],[600,500],[400,260],[200,310],[320,400],[550,200]],float)
        self.xy=project(self.H,self.uv)

    def test_noise_free_dlt_and_holdout(self):
        H=fit_dlt(self.uv[:6],self.xy[:6])
        np.testing.assert_allclose(project(H,self.uv[6:]),self.xy[6:],atol=1e-9)
        self.assertLess(holdout_report(H,self.uv[6:],self.xy[6:])['world_rmse_m'],1e-9)

    def test_outlier_ransac_pixel_units(self):
        xy=self.xy.copy();xy[-1]+=np.array([6.,12.])
        f=fit_ransac(self.uv,xy,threshold_px=.1,seed=12,trials=250)
        self.assertEqual(f.inlier_count,7)
        self.assertFalse(f.inliers[-1])

    def test_degeneracy(self):
        with self.assertRaises(ValueError):fit_dlt([[0,0],[1,1],[2,2],[3,3]],[[0,0],[1,0],[0,1],[1,1]])
        with self.assertRaises(ValueError):project(self.H,[[np.nan,2]])

    def test_speed_irregular(self):
        r=estimate_speed([0.,1.,1.5],[[0,0],[10,0],[12,0]])
        self.assertAlmostEqual(r[0]['speed_kmh'],36.)
        self.assertAlmostEqual(r[1]['speed_kmh'],14.4)
        self.assertEqual(estimate_speed([0,3],[[0,0],[10,0]])[0]['status'],'GAP')
        with self.assertRaises(ValueError):estimate_speed([0,0],[[0,0],[1,0]])

    def test_calibration_and_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td);cal=td/'cal.json';dets=td/'obs.csv'
            marks=[{'id':str(i),'image_uv_px':p.tolist(),'world_xy_m':q.tolist(),'split':'fit' if i<6 else 'holdout'} for i,(p,q) in enumerate(zip(self.uv,self.xy))]
            cal.write_text(json.dumps({'schema_version':'1.0','camera_id':'cam1','resolution_px':[800,600],
              'coordinate_frame':{'units':'m'},'pixel_convention':'undistorted','lens_calibration_id':'fixed-preprocessed',
              'calibrated_at_utc':'2026-10-11T12:00:00Z','measurement_provenance':'synthetic',
              'correspondences':marks,'valid_roi_uv_px':[[0,0],[800,600]]}),encoding='utf-8')
            with dets.open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=['frame','timestamp_s','track_id','x1_px','y1_px','x2_px','y2_px','confidence']);w.writeheader()
                for i,x in enumerate((110,130,160)):
                    w.writerow(dict(frame=i,timestamp_s=[0,.5,1.5][i],track_id='a',x1_px=x-10,y1_px=200,x2_px=x+10,y2_px=300,confidence=.9))
            result=analyze(cal,dets,td/'run',camera_id='cam1')
            self.assertEqual(result['speed_intervals'],2)
            self.assertEqual(result['flags'],0)
            self.assertLess(result['calibration']['holdout']['world_rmse_m'],1e-8)
            self.assertTrue((td/'run'/'manifest.json').exists())
            with self.assertRaisesRegex(ValueError,'CAMERA_MISMATCH'):
                analyze(cal,dets,td/'wrong',camera_id='cam2')

    def test_uncertainty_seed_replay(self):
        kwargs=dict(fit_pixel_px=self.uv[:6],fit_world_m=self.xy[:6],trajectory_pixel_px=self.uv[:2],times_s=[0,1],trials=30,seed=7)
        a=monte_carlo_interval(**kwargs);b=monte_carlo_interval(**kwargs)
        self.assertEqual(a,b)
        self.assertLess(a['p025_kmh'],a['p975_kmh'])

if __name__=='__main__':unittest.main()