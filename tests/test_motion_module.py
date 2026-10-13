"""Independent synthetic mathematical contracts for integrated BEV engine."""
import math
import unittest
import numpy as np

from bevmotionlab.motion_module import (
    BEVMotionModule, compute_homography_dlt,
    project_cross_frame_covariance_to_bev,
    extract_error_ellipse_parameters, calculate_physical_speed,
)
from bevmotionlab.covariance_projection import project_pixel_covariance_to_bev


def calibration():
    px=np.array([[0,0],[100,0],[100,100],[0,100],[50,20],[30,90.]],float)
    hold=np.array([[20,20],[80,20],[20,80],[80,80]],float)
    def world(x):return np.column_stack([x[:,0]*.05+3,x[:,1]*.08+5])
    return {'fit_pixel_px':px.tolist(),'fit_world_m':world(px).tolist(),
        'holdout_pixel_px':hold.tolist(),'holdout_world_m':world(hold).tolist(),
        'camera_id':'camera-synthetic-1','lens_calibration_id':'lens-synthetic-1',
        'measurement_provenance':'synthetic exact surveyed correspondences',
        'pixel_convention':'undistorted'}


def sample(t,u=40,v=40,**kwargs):
    return {'track_id':'T1','timestamp_s':t,'u':u,'v':v,
            'cov_uv':[[.04,0],[0,.04]],'contact_is_ground':True,
            'segment':'s0','camera_id':'camera-synthetic-1',
            'lens_calibration_id':'lens-synthetic-1',**kwargs}


class BEVMotionTests(unittest.TestCase):
    def test_dlt_jacobian_and_ellipse(self):
        model=BEVMotionModule.from_correspondences(**calibration())
        H=compute_homography_dlt(calibration()['fit_pixel_px'],calibration()['fit_world_m'])
        np.testing.assert_allclose(H/H[2,2],model.H/model.H[2,2],atol=1e-9)
        res=project_pixel_covariance_to_bev(30,40,[[4,0],[0,1]],H)
        np.testing.assert_allclose(res['covariance_m2'],[[.01,0],[0,.0064]],atol=1e-9)
        ell=extract_error_ellipse_parameters(res['covariance_m2'])
        self.assertAlmostEqual(ell['semi_major_m'],math.sqrt(5.991464547*.01))
        self.assertAlmostEqual(ell['semi_minor_m'],math.sqrt(5.991464547*.0064))
        self.assertGreaterEqual(ell['semi_major_m'],ell['semi_minor_m'])

    def test_projection_speed_and_acceleration(self):
        m=BEVMotionModule.from_correspondences(**calibration(),max_gap_s=.6)
        first=m.process_sample(sample(0.,u=40))
        second=m.process_sample(sample(.25,u=60))
        third=m.process_sample(sample(.5,u=85))
        self.assertEqual(first['status'],'INITIALIZED')
        self.assertEqual(second['status'],'VALID_CONDITIONAL')
        self.assertAlmostEqual(second['speed']['speed_kmh'],14.4,places=6)
        self.assertTrue(second['speed_valid'])
        self.assertEqual(len(third['acceleration_mps2']),2)
        self.assertGreater(third['acceleration_mps2'][0],0)
        self.assertEqual(m.evidence.numerical_quality_gate['status'],'NUMERICAL_PASS')
        self.assertFalse(second['speed']['systematic_errors_included'])
        self.assertFalse(second['speed']['cross_frame_correlation_accounted'])

    def test_covariance_cross_time_not_invented(self):
        p0={'timestamp_s':0.,'x_m':0.,'y_m':0.,'covariance_m2':[[1,0],[0,1]]}
        p1={'timestamp_s':.25,'x_m':1.,'y_m':0.,'covariance_m2':[[1,0],[0,1]]}
        approx=calculate_physical_speed(p0,p1,max_speed_sigma_kmh=200.)
        corr=calculate_physical_speed(p0,p1,cross_covariance_m2=[[1,0],[0,1]],max_speed_sigma_kmh=200.)
        self.assertTrue(approx['speed_valid'])
        self.assertGreater(approx['speed_sigma_kmh'],corr['speed_sigma_kmh'])
        self.assertTrue(corr['cross_frame_correlation_accounted'])
        with self.assertRaises(ValueError):
            calculate_physical_speed(p0,p1,cross_covariance_m2=[[10,0],[0,10]],max_speed_sigma_kmh=200.)

    def test_fail_closed_occlusion_contact_gap_segment(self):
        m=BEVMotionModule.from_correspondences(**calibration(),max_gap_s=.3)
        self.assertEqual(m.process_sample(sample(0.))['status'],'INITIALIZED')
        self.assertEqual(m.process_sample(sample(.1,contact_is_ground=False))['reason'],'UNCERTAIN_GROUND_CONTACT')
        self.assertEqual(m.process_sample(sample(.2))['status'],'INITIALIZED')
        self.assertEqual(m.process_sample(sample(.3,occluded=True))['reason'],'OCCLUSION')
        self.assertEqual(m.process_sample(sample(.4))['status'],'INITIALIZED')
        self.assertEqual(m.process_sample(sample(1.,u=60))['status'],'GAP')
        self.assertEqual(m.process_sample(sample(1.1,u=64,segment='new'))['status'],'INITIALIZED')
        self.assertEqual(m.process_sample(sample(1.2,u=68,identity_discontinuity=True))['reason'],'IDENTITY_DISCONTINUITY')
        self.assertEqual(m.process_sample(sample(1.3,u=70))['status'],'INITIALIZED')
        self.assertEqual(m.process_sample(sample(1.4,u=105))['reason'],'OUTSIDE_CALIBRATED_ROI')

    def test_cross_frame_jacobian_product(self):
        m=BEVMotionModule.from_correspondences(**calibration())
        cross=project_cross_frame_covariance_to_bev(30,20,40,20,[[2,.4],[.5,3]],m.H)
        np.testing.assert_allclose(cross,[[.005,.0016],[.002, .0192]],atol=1e-10)

    def test_hull_excludes_unobserved_triangle_corner(self):
        data=calibration()
        data['fit_pixel_px']=[[0,0],[100,0],[0,100],[15,15]]
        def project(pts):return [[.05*x+3,.08*y+5] for x,y in pts]
        data['fit_world_m']=project(data['fit_pixel_px'])
        data['holdout_pixel_px']=[[10,10],[40,10],[10,40],[25,25]]
        data['holdout_world_m']=project(data['holdout_pixel_px'])
        m=BEVMotionModule.from_correspondences(**data)
        out=m.process_sample(sample(0.,u=80,v=80))
        self.assertEqual(out['reason'],'OUTSIDE_CALIBRATED_ROI')

    def test_strict_horizon_behind_camera(self):
        m=BEVMotionModule.from_correspondences(**calibration())
        m.H[2,:]=[-.02,0,1]
        self.assertEqual(m.process_sample(sample(0.,u=60))['reason'],'NEAR_OR_BEHIND_HORIZON')
        # Numerical-only Jacobian function can be used separately; module enforces orientation.

    def test_invalid_non_psd_threshold_and_sample_schema(self):
        with self.assertRaises(ValueError):extract_error_ellipse_parameters([[1,2],[2,1]])
        with self.assertRaises(ValueError):BEVMotionModule.from_correspondences(**{**calibration(),'pixel_convention':'raw_distorted'})
        with self.assertRaises(ValueError):BEVMotionModule.from_correspondences(**{**calibration(),'holdout_world_m':[[0,0]]*4})
        m=BEVMotionModule.from_correspondences(**calibration(),max_major_axis_m=.01)
        out=m.process_sample(sample(0.,u=40))
        self.assertEqual(out['status'],'UNCERTAIN_PROJECTION')
        self.assertFalse(out['speed_valid'])
        with self.assertRaises(ValueError):m.process_sample(sample(0.,cov_uv=[[1,3],[3,1]]))
        with self.assertRaises(ValueError):m.process_sample(sample(0.,unexpected='no'))

    def test_speed_outliers_uncertain_and_timestamp(self):
        m=BEVMotionModule.from_correspondences(**calibration(),max_speed_kmh=10.,max_speed_sigma_kmh=35.)
        m.process_sample(sample(0.,u=40))
        speed=m.process_sample(sample(.01,u=90))
        self.assertEqual(speed['status'],'SPEED_OUTLIER')
        with self.assertRaises(ValueError):m.process_sample(sample(.005,u=91))
        m.reset_track('T1')
        self.assertEqual(m.process_sample(sample(.02,u=91))['status'],'INITIALIZED')
        m.reset_track('T1')
        x=m.process_sample(sample(0.,u=40))
        y=m.process_sample(sample(.1,u=40))
        self.assertEqual(y['status'],'UNCERTAIN_PROJECTION')
        self.assertEqual(y['speed']['reason'],'ZERO_DISPLACEMENT_DIRECTION_UNDEFINED')

    def test_provenance_mismatch(self):
        m=BEVMotionModule.from_correspondences(**calibration())
        out=m.process_sample(sample(0.,camera_id='other'))
        self.assertEqual(out['reason'],'CAMERA_LENS_MISMATCH')
        self.assertFalse(out['speed_valid'])

if __name__=='__main__':unittest.main()
