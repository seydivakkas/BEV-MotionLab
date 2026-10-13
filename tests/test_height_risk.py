import unittest
from bevmotionlab.height_risk import ground_bias_bound,speed_bias_bound
def camera():
    return {'schema_version':'1.0','world_units':'m','plane_z_m':0,
            'lens_calibration_id':'synthetic','pixel_convention':'undistorted',
            'K':[[120,0,200],[0,120,160],[0,0,1]],
            'R_cw':[[1,0,0],[0,-1,0],[0,0,-1]],
            'camera_center_world_m':[0,0,10],
            'distortion':dict(k1=0.,k2=0.,p1=0.,p2=0.,k3=0.)}
class HeightRiskTests(unittest.TestCase):
    def test_nadir(self):
        r=ground_bias_bound(camera(),[200,160],maximum_height_m=1.)
        self.assertAlmostEqual(r['xy_position_bias_upper_bound_m'],0.)
        self.assertFalse(r['physical_height_measured'])
    def test_acute_ray_speed(self):
        r=speed_bias_bound(camera(),[200,160],[320,160],elapsed_s=1.,maximum_height_m=1.)
        self.assertAlmostEqual(r['speed_bias_upper_bound_kmh'],3.6)
    def test_scale(self):
        a=speed_bias_bound(camera(),[320,160],[200,160],elapsed_s=1,maximum_height_m=.5)
        b=speed_bias_bound(camera(),[320,160],[200,160],elapsed_s=.5,maximum_height_m=1.)
        self.assertAlmostEqual(b['speed_bias_upper_bound_kmh'],4*a['speed_bias_upper_bound_kmh'])
    def test_fail_closed(self):
        for h in (-1,float('nan'),10,True):
            with self.assertRaises(ValueError):
                ground_bias_bound(camera(),[320,160],maximum_height_m=h)
        with self.assertRaisesRegex(ValueError,'INVALID_INTERVAL'):
            speed_bias_bound(camera(),[200,160],[210,160],elapsed_s=0,maximum_height_m=.5)
    def test_horizon(self):
        m=camera();m['R_cw']=[[1,0,0],[0,0,-1],[0,1,0]]
        with self.assertRaisesRegex(ValueError,'FORWARD_GROUND'):
            ground_bias_bound(m,[200,160],maximum_height_m=.2)
