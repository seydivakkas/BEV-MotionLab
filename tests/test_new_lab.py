import unittest
import numpy as np
from bevmotionlab.covariance_projection import project_pixel_covariance_to_bev
class Cases(unittest.TestCase):
 def test_jacobian_affine(self):
  H=[[.1,0,0],[0,.2,0],[0,0,1]]
  r=project_pixel_covariance_to_bev(10,15,[[4,0],[0,9]],H)
  self.assertEqual(r['status'],'VALID');self.assertAlmostEqual(r['covariance_m2'][0][0],.04)
 def test_horizon(self):
  r=project_pixel_covariance_to_bev(10,0,np.eye(2),[[1,0,0],[0,1,0],[.1,0,-1]])
  self.assertEqual(r['status'],'UNCERTAIN_PROJECTION')
 def test_reject_indefinite(self):
  with self.assertRaises(ValueError):project_pixel_covariance_to_bev(0,0,[[1,2],[2,1]],np.eye(3))
