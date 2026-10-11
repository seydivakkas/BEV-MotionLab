import unittest
from bevmotionlab.video import validate_pts,read_ffprobe_pts

class VideoTimestampTests(unittest.TestCase):
 def test_vfr_pts(self):
  d=validate_pts({'frames':[{'best_effort_timestamp_time':x} for x in ('0.00','0.033','0.080','0.113')]})
  self.assertTrue(d['variable_frame_period'])
  self.assertEqual(d['frame_count'],4)
  self.assertAlmostEqual(d['maximum_dt_s'],.047)
 def test_invalid_pts(self):
  for arr in ([],[{'best_effort_timestamp_time':'nan'}],[{'best_effort_timestamp_time':'0'},{'best_effort_timestamp_time':'0'}],[{}]):
   with self.assertRaises(ValueError):validate_pts({'frames':arr})
 def test_requires_local_video(self):
  with self.assertRaises(ValueError):read_ffprobe_pts('/path/that/does/not/exist.mp4')