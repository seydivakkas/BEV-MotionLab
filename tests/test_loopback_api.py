"""Authenticated local HTTP contract, numerical only."""
import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from bevmotionlab.api import server, dispatch
from bevmotionlab.loopback import create_server

DATA={'fit_pixel_px':[[0,0],[200,0],[0,200],[200,200]],
      'fit_world_m':[[0,0],[4,0],[0,4],[4,4]],
      'holdout_pixel_px':[[100,100]],'holdout_world_m':[[2,2]]}

class APIContractTests(unittest.TestCase):
    def test_calibrate_and_project_speeds(self):
        payload={'schema_version':'1.0','operation':'project_speed','input':{
            **DATA,'trajectory_pixel_px':[[50,50],[100,50]],'timestamps_s':[0.,1.]}}
        x=dispatch(payload)
        self.assertEqual(x['quality_gate']['status'],'NUMERICAL_PASS')
        self.assertAlmostEqual(x['speed_intervals'][0]['speed_kmh'],3.6)
        with self.assertRaises(ValueError):dispatch({**payload,'schema_version':'2.0'})
        with self.assertRaises(ValueError):dispatch({**payload,'input':{**payload['input'],'path':'/etc/passwd'}})

    def test_loopback_only_authenticated_no_origin(self):
        with self.assertRaisesRegex(ValueError,'LOOPBACK_ONLY'):
            create_server(dispatch,token='T'*30,host='0.0.0.0')
        http=server(token='T'*30,port=0)
        worker=threading.Thread(target=http.serve_forever,daemon=True)
        worker.start()
        base=f'http://127.0.0.1:{http.server_address[1]}'
        try:
            with self.assertRaises(HTTPError) as auth:
                urlopen(base+'/v1/health',timeout=2)
            self.assertEqual(auth.exception.code,403)
            data=json.dumps({'schema_version':'1.0','operation':'calibrate','input':DATA}).encode()
            req=Request(base+'/v1/execute',data=data,headers={'Authorization':'Bearer '+'T'*30,'Content-Type':'application/json'},method='POST')
            reply=json.load(urlopen(req,timeout=2))
            self.assertTrue(reply['result']['quality_gate']['passed'])
            reject=Request(base+'/v1/execute',data=data,headers={'Authorization':'Bearer '+'T'*30,'Content-Type':'application/json','Origin':'https://evil.example'},method='POST')
            with self.assertRaises(HTTPError) as origin: urlopen(reject,timeout=2)
            self.assertEqual(origin.exception.code,403)
        finally:
            http.shutdown();http.server_close();worker.join(timeout=2)

if __name__=='__main__':unittest.main()