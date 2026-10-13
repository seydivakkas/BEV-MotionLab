"""Offline JSONL BEV batch CLI: no detector, no camera/field accuracy claim.

Usage:
python -m bevmotionlab.motion_cli --calibration examples/motion_calibration.json \\
    --input examples/motion_samples.jsonl --out output/motion
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from .api import dispatch


def _encode(obj):
    return (json.dumps(obj, ensure_ascii=False, indent=2,
                       sort_keys=True, allow_nan=False)+'\n').encode('utf-8')


def run(calibration_file, samples_file, out_dir, *, ransac=False,
        max_major_axis_m=3., max_gap_s=.5, max_speed_kmh=240.,
        max_speed_sigma_kmh=35.):
    """Materialize *only after* entire input is accepted and computed."""
    out=Path(out_dir)
    if out.exists():raise FileExistsError('OUTPUT_DIRECTORY_ALREADY_EXISTS')
    cal_raw=Path(calibration_file).read_bytes()
    sample_raw=Path(samples_file).read_bytes()
    if len(cal_raw)>100000 or len(sample_raw)>3000000:
        raise ValueError('INPUT_SIZE_EXCEEDED')
    calibration=json.loads(cal_raw)
    lines=sample_raw.decode('utf8').splitlines()
    if not lines or len(lines)>2000 or any(not line.strip() for line in lines):
        raise ValueError('INVALID_JSONL_LENGTH_OR_EMPTY_LINE')
    samples=[json.loads(line) for line in lines]
    res=dispatch({'schema_version':'1.0','operation':'project_tracks',
                  'input':{'calibration':calibration,'samples':samples,'ransac':ransac,
                           'limits':{'max_major_axis_m':max_major_axis_m,
                                     'max_gap_s':max_gap_s,
                                     'max_speed_kmh':max_speed_kmh,
                                     'max_speed_sigma_kmh':max_speed_sigma_kmh}}})
    files={'motion_report.json':_encode(res),
           'motion_records.jsonl':b''.join(_encode(row) for row in res['records'])}
    manifest={'schema_version':'1.0','run':'BEV_CONDITIONAL_SYNTHETIC_FIRST',
              'input_sha256':{'calibration_json':hashlib.sha256(cal_raw).hexdigest(),
                              'samples_jsonl':hashlib.sha256(sample_raw).hexdigest()},
              'sha256':{name:hashlib.sha256(payload).hexdigest() for name,payload in files.items()},
              'numerical_quality_pass':res['calibration']['quality_gate']['passed'],
              'real_data_accepted':False,'field_speed_accuracy_claim':False}
    files['manifest.json']=_encode(manifest)
    out.mkdir(parents=True,exist_ok=False)
    try:
        for name,payload in files.items():(out/name).write_bytes(payload)
    except Exception:
        shutil.rmtree(out)
        raise
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description='Offline BEV physical geometry lab, synthetic-first')
    parser.add_argument('--calibration',required=True)
    parser.add_argument('--input',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--ransac',action='store_true')
    parser.add_argument('--max-major-axis-m',type=float,default=3.)
    parser.add_argument('--max-gap-s',type=float,default=.5)
    parser.add_argument('--max-speed-kmh',type=float,default=240.)
    parser.add_argument('--max-speed-sigma-kmh',type=float,default=35.)
    a=parser.parse_args(argv)
    print(json.dumps(run(a.calibration,a.input,a.out,ransac=a.ransac,
                         max_major_axis_m=a.max_major_axis_m,max_gap_s=a.max_gap_s,
                         max_speed_kmh=a.max_speed_kmh,
                         max_speed_sigma_kmh=a.max_speed_sigma_kmh),sort_keys=True))

if __name__=='__main__':main()
