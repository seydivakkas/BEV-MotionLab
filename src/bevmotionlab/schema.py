"""Strict local CSV/JSON input validation (no automatic guesses on units)."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import math


def finite(v, label):
    try: x = float(v)
    except (TypeError, ValueError) as e: raise ValueError(f"{label} is not numeric") from e
    if not math.isfinite(x): raise ValueError(f"{label} is not finite")
    return x


@dataclass(frozen=True)
class Detection:
    frame: int
    timestamp_s: float
    track_id: str
    box: tuple[float, float, float, float]
    confidence: float

    @classmethod
    def from_csv(cls, row, resolution):
        frame = int(row['frame'])
        t = finite(row['timestamp_s'], 'timestamp_s')
        x1,y1,x2,y2 = (finite(row[key], key) for key in ('x1_px','y1_px','x2_px','y2_px'))
        conf = finite(row['confidence'], 'confidence')
        w,h = resolution
        if frame < 0 or not 0 <= x1 < x2 <= w or not 0 <= y1 < y2 <= h or not 0 < conf <= 1:
            raise ValueError('invalid box, frame or confidence')
        tid = str(row['track_id'])
        if not tid: raise ValueError('empty track_id')
        return cls(frame,t,tid,(x1,y1,x2,y2),conf)

    @property
    def foot_px(self):
        x1,y1,x2,y2 = self.box
        return ((x1+x2)/2,y2)


def load_calibration(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('schema_version') not in ('1.0-design', '1.0'):
        raise ValueError('unsupported calibration schema version')
    if data.get('pixel_convention') != 'undistorted':
        raise ValueError('only previously undistorted image points supported')
    if not data.get('lens_calibration_id'):
        raise ValueError('missing lens calibration provenance')
    if not data.get('camera_id'):
        raise ValueError('camera_id is required')
    resolution = data['resolution_px']
    if len(resolution) != 2 or any(isinstance(v,bool) or not isinstance(v,int) or v<=0 for v in resolution):
        raise ValueError('resolution must be positive integer width,height')
    if data.get('coordinate_frame', {}).get('units') != 'm':
        raise ValueError('world coordinates must use metres')
    if not data.get('measurement_provenance'):
        raise ValueError('measurement provenance required')
    try: dt=datetime.fromisoformat(data['calibrated_at_utc'].replace('Z','+00:00'))
    except (KeyError,ValueError,TypeError) as e: raise ValueError('invalid calibration timestamp') from e
    if dt.tzinfo is None: raise ValueError('calibration time needs timezone')
    ids=set(); fits=[]; holds=[]
    for c in data['correspondences']:
        if c['id'] in ids: raise ValueError('duplicate landmark ID')
        ids.add(c['id'])
        uv = tuple(finite(v,'image_uv_px') for v in c['image_uv_px'])
        xy = tuple(finite(v,'world_xy_m') for v in c['world_xy_m'])
        if len(uv)!=2 or len(xy)!=2: raise ValueError('landmarks must be 2D')
        if not 0 <= uv[0] <= resolution[0] or not 0 <= uv[1] <= resolution[1]:
            raise ValueError('landmark outside resolution')
        split=c['split']
        if split not in ('fit','holdout'): raise ValueError('split must be fit or holdout')
        (fits if split=='fit' else holds).append((uv,xy))
    if len(fits)<4: raise ValueError('at least four fit landmarks required')
    roi = data.get('valid_roi_uv_px')
    if roi is not None:
        if len(roi)!=2 or len(roi[0])!=2 or len(roi[1])!=2:
            raise ValueError('invalid ROI')
        (a,b),(c,d)=roi
        if not 0<=a<c<=resolution[0] or not 0<=b<d<=resolution[1]:
            raise ValueError('ROI outside image')
    return data, fits, holds