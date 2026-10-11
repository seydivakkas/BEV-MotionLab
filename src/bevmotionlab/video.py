"""Optional local FFprobe PTS import; never infer timestamps from nominal FPS."""
from __future__ import annotations
import json
import subprocess
from pathlib import Path
import numpy as np


def validate_pts(payload):
    """Parse ffprobe JSON, retaining the original presentation timestamp order."""
    if not isinstance(payload,dict) or not isinstance(payload.get('frames'),list):
        raise ValueError('ffprobe JSON must contain frames array')
    values=[]
    for index,frame in enumerate(payload['frames']):
        if not isinstance(frame,dict):raise ValueError(f'frame {index}: invalid')
        stamp=frame.get('best_effort_timestamp_time',frame.get('pkt_pts_time'))
        if stamp is None or stamp=='N/A':raise ValueError(f'frame {index}: missing PTS')
        try: t=float(stamp)
        except (ValueError,TypeError) as exc: raise ValueError(f'frame {index}: invalid PTS') from exc
        if not np.isfinite(t):raise ValueError(f'frame {index}: nonfinite PTS')
        if values and t<=values[-1]:raise ValueError(f'frame {index}: nonmonotonic or duplicate PTS')
        values.append(t)
    if not values:raise ValueError('video has no frame timestamps')
    diffs=np.diff(values)
    return {'frame_count':len(values),'timestamps_s':values,
            'minimum_dt_s':float(diffs.min()) if len(diffs) else None,
            'maximum_dt_s':float(diffs.max()) if len(diffs) else None,
            'variable_frame_period':bool(len(diffs)>1 and not np.allclose(diffs,diffs[0],rtol=1e-4,atol=1e-6))}


def read_ffprobe_pts(video_path, *, executable='ffprobe', timeout_s=30):
    path=Path(video_path)
    if not path.is_file():raise ValueError('video file missing')
    if timeout_s<=0 or timeout_s>300:raise ValueError('invalid timeout')
    cmd=[executable,'-v','error','-select_streams','v:0','-show_entries',
         'frame=best_effort_timestamp_time,pkt_pts_time','-of','json',str(path)]
    result=subprocess.run(cmd,check=True,timeout=timeout_s,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    return validate_pts(json.loads(result.stdout))


def validate_detection_timestamps(records, pts_report, *, tolerance_s=0.005, offset_s=0.):
    """Audit frame-to-PTS correspondence, never replacing supplied timestamps.

    `offset_s` must be explicitly known; no automatic FPS/time-origin guessing.
    Multiple tracks may share the same frame and PTS, but may not disagree.
    """
    import math
    if not math.isfinite(tolerance_s) or tolerance_s < 0:
        raise ValueError('invalid PTS tolerance')
    if not math.isfinite(offset_s):
        raise ValueError('non-finite PTS time-origin offset')
    timestamps = pts_report['timestamps_s']
    audited = 0
    seen = {}
    max_error = 0.
    for record in records:
        frame, t = record['frame'], record['timestamp_s']
        if type(frame) is not int or not 0 <= frame < len(timestamps) or not math.isfinite(t):
            raise ValueError('PTS_FRAME_OUT_OF_RANGE_OR_INVALID')
        if frame in seen and not math.isclose(t, seen[frame], abs_tol=1e-9, rel_tol=0):
            raise ValueError('PTS_CONFLICTING_TIMES_FOR_FRAME')
        seen[frame] = t
        err = abs(t - (timestamps[frame] + offset_s))
        if err > tolerance_s + 1e-12:
            raise ValueError(f'PTS_MISMATCH frame={frame} error_s={err:.9g}')
        max_error = max(max_error, err)
        audited += 1
    if not audited:
        raise ValueError('PTS_NO_DETECTIONS')
    return {'status':'MATCHED','checked_observations':audited,
            'distinct_frames':len(seen),'max_abs_error_s':max_error,
            'tolerance_s':tolerance_s,'explicit_offset_s':offset_s,
            'source':'ffprobe frame presentation timestamps'}