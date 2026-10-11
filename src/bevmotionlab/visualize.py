"""Dependency-free SVG rendering of *valid* ground-plane track points."""
from __future__ import annotations
import csv
from html import escape
from pathlib import Path


def render_bev_svg(projected_csv, output_svg):
    with Path(projected_csv).open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    valid=[r for r in rows if r['status']=='VALID' and r['x_m'] and r['y_m']]
    if not valid:raise ValueError('no valid projected points to draw')
    xmin=min(float(r['x_m']) for r in valid);xmax=max(float(r['x_m']) for r in valid)
    ymin=min(float(r['y_m']) for r in valid);ymax=max(float(r['y_m']) for r in valid)
    dx=max(xmax-xmin,1e-6);dy=max(ymax-ymin,1e-6)
    sx=lambda v:60.+(v-xmin)/dx*680.
    sy=lambda v:480.-(v-ymin)/dy*420.
    tracks={}
    for r in valid:tracks.setdefault(r['track_id'],[]).append((float(r['timestamp_s']),float(r['x_m']),float(r['y_m'])))
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="550" viewBox="0 0 800 550">',
           '<rect width="800" height="550" fill="white"/>',
           '<text x="55" y="28" font-family="sans-serif" font-size="19">BEV-MotionLab: world-plane trajectory (metres)</text>',
           '<path d="M60 60 V480 H740" stroke="#333" fill="none"/>']
    palette=['#125cb2','#b24725','#28815e','#7d4db8','#947600']
    for idx,(tid,pts) in enumerate(sorted(tracks.items())):
        pts=sorted(pts)
        xy=' '.join(f'{sx(x):.2f},{sy(y):.2f}' for _,x,y in pts)
        col=palette[idx%len(palette)]
        parts.append(f'<polyline points="{xy}" fill="none" stroke="{col}" stroke-width="2"/>')
        _,x,y=pts[-1]
        parts.append(f'<text x="{sx(x)+3:.1f}" y="{sy(y)-5:.1f}" font-family="sans-serif" font-size="13">{escape(tid)}</text>')
    parts.append(f'<text x="55" y="520" font-family="sans-serif" font-size="12">X [{xmin:.2f}, {xmax:.2f}] m; Y [{ymin:.2f}, {ymax:.2f}] m — 2D planar proxy, not verified ground truth.</text>')
    parts.append('</svg>')
    path=Path(output_svg);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('\n'.join(parts)+'\n',encoding='utf-8')
    return path