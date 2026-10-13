"""Numerically normalized planar homography, robust fitting and validation.

H maps *image pixel* coordinates to *world metres*. RANSAC distances are
computed in pixels via H^-1, not in metres with a pixel threshold.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def _points(a, name):
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2 or len(a) < 1 or not np.isfinite(a).all():
        raise ValueError(f"{name}: expected a finite Nx2 coordinate array")
    return a


def _nondegenerate(p):
    if len(p) < 4 or len(np.unique(p, axis=0)) < 4:
        return False
    centered = p - p.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    return bool(singular[0] > 0 and singular[1] / singular[0] > 1e-6)


def _normalize(points):
    centre = points.mean(axis=0)
    d = np.linalg.norm(points - centre, axis=1).mean()
    if d < 1e-12:
        raise ValueError("point spread is insufficient")
    s = np.sqrt(2.0) / d
    T = np.array([[s, 0, -s * centre[0]], [0, s, -s * centre[1]], [0, 0, 1.]])
    homogeneous = np.c_[points, np.ones(len(points))]
    return (T @ homogeneous.T).T[:, :2], T


def project(H, points):
    p = _points(points, "points")
    M = np.asarray(H, dtype=float)
    if M.shape != (3, 3) or not np.isfinite(M).all():
        raise ValueError("H must be finite 3x3")
    q = (M @ np.c_[p, np.ones(len(p))].T).T
    if np.any(np.abs(q[:, 2]) < 1e-12):
        raise ValueError("projection approaches the horizon; no finite ground coordinate")
    result = q[:, :2] / q[:, 2, None]
    if not np.isfinite(result).all():
        raise ValueError("non-finite projection")
    return result


def fit_dlt(image_px, world_m):
    src, dst = _points(image_px, "image_px"), _points(world_m, "world_m")
    if src.shape != dst.shape or not _nondegenerate(src) or not _nondegenerate(dst):
        raise ValueError("homography requires >=4 spread, non-collinear correspondences")
    a, Ts = _normalize(src)
    b, Td = _normalize(dst)
    rows = []
    for (x, y), (u, v) in zip(a, b):
        rows.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
        rows.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])
    A = np.asarray(rows)
    _, singular, Vt = np.linalg.svd(A, full_matrices=True)
    # 4 pairs => 8x9; rank 8 expected. More pairs => 2Nx9; rank >=8.
    if singular[0] == 0 or singular[min(7, len(singular)-1)] / singular[0] < 1e-11:
        raise ValueError("degenerate DLT design matrix")
    H = np.linalg.solve(Td, Vt[-1].reshape(3, 3) @ Ts)
    if abs(np.linalg.det(H)) < 1e-14 * max(1., np.linalg.norm(H)**3):
        raise ValueError("singular homography")
    H /= np.linalg.norm(H)
    # Canonicalize sign for reproducible artifacts; projective scale is arbitrary.
    if H.flat[np.argmax(np.abs(H))] < 0:
        H = -H
    return H


def reprojection_error_px(H, image_px, world_m):
    xy = _points(world_m, "world_m")
    uv = _points(image_px, "image_px")
    if uv.shape != xy.shape:
        raise ValueError("point arrays mismatch")
    predicted = project(np.linalg.inv(H), xy)
    return np.linalg.norm(predicted - uv, axis=1)


@dataclass(frozen=True)
class FitResult:
    H: np.ndarray
    inliers: np.ndarray
    fit_rmse_px: float
    inlier_count: int


def fit_ransac(image_px, world_m, *, threshold_px=2.0, seed=42, trials=500):
    src, dst = _points(image_px, "image_px"), _points(world_m, "world_m")
    if src.shape != dst.shape or len(src) < 4:
        raise ValueError("need >=4 matched points")
    if not np.isfinite(threshold_px) or threshold_px <= 0 or not 1 <= trials <= 10000:
        raise ValueError("invalid RANSAC parameter")
    rng = np.random.default_rng(seed)
    best = None
    best_count, best_error = 0, float('inf')
    for _ in range(trials):
        ids = rng.choice(len(src), 4, replace=False)
        try:
            H = fit_dlt(src[ids], dst[ids])
            errors = reprojection_error_px(H, src, dst)
        except (ValueError, np.linalg.LinAlgError):
            continue
        mask = errors <= threshold_px
        count = int(mask.sum())
        score = float(np.sum(errors[mask]**2))
        if count > best_count or (count == best_count and score < best_error):
            best, best_count, best_error = mask, count, score
    if best is None or best_count < 4:
        raise ValueError("RANSAC found fewer than four valid inliers")
    H = fit_dlt(src[best], dst[best])
    residual = reprojection_error_px(H, src[best], dst[best])
    return FitResult(H, best, float(np.sqrt(np.mean(residual**2))), int(best.sum()))


def holdout_report(H, image_px, world_m):
    src, dst = _points(image_px, "image_px"), _points(world_m, "world_m")
    if src.shape != dst.shape:
        raise ValueError("holdout shape mismatch")
    pred = project(H, src)
    world_err = np.linalg.norm(pred - dst, axis=1)
    pixel_err = reprojection_error_px(H, src, dst)
    return {"n": len(src), "world_rmse_m": float(np.sqrt(np.mean(world_err**2))),
            "reprojection_rmse_px": float(np.sqrt(np.mean(pixel_err**2)))}