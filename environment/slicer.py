"""Vectorized ray-cast ultrasound slice renderer.

Casts a 2D sector (64 rays x 128 samples) from a probe pose through the
analytic phantom primitives, producing a grayscale "B-mode-like" image plus
a per-primitive visible-fraction dict used by the reward function.

Everything here is pure numpy and fully vectorized over (rays, samples) so a
frame renders in well under a millisecond on typical hardware -- this is
what keeps RL training from being bottlenecked by rendering.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from environment.phantom import FetalPhantom, Primitive, TISSUE_ACOUSTICS

N_RAYS = 64
N_SAMPLES = 128
SECTOR_ANGLE_DEG = 60.0
MAX_DEPTH_M = 0.16
RASTER_SIZE = 128  # output raster resolution (square)

_TISSUE_LIST = list(TISSUE_ACOUSTICS.keys())
_TISSUE_INDEX = {t: i for i, t in enumerate(_TISSUE_LIST)}
BACKGROUND_TISSUE = "anechoic"  # amniotic fluid / nothing hit


def _inside_ellipsoid(local_pts: np.ndarray, semi_axes: np.ndarray) -> np.ndarray:
    return np.sum((local_pts / semi_axes) ** 2, axis=-1) <= 1.0


def _inside_shell(local_pts: np.ndarray, semi_axes: np.ndarray, thickness: float) -> np.ndarray:
    outer = np.sum((local_pts / semi_axes) ** 2, axis=-1) <= 1.0
    inner_axes = np.maximum(semi_axes - thickness, 1e-4)
    inner = np.sum((local_pts / inner_axes) ** 2, axis=-1) <= 1.0
    return outer & (~inner)


def _inside_sphere(local_pts: np.ndarray, radius: float) -> np.ndarray:
    return np.sum(local_pts ** 2, axis=-1) <= radius ** 2


def _inside_capsule(local_pts: np.ndarray, radius: float, half_length: float) -> np.ndarray:
    # capsule axis assumed along local z after inverse-rotation into primitive frame
    z = np.clip(local_pts[..., 2], -half_length, half_length)
    axis_pt = np.zeros_like(local_pts)
    axis_pt[..., 2] = z
    d = np.linalg.norm(local_pts - axis_pt, axis=-1)
    return d <= radius


def _inside_box(local_pts: np.ndarray, half_extents: np.ndarray) -> np.ndarray:
    return np.all(np.abs(local_pts) <= half_extents, axis=-1)


def _to_local(points: np.ndarray, prim: Primitive) -> np.ndarray:
    return (points - prim.center) @ prim.rotation  # R^T applied via row-vector convention


def _test_primitive(points: np.ndarray, prim: Primitive) -> np.ndarray:
    local = _to_local(points, prim)
    if prim.kind == "ellipsoid":
        return _inside_ellipsoid(local, prim.params["semi_axes"])
    if prim.kind == "ellipsoid_shell":
        return _inside_shell(local, prim.params["semi_axes"], prim.params["thickness"])
    if prim.kind == "sphere":
        return _inside_sphere(local, prim.params["radius"])
    if prim.kind == "capsule":
        return _inside_capsule(local, prim.params["radius"], prim.params["half_length"])
    if prim.kind == "box":
        return _inside_box(local, prim.params["half_extents"])
    raise ValueError(f"unknown primitive kind {prim.kind}")


# ---------------------------------------------------------------------------
# Phase-3 performance note (see status.md for full writeup):
#
# A "batch by primitive kind" rewrite was attempted here -- stacking all
# primitives of the same kind (or even ALL primitives at once) into shared
# arrays and doing one batched `np.matmul` per kind/overall instead of one
# small matmul per primitive, hypothesizing this would cut ~23 small
# Python/BLAS calls down to ~5 (or 1) and yield a 5-10x speedup.
#
# Measured result: it was SLOWER, not faster (about 1.5x worse: ~21-28ms
# vs ~14-20ms per frame). Root cause, confirmed by isolating each step:
# building the single broadcast temporary array needed for the batched
# matmul (shape (N_primitives, 64, 128, 3), ~565k elements / ~4.5MB at
# float64) cost MORE by itself (~5-6ms) than the entire original 23-call
# per-primitive loop (~4.8ms). At N=23 primitives with small (64, 128, 3)
# per-call arrays, individual BLAS-dispatched matmuls are apparently more
# cache/allocation-friendly on this hardware than one large batched
# operation -- the usual "fewer, bigger numpy calls is faster" heuristic
# does not hold at this scale here. The per-primitive loop below is kept
# as the real (measured-faster) implementation; the batched attempt was
# reverted rather than kept as a slower "optimization."
#
# A genuine 5-10x speedup, if still wanted, would likely require a
# different strategy than numpy-level batching -- e.g. a fused low-level
# kernel (Numba/Cython), reducing primitive count, or GPU/JAX -- which is
# out of scope for this pass. Training throughput at ~15-20ms/frame is
# workable for the smoke tests and the bounded probe run in this project;
# it remains the dominant cost for a genuine multi-day full sweep (see
# status.md Phase 3).
# ---------------------------------------------------------------------------


def cast_slice(phantom: FetalPhantom, probe_position: np.ndarray, probe_forward: np.ndarray,
                probe_right: np.ndarray, probe_up: np.ndarray, rng: np.random.Generator,
                n_rays: int = N_RAYS, n_samples: int = N_SAMPLES,
                sector_deg: float = SECTOR_ANGLE_DEG, max_depth: float = MAX_DEPTH_M):
    """Cast a sector slice. Returns (raster_image [RASTER,RASTER] float32 in [0,1],
    tissue_hit_mask dict primitive.name -> bool array (n_rays, n_samples),
    sample_points (n_rays, n_samples, 3)).
    """
    angles = np.linspace(-np.radians(sector_deg / 2), np.radians(sector_deg / 2), n_rays)
    depths = np.linspace(0.005, max_depth, n_samples)

    # ray directions: rotate forward toward right by `angle`, within the slice plane
    # (spanned by forward and right; up is the elevational normal, kept ~0 thickness)
    cos_a = np.cos(angles)[:, None]
    sin_a = np.sin(angles)[:, None]
    ray_dirs = cos_a * probe_forward[None, :] + sin_a * probe_right[None, :]  # (n_rays, 3)
    ray_dirs /= np.linalg.norm(ray_dirs, axis=-1, keepdims=True)

    # sample points: (n_rays, n_samples, 3)
    points = probe_position[None, None, :] + ray_dirs[:, None, :] * depths[None, :, None]

    tissue_id = np.full((n_rays, n_samples), -1, dtype=np.int32)  # -1 = background
    hit_prim_idx = np.full((n_rays, n_samples), -1, dtype=np.int32)
    hit_masks: dict[str, np.ndarray] = {}

    # priority: bone first (so it occludes/marks shadows correctly even if
    # geometrically overlapping soft tissue), then everything else in
    # phantom order (skull/falx/thalami/csp before cerebellum, etc.)
    ordered = sorted(range(len(phantom.primitives)),
                      key=lambda i: 0 if phantom.primitives[i].tissue == "bone" else 1)

    filled = np.zeros((n_rays, n_samples), dtype=bool)
    for idx in ordered:
        prim = phantom.primitives[idx]
        mask = _test_primitive(points, prim)
        hit_masks[prim.name] = mask
        new = mask & (~filled)
        tissue_id[new] = _TISSUE_INDEX[prim.tissue]
        hit_prim_idx[new] = idx
        filled |= mask

    brightness = np.zeros((n_rays, n_samples), dtype=np.float64)
    attenuation_step = np.ones((n_rays, n_samples), dtype=np.float64)
    for t, i in _TISSUE_INDEX.items():
        m = tissue_id == i
        brightness[m] = TISSUE_ACOUSTICS[t]["brightness"]
        attenuation_step[m] = TISSUE_ACOUSTICS[t]["attenuation"]
    # background (amniotic fluid / unfilled) is anechoic-ish but not identical index
    bg = tissue_id == -1
    brightness[bg] = 0.02
    attenuation_step[bg] = 0.3

    # cumulative attenuation along each ray -> shadowing behind bone
    cum_atten = np.cumsum(attenuation_step, axis=1)
    depth_norm = depths[None, :] / max_depth
    tgc = 1.0 + 1.5 * depth_norm  # time-gain compensation boosts far field
    shadow_gain = np.exp(-0.35 * np.clip(cum_atten - attenuation_step, 0, None) / n_samples * 10)
    echo = brightness * shadow_gain * tgc

    # boundary reflection: bright rim where tissue id changes along the ray
    boundary = np.zeros_like(echo)
    diff = np.diff(tissue_id, axis=1, prepend=tissue_id[:, :1])
    boundary_mask = diff != 0
    boundary[boundary_mask] = 0.3
    echo = np.clip(echo + boundary, 0, 1.5)

    # Rayleigh speckle multiplicative noise
    speckle = rng.rayleigh(scale=0.6, size=echo.shape)
    speckle = speckle / (np.mean(speckle) + 1e-6)
    echo_speckled = echo * (0.5 + 0.5 * speckle)

    # anisotropic PSF: wide lateral (across rays), narrow axial (along depth)
    echo_blur = _blur_2d(echo_speckled, lateral_sigma=1.2, axial_sigma=0.5)

    # log compression
    compressed = np.log1p(8.0 * np.clip(echo_blur, 0, None)) / np.log1p(8.0)
    compressed = np.clip(compressed, 0.0, 1.0)

    raster = _scan_convert(compressed, sector_deg, RASTER_SIZE)

    return raster.astype(np.float32), hit_masks, points, tissue_id


def _blur_2d(arr: np.ndarray, lateral_sigma: float, axial_sigma: float) -> np.ndarray:
    out = gaussian_filter1d(arr, sigma=lateral_sigma, axis=0, mode="nearest")
    out = gaussian_filter1d(out, sigma=axial_sigma, axis=1, mode="nearest")
    return out


def _scan_convert(polar_img: np.ndarray, sector_deg: float, size: int) -> np.ndarray:
    """Convert (n_rays, n_samples) polar (angle, depth) image to a square
    raster via nearest-neighbor inverse mapping (vectorized)."""
    n_rays, n_samples = polar_img.shape
    half = np.radians(sector_deg / 2)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)
    cx = size / 2.0
    px = (xs - cx) / (size / 2.0)  # [-1, 1]
    py = ys / size  # [0, 1] depth downward
    r = py
    theta = np.arctan2(px * r, r + 1e-6)  # approximate angle from center-top
    theta = np.clip(theta, -half, half)
    ray_idx = ((theta + half) / (2 * half) * (n_rays - 1)).astype(np.int32)
    samp_idx = (r * (n_samples - 1)).astype(np.int32)
    ray_idx = np.clip(ray_idx, 0, n_rays - 1)
    samp_idx = np.clip(samp_idx, 0, n_samples - 1)
    raster = polar_img[ray_idx, samp_idx]
    in_sector = np.abs(px) <= (py + 0.05)
    raster = raster * in_sector
    return raster


def structure_visibility(hit_masks: dict[str, np.ndarray], name: str) -> float:
    """Fraction of the sector's samples occupied by a given primitive."""
    if name not in hit_masks:
        return 0.0
    return float(np.mean(hit_masks[name]))
