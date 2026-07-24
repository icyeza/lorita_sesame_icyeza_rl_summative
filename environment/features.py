"""Image -> feature-vector extraction.

Single source of truth for turning a rendered ultrasound raster into the
scalar features used in the RL observation. Imported by both
`custom_env.py` (to build observations) and `scripts/separability_check.py`
/ `tests/test_separability.py` (to validate that these features actually
separate on-target from off-target slices) -- so the exact same function is
what's trained on and what's validated.

No ground-truth phantom state is used here: everything is derived purely
from the rendered image, as it would be from a real ultrasound frame.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

N_IMAGE_FEATURES = 19

BRIGHT_THRESHOLD = 0.55
ANECHOIC_THRESHOLD = 0.12
EDGE_THRESHOLD = 0.15


def _largest_component_stats(mask: np.ndarray):
    """Return (presence, area_frac, eccentricity, centroid_dx, centroid_dy,
    orientation_rad) for the largest connected component of `mask`."""
    h, w = mask.shape
    if not mask.any():
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    labeled, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, n + 1))
    biggest = int(np.argmax(sizes)) + 1
    ys, xs = np.nonzero(labeled == biggest)
    area_frac = float(len(xs)) / (h * w)
    cy, cx = ys.mean(), xs.mean()
    dy = (cy - h / 2.0) / (h / 2.0)
    dx = (cx - w / 2.0) / (w / 2.0)

    if len(xs) >= 3:
        cov = np.cov(np.stack([xs, ys]).astype(np.float64))
        evals, evecs = np.linalg.eigh(cov)
        evals = np.clip(evals, 1e-9, None)
        eccentricity = float(np.sqrt(1 - evals.min() / evals.max()))
        major = evecs[:, np.argmax(evals)]
        orientation = float(np.arctan2(major[1], major[0]))
    else:
        eccentricity, orientation = 0.0, 0.0

    return 1.0, area_frac, eccentricity, dx, dy, orientation


def _midline_score(img: np.ndarray) -> float:
    """How strongly a bright vertical line runs through the image center."""
    h, w = img.shape
    band = img[:, w // 2 - 2: w // 2 + 3]
    return float(np.clip(band.mean() * 2.0, 0.0, 1.0))


def _symmetry_score(img: np.ndarray) -> float:
    flipped = img[:, ::-1]
    a, b = img.flatten(), flipped.flatten()
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0
    return float(np.clip(np.corrcoef(a, b)[0, 1], -1.0, 1.0))


def _depth_bands(img: np.ndarray, n_bands: int = 3):
    h = img.shape[0]
    edges = np.linspace(0, h, n_bands + 1).astype(int)
    return [img[edges[i]:edges[i + 1], :] for i in range(n_bands)]


def extract_image_features(image: np.ndarray) -> np.ndarray:
    """image: (H, W) float32 in [0, 1]. Returns (N_IMAGE_FEATURES,) float32."""
    img = np.asarray(image, dtype=np.float64)
    h, w = img.shape

    bright_mask = img >= BRIGHT_THRESHOLD
    presence, area, ecc, cdx, cdy, orient = _largest_component_stats(bright_mask)

    midline = _midline_score(img)

    anechoic_mask = (img <= ANECHOIC_THRESHOLD) & (img >= 0)
    ane_labeled, ane_n = ndimage.label(anechoic_mask) if anechoic_mask.any() else (None, 0)
    blob_count = float(ane_n)
    _, ane_area, _, ane_cdx, ane_cdy, _ = _largest_component_stats(anechoic_mask)

    illuminated = img > 0.02
    has_lit = illuminated.any(axis=0)
    top_idx = np.argmax(illuminated, axis=0)  # first True per column (0 if none)
    row_idx = np.arange(h)[:, None]
    below_mask = row_idx >= top_idx[None, :]
    dark_below = (img < 0.05) & below_mask
    below_counts = below_mask.sum(axis=0)
    dark_frac = np.divide(dark_below.sum(axis=0), below_counts, out=np.zeros(w), where=below_counts > 5)
    shadow_cols = np.sum(has_lit & (dark_frac > 0.6) & (below_counts > 5))
    shadow_fraction = shadow_cols / max(1, w)

    gx = np.diff(img, axis=1, prepend=img[:, :1])
    gy = np.diff(img, axis=0, prepend=img[:1, :])
    edge_energy = float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))

    symmetry = _symmetry_score(img)

    bands = _depth_bands(img)
    band_means = [float(b.mean()) for b in bands]

    brightest_idx = int(np.argmax(img)) if img.size else 0
    brightest_row = brightest_idx // w
    depth_of_brightest = brightest_row / h

    feats = np.array([
        presence, area, ecc, cdx, cdy, orient,
        midline,
        blob_count / 10.0, ane_area, ane_cdx, ane_cdy,
        shadow_fraction,
        edge_energy,
        symmetry,
        band_means[0], band_means[1], band_means[2],
        float(img.std()),
        depth_of_brightest,
    ], dtype=np.float32)
    assert feats.shape[0] == N_IMAGE_FEATURES
    return feats


IMAGE_FEATURE_NAMES = [
    "bright_presence", "bright_area", "bright_eccentricity",
    "bright_centroid_dx", "bright_centroid_dy", "bright_orientation",
    "midline_score",
    "anechoic_blob_count", "anechoic_largest_area",
    "anechoic_centroid_dx", "anechoic_centroid_dy",
    "shadow_column_fraction", "edge_energy", "symmetry_score",
    "band0_mean", "band1_mean", "band2_mean", "intensity_std",
    "depth_of_brightest",
]
