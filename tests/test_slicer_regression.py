"""Slicer determinism regression test.

Phase 3 attempted a "batch inside-tests by primitive kind" rewrite of
`cast_slice` to speed up rendering. It was measured SLOWER than the
original per-primitive loop (broadcasting the large shared temporary array
needed for batched matmul cost more than the 23 small per-primitive BLAS
calls it was meant to replace -- see the perf note in
`environment/slicer.py` and status.md Phase 3) and was reverted. This test
guards the (kept) per-primitive implementation: given the same phantom and
probe pose, `cast_slice` must be a deterministic function of its `rng`
(same rng state in -> bit-identical image out), so a future performance
attempt can be checked against this baseline before being adopted.
"""
import numpy as np

from environment.phantom import build_phantom
from environment.slicer import cast_slice

SEEDS = [0, 1, 2, 3]


def _sample_frame(phantom):
    pos = np.array([0.05, 0.05, 0.12])
    forward = np.array([0.1, 0.05, -1.0])
    forward /= np.linalg.norm(forward)
    right = np.array([1.0, 0.0, 0.1])
    right -= np.dot(right, forward) * forward
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    return pos, forward, right, up


def test_cast_slice_deterministic_given_same_rng_state():
    for seed in SEEDS:
        phantom = build_phantom(np.random.default_rng(seed))
        pos, forward, right, up = _sample_frame(phantom)

        image_a, masks_a, _, tid_a = cast_slice(phantom, pos, forward, right, up, np.random.default_rng(999))
        image_b, masks_b, _, tid_b = cast_slice(phantom, pos, forward, right, up, np.random.default_rng(999))

        assert np.array_equal(image_a, image_b)
        assert np.array_equal(tid_a, tid_b)
        assert set(masks_a.keys()) == set(masks_b.keys())
        for name in masks_a:
            assert np.array_equal(masks_a[name], masks_b[name])


def test_cast_slice_output_is_well_formed():
    for seed in SEEDS:
        phantom = build_phantom(np.random.default_rng(seed))
        pos, forward, right, up = _sample_frame(phantom)
        image, masks, points, tid = cast_slice(phantom, pos, forward, right, up, np.random.default_rng(seed))

        assert image.shape == (128, 128)
        assert image.dtype == np.float32
        assert np.all(image >= 0.0) and np.all(image <= 1.0)
        assert set(masks.keys()) == {p.name for p in phantom.primitives}
