"""Tests for the GA -> EFW-percentile lookup interface (Phase 4 refactor).

These test the STRUCTURE of the lookup (monotonicity, interpolation,
boundary clamping) -- i.e. that a verified percentile chart could be
dropped into `_EFW_PERCENTILE_TABLE` and be queried correctly. They
deliberately do NOT assert anything about whether the current PLACEHOLDER
numbers are clinically correct (they are not -- see the cross-consistency
warning in `environment/clinical_constants.py`'s module docstring, which
documents that normal-growth biometry currently classifies as SGA at every
gestational age in the training range).
"""
import numpy as np

from environment import clinical_constants as cc


def test_percentile_is_monotonic_in_efw():
    percentiles = [cc.efw_percentile_for_ga(efw, 34.0) for efw in [500, 1000, 2000, 3000, 4000, 6000]]
    assert all(a <= b for a, b in zip(percentiles, percentiles[1:]))


def test_percentile_interpolates_between_ga_nodes():
    nodes = sorted(cc._EFW_PERCENTILE_TABLE.keys())
    ga_lo, ga_hi = nodes[0], nodes[1]
    ga_mid = (ga_lo + ga_hi) / 2
    efw = 2500.0
    p_lo = cc.efw_percentile_for_ga(efw, ga_lo)
    p_mid = cc.efw_percentile_for_ga(efw, ga_mid)
    p_hi = cc.efw_percentile_for_ga(efw, ga_hi)
    assert min(p_lo, p_hi) - 1e-6 <= p_mid <= max(p_lo, p_hi) + 1e-6


def test_percentile_clamps_outside_ga_node_range():
    nodes = sorted(cc._EFW_PERCENTILE_TABLE.keys())
    below = cc.efw_percentile_for_ga(2500.0, nodes[0] - 5.0)
    at_min = cc.efw_percentile_for_ga(2500.0, nodes[0])
    above = cc.efw_percentile_for_ga(2500.0, nodes[-1] + 5.0)
    at_max = cc.efw_percentile_for_ga(2500.0, nodes[-1])
    assert np.isclose(below, at_min)
    assert np.isclose(above, at_max)


def test_classify_growth_returns_valid_flag():
    from environment.phantom import sample_biometry
    for ga in [28.0, 34.0, 38.0]:
        for g_head, g_abdo in [(1.0, 1.0), (0.95, 0.82)]:
            b = sample_biometry(ga, g_head, g_abdo)
            flag = cc.classify_growth(b["BPD"], b["HC"], b["AC"], b["FL"], ga)
            assert flag in ("AGA", "SGA")


def test_asymmetric_iugr_ratio_flags_sga():
    """The HC/AC-ratio arm of classify_growth (independent of the EFW
    percentile table's calibration) should still flag a clearly asymmetric
    (head-sparing) case as SGA -- this is the one part of the classifier
    not affected by the percentile-table cross-consistency issue."""
    assert cc.classify_growth(bpd_mm=85, hc_mm=310, ac_mm=250, fl_mm=60, ga_weeks=34) == "SGA"
