"""Discrimination tests for the AGA/SGA classifier (Bug #4 fix).

These exist specifically to prevent the regression this fix addresses: a
classifier that returns "SGA" (or any single label) for essentially 100%
of episodes regardless of actual growth status, because its EFW-for-GA
"expected" curve and its EFW formula disagreed. See the "CROSS-CONSISTENCY
BUG, FOUND AND FIXED" note in `environment/clinical_constants.py`'s module
docstring for the full history.

None of these tests require the underlying biometry-vs-GA regressions to
be clinically verified -- they only require `classify_growth` to be
*self-consistent* (normal growth -> mostly AGA, IUGR growth -> mostly SGA,
and not a constant function), which holds regardless of whether those
regressions are later corrected.
"""
import numpy as np
import pytest

from environment import clinical_constants as cc
from environment.phantom import sample_biometry

N_SAMPLES = 200
GA_RANGE = (cc.GA_MIN_WEEKS, cc.GA_MAX_WEEKS)


def _sample_ga(rng: np.random.Generator) -> float:
    return float(rng.uniform(*GA_RANGE))


def test_sanity_anchor_30_weeks():
    """Catches a units bug (e.g. a missing/duplicated mm->cm conversion)
    that would put this off by ~10x or by hundreds-to-thousands of grams
    in the wrong direction. NOT a claim that 1000-1700g is a clinically
    precise 30-week EFW range -- it's a coarse guardrail. The band is
    intentionally wider than a tight clinical percentile band because the
    biometry-vs-GA regressions feeding this are still TODO(verify)
    placeholders (see clinical_constants.py); tightening this band is a
    job for whoever verifies those regressions, not this test.
    """
    efw = cc.expected_efw_grams(30.0)
    assert 1000.0 <= efw <= 1700.0, f"expected_efw_grams(30) = {efw:.0f}g outside sanity band"


def test_sanity_anchor_38_weeks():
    """See test_sanity_anchor_30_weeks -- same rationale, term birth weight."""
    efw = cc.expected_efw_grams(38.0)
    assert 2500.0 <= efw <= 3600.0, f"expected_efw_grams(38) = {efw:.0f}g outside sanity band"


def test_unit_conversion_applied_exactly_once():
    """Directly checks the mm->cm contract documented on hadlock_efw_grams:
    converting inputs via /10 exactly once should match calling the formula
    on manually-pre-converted cm values with no further division, and
    should NOT match a double-converted (/100) or unconverted call."""
    bpd, hc, ac, fl = 80.0, 300.0, 280.0, 60.0  # mm
    once = cc.hadlock_efw_grams(bpd, hc, ac, fl)

    def hadlock_from_cm(bpd_cm, hc_cm, ac_cm, fl_cm):
        log10_efw = (
            cc.HADLOCK_INTERCEPT
            + cc.HADLOCK_AC_FL_COEF * ac_cm * fl_cm
            + cc.HADLOCK_HC_COEF * hc_cm
            + cc.HADLOCK_BPD_AC_COEF * bpd_cm * ac_cm
            + cc.HADLOCK_AC_COEF * ac_cm
            + cc.HADLOCK_FL_COEF * fl_cm
        )
        return 10.0 ** log10_efw

    expected_once = hadlock_from_cm(bpd / 10.0, hc / 10.0, ac / 10.0, fl / 10.0)
    assert once == pytest.approx(expected_once, rel=1e-9)

    # A double conversion (/100 total) would give a wildly different (much
    # smaller) EFW; a missing conversion (mm treated as cm) would give a
    # wildly different (much larger) EFW. Both must be clearly distinguishable.
    double_converted = hadlock_from_cm(bpd / 100.0, hc / 100.0, ac / 100.0, fl / 100.0)
    unconverted = hadlock_from_cm(bpd, hc, ac, fl)
    assert not (once == pytest.approx(double_converted, rel=0.5))
    assert not (once == pytest.approx(unconverted, rel=0.5))


def test_median_biometry_matches_phantom_sample_biometry():
    """`_median_biometry_mm` duplicates `phantom.sample_biometry(ga, 1, 1)`
    to avoid a circular import -- this test is the tripwire that catches
    the two drifting apart if either is edited in isolation."""
    for ga in [28.0, 31.5, 34.0, 38.0]:
        a = cc._median_biometry_mm(ga)
        b = sample_biometry(ga, 1.0, 1.0)
        for key in ("BPD", "HC", "AC", "FL"):
            assert a[key] == pytest.approx(b[key], rel=1e-9)


def test_normal_growth_is_mostly_classified_aga():
    """NOTE on the threshold below: isolating the EFW-percentile signal
    alone (dropping the complementary HC/AC-ratio check) gives a ~0.91-0.92
    AGA rate on this same sampling -- i.e. the Bug #4 fix (EFW threshold
    now self-consistent with the Hadlock formula) works as intended. The
    combined rate is lower (~0.70-0.75) because the PRE-EXISTING, separately
    placeholder `HC_AC_RATIO_IUGR_THRESHOLD` (1.10) fires on independent
    sampling noise alone: g_head and g_abdo are sampled independently with
    5% SD each, so roughly 1-in-5 "normal" fetuses have a HC/AC ratio over
    1.10 purely by chance, with no actual asymmetric growth. This is a
    real, distinct finding from Bug #4 -- see the TODO(verify) note on
    `HC_AC_RATIO_IUGR_THRESHOLD` in clinical_constants.py -- and is NOT
    fixed here: that threshold is itself an unverified placeholder, and
    retuning it is out of scope for this pass (same "don't invent/adjust
    unverified constants" rule that applies to the IUGR growth factors).
    The bound below reflects the honestly-measured COMBINED behavior with
    margin, not the illustrative EFW-only figure.
    """
    rng = np.random.default_rng(0)
    labels = []
    for _ in range(N_SAMPLES):
        ga = _sample_ga(rng)
        g_head = rng.normal(cc.NORMAL_GROWTH_MEAN, cc.NORMAL_GROWTH_SD)
        g_abdo = rng.normal(cc.NORMAL_GROWTH_MEAN, cc.NORMAL_GROWTH_SD)
        b = sample_biometry(ga, g_head, g_abdo)
        labels.append(cc.classify_growth(b["BPD"], b["HC"], b["AC"], b["FL"], ga))

    aga_rate = labels.count("AGA") / len(labels)
    print(f"\nnormal-growth AGA rate: {aga_rate:.3f} (n={N_SAMPLES})")
    # Not ~1.0 (ratio-check false positives, see docstring above) but must
    # be emphatically not ~0 (the pre-fix broken state, which was ~0.0).
    assert aga_rate >= 0.60, f"AGA rate {aga_rate:.3f} too low for normal growth -- classifier still miscalibrated"


def test_efw_signal_alone_is_well_calibrated():
    """Isolates the EFW-percentile signal (the actual subject of Bug #4)
    from the pre-existing HC/AC-ratio check, so a future change to the
    ratio threshold can't mask a regression in the EFW fix itself."""
    rng = np.random.default_rng(0)
    n_aga = 0
    for _ in range(N_SAMPLES):
        ga = _sample_ga(rng)
        g_head = rng.normal(cc.NORMAL_GROWTH_MEAN, cc.NORMAL_GROWTH_SD)
        g_abdo = rng.normal(cc.NORMAL_GROWTH_MEAN, cc.NORMAL_GROWTH_SD)
        b = sample_biometry(ga, g_head, g_abdo)
        efw = cc.hadlock_efw_grams(b["BPD"], b["HC"], b["AC"], b["FL"])
        if efw >= cc.sga_threshold_grams(ga):
            n_aga += 1
    rate = n_aga / N_SAMPLES
    print(f"\nEFW-signal-only AGA rate (ratio check excluded): {rate:.3f} (n={N_SAMPLES})")
    assert rate >= 0.85, f"EFW-only AGA rate {rate:.3f} -- the Bug #4 fix itself regressed"


def test_iugr_growth_is_mostly_classified_sga():
    rng = np.random.default_rng(1)
    labels = []
    for _ in range(N_SAMPLES):
        ga = _sample_ga(rng)
        g_head = cc.IUGR_HEAD_GROWTH_FACTOR + rng.normal(0, 0.02)
        g_abdo = cc.IUGR_ABDO_GROWTH_FACTOR + rng.normal(0, 0.02)
        b = sample_biometry(ga, g_head, g_abdo)
        labels.append(cc.classify_growth(b["BPD"], b["HC"], b["AC"], b["FL"], ga))

    sga_rate = labels.count("SGA") / len(labels)
    print(f"\nIUGR-growth SGA rate: {sga_rate:.3f} (n={N_SAMPLES})")
    assert sga_rate >= 0.70, f"SGA detection rate {sga_rate:.3f} too low for asymmetric IUGR growth"


def test_classifier_is_not_a_constant_function():
    """The specific regression this fix targets: classify_growth returning
    the same label regardless of input. Mixes normal and IUGR populations
    and asserts both labels actually appear."""
    rng = np.random.default_rng(2)
    labels = set()
    for i in range(N_SAMPLES):
        ga = _sample_ga(rng)
        if i % 2 == 0:
            g_head, g_abdo = rng.normal(1.0, 0.05), rng.normal(1.0, 0.05)
        else:
            g_head = cc.IUGR_HEAD_GROWTH_FACTOR + rng.normal(0, 0.02)
            g_abdo = cc.IUGR_ABDO_GROWTH_FACTOR + rng.normal(0, 0.02)
        b = sample_biometry(ga, g_head, g_abdo)
        labels.add(cc.classify_growth(b["BPD"], b["HC"], b["AC"], b["FL"], ga))

    assert labels == {"AGA", "SGA"}, f"classifier only ever returned {labels} -- not discriminating"
