"""
Single source of truth for every clinical / biometric constant used in this project.

*** WARNING ***
Every value below was recalled from memory by the project author while building
this simulation. NONE of them have been checked against a primary source
(textbook, peer-reviewed regression, or clinical guideline). They are good
enough to build a *plausible, internally-consistent* training environment,
but they MUST NOT be treated as clinically valid until verified.

Each constant/group carries a `TODO(verify)` comment naming what needs checking
and against what kind of source. Do not copy these numbers into a report or
downstream tool without doing that verification first.

*** CROSS-CONSISTENCY BUG, FOUND AND FIXED ***
A previous validation pass found that `expected_efw_grams()` was an
independently-recalled linear curve that never agreed with
`hadlock_efw_grams()`: normal-growth (no IUGR, growth factor 1.0) biometry
run through Hadlock came out ~800-1000g below the old "expected" curve at
every GA from 28-38 weeks, so `classify_growth()` returned "SGA" for
essentially 100% of episodes, including fully normal ones. The fix was not
to reconcile two independently-guessed curves -- it was to delete the
second one. `expected_efw_grams(GA)` is now DERIVED from the same
`hadlock_efw_grams()` function and the same biometry-vs-GA regressions
every episode already uses (median biometry, growth_factor=1.0, no noise),
so normal biometry lands at the median by construction and the classifier
is self-consistent and discriminating regardless of whether the
biometry-vs-GA regressions themselves are later found to be clinically
accurate. See `expected_efw_grams` and `sga_threshold_grams` below.

This means the REMAINING unverified items in this file (the biometry-vs-GA
regressions, the IUGR growth factors, prevalence, presentation
probabilities) now affect only the CLINICAL REALISM of the simulated
population -- not whether the reward signal functions and discriminates.
Verify the biometry-vs-GA regressions against INTERGROWTH-21st / WHO fetal
growth charts when doing that pass; they feed both per-episode biometry
and (now) the "expected" curve, so a correction there improves both
consistently without touching `classify_growth`.
"""

# ---------------------------------------------------------------------------
# Gestational-age -> base fetal biometry regressions (mm), linear-in-GA(weeks)
# TODO(verify): These linear fits approximate Hadlock/INTERGROWTH-21st growth
# charts by eye/memory. Verify against a published GA->biometry regression
# table (e.g. Hadlock 1984, INTERGROWTH-21st 2014) and replace coefficients.
# ---------------------------------------------------------------------------
BPD_SLOPE_MM_PER_WEEK = 2.4
BPD_INTERCEPT_MM = 5.0

HC_SLOPE_MM_PER_WEEK = 8.9
HC_INTERCEPT_MM = 10.0

AC_SLOPE_MM_PER_WEEK = 9.9
AC_INTERCEPT_MM = -37.0

FL_SLOPE_MM_PER_WEEK = 1.9
FL_INTERCEPT_MM = 1.2

# Gestational age sampling range for training (weeks)
GA_MIN_WEEKS = 28.0
GA_MAX_WEEKS = 38.0

# ---------------------------------------------------------------------------
# Growth-factor model (multiplicative noise / asymmetric IUGR)
# TODO(verify): The normal-growth noise SD (5%) and the specific head/abdomen
# growth-factor values for asymmetric (head-sparing) IUGR are illustrative
# placeholders, not derived from a growth-restriction cohort study. Verify
# against IUGR/FGR literature (e.g. ACOG practice bulletins, Figueras &
# Gardosi 2011) before using this to justify any clinical claim.
# ---------------------------------------------------------------------------
NORMAL_GROWTH_MEAN = 1.0
NORMAL_GROWTH_SD = 0.05

IUGR_HEAD_GROWTH_FACTOR = 0.95
IUGR_ABDO_GROWTH_FACTOR = 0.82

# Probability an episode samples the asymmetric-IUGR growth pattern during
# training (kept separate from the held-out "severe IUGR" generalization band
# used in evaluation/generalization.py).
# TODO(verify): Prevalence placeholder, not from an epidemiological source.
IUGR_TRAINING_PREVALENCE = 0.15

# HC/AC ratio threshold used (together with EFW percentile) to flag
# asymmetric IUGR. TODO(verify): placeholder threshold; real classification
# uses population percentile charts for HC/AC by GA, not a fixed ratio.
#
# KNOWN CALIBRATION ISSUE (found while fixing Bug #4, not itself fixed --
# same "don't retune unverified constants" rule applies here): because
# g_head/g_abdo are sampled INDEPENDENTLY (5% SD each, see
# NORMAL_GROWTH_SD) with no correlation, this fixed 1.10 threshold fires on
# independent sampling noise alone for roughly 1-in-5 fully normal (non-
# IUGR) fetuses, contributing a real false-positive rate to
# `classify_growth` on top of the (now correctly-calibrated) EFW-percentile
# signal. See tests/test_classification_discrimination.py, which isolates
# and quantifies this. A verified fix would likely make this threshold
# GA-dependent and/or derived from a real HC/AC-by-GA percentile chart
# rather than a single fixed ratio -- out of scope here.
HC_AC_RATIO_IUGR_THRESHOLD = 1.10

# ---------------------------------------------------------------------------
# Fetal pose sampling
# ---------------------------------------------------------------------------
CEPHALIC_PRESENTATION_PROB = 0.85
BREECH_PRESENTATION_PROB = 0.15
SPINE_ANTERIOR_PROB = 0.5  # remainder is spine-posterior (harder to acquire)

# ---------------------------------------------------------------------------
# Hadlock estimated fetal weight (EFW) formula.
# VERIFIED (Hadlock et al. 1985, 4-parameter BPD/HC/AC/FL log10 regression;
# cross-checked against the standard reference implementation reproduced by
# perinatology.com's EFW calculator). Coefficients and units below are
# correct as-is -- do not re-derive these.
#   log10(EFW_g) = 1.3596 - 0.00386*AC*FL + 0.0064*HC
#                  + 0.00061*BPD*AC + 0.0424*AC + 0.174*FL   (AC, FL, HC, BPD in cm)
# ---------------------------------------------------------------------------
HADLOCK_INTERCEPT = 1.3596
HADLOCK_AC_FL_COEF = -0.00386
HADLOCK_HC_COEF = 0.0064
HADLOCK_BPD_AC_COEF = 0.00061
HADLOCK_AC_COEF = 0.0424
HADLOCK_FL_COEF = 0.174


def hadlock_efw_grams(bpd_mm: float, hc_mm: float, ac_mm: float, fl_mm: float) -> float:
    """Estimated fetal weight (grams) via the (verified) Hadlock formula above.

    UNIT CONTRACT: all four biometry inputs are in MILLIMETRES (matching
    every other biometry value in this codebase -- `phantom.sample_biometry`,
    the acquired-measurement dict in `custom_env.py`, etc.). The Hadlock
    formula itself operates on CENTIMETRES, so each input is divided by 10
    exactly once, immediately below, and nowhere else. If you see a second
    /10 or a missing one anywhere EFW is computed, that's the bug -- it's
    exactly the kind of unit slip that caused the cross-consistency bug
    documented in the module docstring (a factor-of-10-ish error would have
    been obvious; the actual bug was subtler, but this contract is still
    worth stating explicitly since it's the most common way to reintroduce
    a similar class of bug).
    """
    bpd_cm, hc_cm, ac_cm, fl_cm = bpd_mm / 10.0, hc_mm / 10.0, ac_mm / 10.0, fl_mm / 10.0
    log10_efw = (
        HADLOCK_INTERCEPT
        + HADLOCK_AC_FL_COEF * ac_cm * fl_cm
        + HADLOCK_HC_COEF * hc_cm
        + HADLOCK_BPD_AC_COEF * bpd_cm * ac_cm
        + HADLOCK_AC_COEF * ac_cm
        + HADLOCK_FL_COEF * fl_cm
    )
    return 10.0 ** log10_efw


# ---------------------------------------------------------------------------
# EFW-percentile / SGA cutoff.
#
# TODO(verify): The 10th-percentile-for-GA "SGA" cutoff is standard in
# principle (ACOG/ISUOG use the 10th percentile). What IS now internally
# consistent (see module docstring's "CROSS-CONSISTENCY BUG, FOUND AND
# FIXED") is that the "expected"/median EFW-for-GA curve below is DERIVED
# from `hadlock_efw_grams()` + the same biometry-vs-GA regressions used
# everywhere else, not a separately-guessed curve. What's still a
# placeholder is the biometry-vs-GA regressions themselves (see their
# TODO(verify) above) -- verifying those improves this curve automatically,
# with no separate curve to keep in sync.
# ---------------------------------------------------------------------------
SGA_EFW_PERCENTILE_CUTOFF = 10.0

# Hadlock's published fetal-weight percentile tables show an approximately
# constant coefficient of variation (~13%) across the third trimester.
# Used both to derive the percentile table's SD-for-GA below (SD = CV *
# median) and to derive `SGA_MEDIAN_FRACTION`. TODO(verify): the *value*
# 0.13 is a commonly-cited approximation, not pulled from a specific table
# for this project's exact GA range -- refine against a primary source
# alongside the biometry-vs-GA regressions.
HADLOCK_EFW_CV = 0.13

# 10th-percentile-as-fraction-of-median: 1 + z_10 * CV, where z_10=-1.282 is
# the exact standard-normal 10th-percentile z-score (not itself a
# placeholder -- it's how you convert a CV to a percentile fraction under a
# normal assumption). 1 - 1.282*0.13 ~= 0.83.
SGA_MEDIAN_FRACTION = 1.0 + (-1.282) * HADLOCK_EFW_CV

# GA nodes (weeks) at which the percentile table below is defined --
# matches the GA_MIN_WEEKS..GA_MAX_WEEKS training range. A verified
# replacement table can use different/finer nodes; `efw_percentile_for_ga`
# linearly interpolates between whatever nodes are present.
_EFW_PERCENTILE_GA_NODES = [28.0, 30.0, 32.0, 34.0, 36.0, 38.0]


def _median_biometry_mm(ga_weeks: float) -> dict:
    """Median (growth_factor=1.0), noise-free biometry (mm) at `ga_weeks` --
    the exact same formula as `phantom.sample_biometry(ga_weeks, 1.0, 1.0)`,
    duplicated here (rather than imported) because `phantom.py` already
    imports this module and a reverse import would be circular.
    `tests/test_classification_discrimination.py` cross-checks this
    function against `phantom.sample_biometry` directly so the two can
    never silently drift apart again."""
    return dict(
        BPD=BPD_SLOPE_MM_PER_WEEK * ga_weeks + BPD_INTERCEPT_MM,
        HC=HC_SLOPE_MM_PER_WEEK * ga_weeks + HC_INTERCEPT_MM,
        AC=AC_SLOPE_MM_PER_WEEK * ga_weeks + AC_INTERCEPT_MM,
        FL=FL_SLOPE_MM_PER_WEEK * ga_weeks + FL_INTERCEPT_MM,
    )


def expected_efw_grams(ga_weeks: float) -> float:
    """Median EFW-for-GA, DERIVED from the same (verified) Hadlock formula
    and the same biometry-vs-GA regressions every episode's actual EFW
    uses -- not an independently-authored curve. This is the fix for the
    cross-consistency bug described in the module docstring: normal-growth
    biometry now lands at the median (efw == expected_efw_grams(ga)) by
    construction, for any biometry-vs-GA regression coefficients, verified
    or not.
    """
    b = _median_biometry_mm(ga_weeks)
    return hadlock_efw_grams(b["BPD"], b["HC"], b["AC"], b["FL"])


def sga_threshold_grams(ga_weeks: float) -> float:
    """EFW threshold below which `classify_growth` flags SGA: ~83% of the
    Hadlock-derived median EFW at this GA (see `SGA_MEDIAN_FRACTION`)."""
    return expected_efw_grams(ga_weeks) * SGA_MEDIAN_FRACTION


def _percentile_row(ga_weeks: float) -> dict:
    """Builds one row of the percentile table from the (now Hadlock-
    derived) expected-EFW curve + a CV-proportional SD, both defined above.
    TODO(verify): a verified reference chart would hardcode these
    percentile values per GA node directly from a published table instead
    of deriving them from a normal approximation -- but even placeholder,
    this is now self-consistent with `hadlock_efw_grams`.
    """
    mean = expected_efw_grams(ga_weeks)
    sd = mean * HADLOCK_EFW_CV
    # z-scores for a small set of standard percentiles (normal approximation)
    z_by_percentile = {3: -1.881, 10: -1.282, 50: 0.0, 90: 1.282, 97: 1.881}
    return {p: mean + z * sd for p, z in z_by_percentile.items()}


# TODO(verify): REPLACE THIS TABLE with a real EFW-for-GA percentile chart.
# Structure to preserve: {ga_weeks: {percentile: efw_grams, ...}, ...}.
_EFW_PERCENTILE_TABLE: dict[float, dict[int, float]] = {
    ga: _percentile_row(ga) for ga in _EFW_PERCENTILE_GA_NODES
}


def _interp_percentile_row(ga_weeks: float) -> dict:
    """Linearly interpolate the percentile table's rows to an arbitrary GA.
    This interpolation logic itself is NOT placeholder -- it's how any
    percentile table (placeholder or verified) should be queried at a GA
    that falls between the table's nodes."""
    nodes = sorted(_EFW_PERCENTILE_TABLE.keys())
    ga_clamped = min(max(ga_weeks, nodes[0]), nodes[-1])
    if ga_clamped in _EFW_PERCENTILE_TABLE:
        return _EFW_PERCENTILE_TABLE[ga_clamped]
    lo = max(g for g in nodes if g <= ga_clamped)
    hi = min(g for g in nodes if g >= ga_clamped)
    if lo == hi:
        return _EFW_PERCENTILE_TABLE[lo]
    t = (ga_clamped - lo) / (hi - lo)
    row_lo, row_hi = _EFW_PERCENTILE_TABLE[lo], _EFW_PERCENTILE_TABLE[hi]
    return {p: (1 - t) * row_lo[p] + t * row_hi[p] for p in row_lo}


def efw_percentile_for_ga(efw_grams: float, ga_weeks: float) -> float:
    """Approximate percentile rank of `efw_grams` at `ga_weeks`, via linear
    interpolation both across the table's GA nodes and across its
    percentile columns. This is the GA -> percentile-curve LOOKUP
    interface `classify_growth` uses -- swap `_EFW_PERCENTILE_TABLE` for a
    verified chart and this function (and `classify_growth`) need no
    changes."""
    row = _interp_percentile_row(ga_weeks)
    percentiles = sorted(row.keys())
    values = [row[p] for p in percentiles]
    if efw_grams <= values[0]:
        return float(percentiles[0])
    if efw_grams >= values[-1]:
        return float(percentiles[-1])
    for i in range(len(values) - 1):
        if values[i] <= efw_grams <= values[i + 1]:
            t = (efw_grams - values[i]) / (values[i + 1] - values[i])
            return percentiles[i] + t * (percentiles[i + 1] - percentiles[i])
    return 50.0  # unreachable given the bounds checks above


def classify_growth(bpd_mm, hc_mm, ac_mm, fl_mm, ga_weeks) -> str:
    """Return 'AGA' or 'SGA'.

    Primary signal: EFW < ~83% of the Hadlock-derived median EFW for this
    GA (`sga_threshold_grams`) -- self-consistent by construction (see
    module docstring), so this is discriminating regardless of whether the
    underlying biometry-vs-GA regressions are later found to be clinically
    accurate. Complementary signal: an asymmetric (head-sparing) HC/AC
    ratio above `HC_AC_RATIO_IUGR_THRESHOLD`, which can flag IUGR even when
    overall weight isn't yet below the percentile cutoff.

    TODO(verify): `HC_AC_RATIO_IUGR_THRESHOLD` and the biometry-vs-GA
    regressions feeding both signals are still placeholders -- see their
    respective TODO(verify) notes above. `SGA_MEDIAN_FRACTION` follows
    directly from `HADLOCK_EFW_CV`, which is a commonly-cited approximation
    rather than a value pulled from a specific published table.
    """
    efw = hadlock_efw_grams(bpd_mm, hc_mm, ac_mm, fl_mm)
    threshold = sga_threshold_grams(ga_weeks)
    hc_ac_ratio = hc_mm / ac_mm if ac_mm > 0 else 0.0

    below_threshold = efw < threshold
    asymmetric_ratio = hc_ac_ratio > HC_AC_RATIO_IUGR_THRESHOLD

    if below_threshold or asymmetric_ratio:
        return "SGA"
    return "AGA"
