"""The day-1 feature-separability gate, as an assertable test.

If this fails (accuracy well below ~0.90 for any target), the phantom lacks
enough structural contrast, or the feature set can't distinguish on-target
from off-target slices -- fix `environment/phantom.py` or
`environment/features.py` before trusting any RL result built on top.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.separability_check import generate_dataset, TARGETS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

MIN_ACCURACY = 0.85  # slightly relaxed vs the 0.90 target to tolerate CV noise at small N


def test_separability_gate():
    X, y, groups, _ = generate_dataset(seed=42)
    for target in TARGETS:
        mask = groups == target
        clf = LogisticRegression(max_iter=2000)
        scores = cross_val_score(clf, X[mask], y[mask], cv=5)
        acc = scores.mean()
        assert acc >= MIN_ACCURACY, f"{target} separability accuracy {acc:.3f} below {MIN_ACCURACY}"
