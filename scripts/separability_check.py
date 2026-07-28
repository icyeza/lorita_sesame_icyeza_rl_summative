"""Day-1 separability gate.

Before any RL code is trusted: render slices at each target plane's ideal
pose and at random off-target probe poses, extract features via
`environment.features`, fit a logistic regression, and report how well the
features separate "on target" from "off target" for each plane type.

If this fails (accuracy well below ~0.90), the phantom lacks structural
contrast and/or the feature set can't distinguish the goal -- the phantom
must be fixed before any RL training is trusted.

Usage: uv run python scripts/separability_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from environment.phantom import build_phantom
from environment.slicer import cast_slice
from environment.features import extract_image_features

OUT_DIR = Path(__file__).resolve().parent.parent / "logs" / "separability"
N_PER_CLASS = 30
TARGETS = ["head", "abdomen", "femur"]


def _probe_frame_for_plane(phantom, target: str, rng: np.random.Generator, on_target: bool, jitter_scale: float = 1.0):
    pt = phantom.plane_targets[target]
    normal = pt.normal / np.linalg.norm(pt.normal)

    tangent = np.array([1.0, 0.0, 0.0]) - np.dot([1.0, 0.0, 0.0], normal) * normal
    if np.linalg.norm(tangent) < 1e-6:
        tangent = np.array([0.0, 1.0, 0.0])
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)

    if on_target:
        pos = pt.point - normal * rng.uniform(0.03, 0.05)
        forward = normal
    else:
        angle_jitter = rng.uniform(np.radians(35), np.radians(80)) * jitter_scale
        axis = rng.choice([tangent, bitangent])
        c, s = np.cos(angle_jitter), np.sin(angle_jitter)
        forward = normal * c + np.cross(axis, normal) * s
        forward /= np.linalg.norm(forward)
        offset = tangent * rng.uniform(-0.09, 0.09) + bitangent * rng.uniform(-0.09, 0.09)
        pos = pt.point - forward * rng.uniform(0.03, 0.07) + offset

    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, forward)
    if np.linalg.norm(right) < 1e-6:
        right = tangent
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    return pos, forward, right, up


def generate_dataset(seed: int = 0):
    rng = np.random.default_rng(seed)
    X, y, groups, example_images = [], [], [], {}
    for target in TARGETS:
        for label, on_target in [(1, True), (0, False)]:
            for i in range(N_PER_CLASS):
                phantom = build_phantom(rng)
                pos, forward, right, up = _probe_frame_for_plane(phantom, target, rng, on_target)
                image, hit_masks, points, tid = cast_slice(phantom, pos, forward, right, up, rng)
                feats = extract_image_features(image)
                X.append(feats)
                y.append(label)
                groups.append(target)
                if i == 0:
                    example_images[(target, on_target)] = image
    return np.array(X), np.array(y), np.array(groups), example_images


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y, groups, examples = generate_dataset()

    report = {}
    fig, axes = plt.subplots(len(TARGETS), 2, figsize=(5, 2.5 * len(TARGETS)))
    for row, target in enumerate(TARGETS):
        mask = groups == target
        Xt, yt = X[mask], y[mask]
        clf = LogisticRegression(max_iter=2000)
        scores = cross_val_score(clf, Xt, yt, cv=5)
        acc = float(scores.mean())
        report[target] = acc
        clf.fit(Xt, yt)

        axes[row, 0].imshow(examples[(target, True)], cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"{target}: on-target")
        axes[row, 0].axis("off")
        axes[row, 1].imshow(examples[(target, False)], cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title(f"{target}: off-target")
        axes[row, 1].axis("off")

    fig.suptitle("Separability gate: example slices per target")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "example_slices.png", dpi=120)
    plt.close(fig)

    overall_clf = LogisticRegression(max_iter=2000)
    overall_scores = cross_val_score(overall_clf, X, y, cv=5)
    report["overall"] = float(overall_scores.mean())

    with open(OUT_DIR / "report.txt", "w") as f:
        f.write("Separability gate report (5-fold CV logistic-regression accuracy)\n")
        for k, v in report.items():
            f.write(f"{k}: {v:.4f}\n")

    print("Separability gate results (5-fold CV accuracy):")
    for k, v in report.items():
        print(f"  {k:10s}: {v:.4f}")
    print(f"\nMontage + report saved to {OUT_DIR}")
    return report


if __name__ == "__main__":
    run()
