"""Held-out distribution evaluation: transverse lie + severe IUGR.

Compares a trained model's plane-acquisition success, measurement error,
and classification accuracy on the standard training distribution vs a
held-out distribution the model never saw during training (transverse fetal
lie, which `build_phantom()` only samples when `allow_transverse=True`, plus
forced asymmetric-IUGR growth). This is what `train`-vs-`generalization`
comparisons in `evaluation/plots.py` read from.

Usage:
    uv run python -m evaluation.generalization --algo ppo --model-path models/ppo/best/model.zip
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from environment.custom_env import UltrasoundProbeEnv
from evaluation.evaluate import load_model, run_episode


def evaluate_distribution(algo: str, model_path: str, n_episodes: int,
                           allow_transverse: bool, force_iugr: bool | None, seed: int):
    # single_target=False pinned EXPLICITLY: this function's whole purpose
    # (classification_rate, biometry error across all 3 targets) requires
    # the full head->abdomen->femur->classification task. UltrasoundProbeEnv's
    # own default flipped to single_target=True (status.md "make
    # single_target the environment default" pass) -- this call must not
    # silently inherit that, or classification_rate/mean_biometry_error_pct
    # would go permanently empty/None.
    env = UltrasoundProbeEnv(seed=seed, single_target=False,
                              allow_transverse=allow_transverse, force_iugr=force_iugr)
    model = load_model(algo, model_path, env)

    rewards, successes, flags, biometry_err = [], [], [], []
    for i in range(n_episodes):
        env.reset(seed=seed + i)
        r, l, info = run_episode(model, env)
        rewards.append(r)
        successes.append(len(info.get("acquired", [])) == len(env.targets))
        flags.append(info.get("flag"))
        if len(env.acquired) == 3:
            true = env.phantom.biometry_mm
            errs = []
            for key, target in [("BPD", "head"), ("HC", "head"), ("AC", "abdomen"), ("FL", "femur")]:
                if key in env.acquired.get(target, {}):
                    errs.append(abs(env.acquired[target][key] - true[key]) / true[key])
            if errs:
                biometry_err.append(float(np.mean(errs)))

    return dict(
        n_episodes=n_episodes,
        mean_reward=float(np.mean(rewards)),
        success_rate=float(np.mean(successes)),
        classification_rate=float(np.mean([f is not None for f in flags])),
        mean_biometry_error_pct=float(np.mean(biometry_err) * 100) if biometry_err else None,
    )


def run(algo: str, model_path: str, n_episodes: int = 50, seed: int = 5000):
    in_dist = evaluate_distribution(algo, model_path, n_episodes,
                                     allow_transverse=False, force_iugr=None, seed=seed)
    held_out = evaluate_distribution(algo, model_path, n_episodes,
                                      allow_transverse=True, force_iugr=True, seed=seed + 10_000)
    return dict(in_distribution=in_dist, held_out_transverse_severe_iugr=held_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--n-episodes", type=int, default=50)
    args = parser.parse_args()
    results = run(args.algo, args.model_path, args.n_episodes)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
