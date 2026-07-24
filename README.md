# Ultrasound Probe-Guidance RL Summative

A custom Gymnasium environment simulating obstetric ultrasound plane
acquisition, plus four RL algorithms (DQN, REINFORCE, A2C, PPO) trained and
compared on it.

## ⚠️ Clinical constants are unverified

**Every gestational-age biometry regression, the Hadlock EFW formula, and
the IUGR growth-factor values used in this project were recalled from
memory by the author and have NOT been checked against a primary source.**
They live in one place, `environment/clinical_constants.py`, each tagged
with a `# TODO(verify): ...` comment naming what needs checking. Do not use
any number from this project for a clinical claim, report figure, or
downstream tool without verifying it first. See that file's docstring for
the full list.

## What this is

The agent controls a virtual ultrasound probe constrained to a maternal
abdominal surface (5 degrees of freedom: two surface angles + roll/pitch/yaw
offset from the surface normal). It must locate three standard fetal
biometry planes in sequence -- head (BPD/HC), abdomen (AC), femur (FL) --
freeze on each within tolerance, then a simulated Hadlock EFW + HC/AC-ratio
classifier emits **AGA** or **SGA -- IUGR suspected, refer**.

The task is partially observable by design: the observation never contains
the target plane's ground-truth pose, only image-derived features (from a
simulated B-mode slice) plus proprioception (probe angles) and task context
(which target, how many acquired, gestational age). The agent has to infer
where it is from what the (simulated) ultrasound image looks like.

All four algorithms share **one environment and one `Discrete(12)` action
space** so the comparison between them is meaningful (DQN requires discrete
actions).

Full design brief, reward shaping (potential-based, Ng et al. 1999),
phantom anatomy, and repo layout are as specified in the project brief this
was built from.

### Curriculum: single-target vs. the full task

`single_target` and the full 3-target sequential task are two **curriculum
stages of the same environment** -- same `custom_env.py`, same
`Discrete(12)` action space, same observation contract -- not two separate
environments. The `--curriculum` flag on `training/sweep.py` switches
between them:

- `single_target` (one random target per episode, no clinical output) is
  what the hyperparameter grid tables use, since getting comparable
  cross-algorithm learning signal is far more tractable on it.
- `full_task` (head -> abdomen -> femur, then AGA/SGA classification) is
  the actual graded task. It's reserved for a handful of *headline* runs
  using each algorithm's best grid config, plus the `main.py` demo.

### Known limitation: the femur plane isn't fully disambiguated

The reward's femur target plane is constructed to contain the femur's long
axis (a legitimate "long-axis view"), but there's a full 180-degree family
of planes containing that axis, and the reward doesn't distinguish which
one within that family is "more correct." This is a modeling
simplification, not a bug -- see `phantom.py::_build_plane_targets` -- and
means the femur sub-task may be easier (or differently shaped) than the
head/abdomen sub-tasks, which have a uniquely-defined target plane.

## Repository layout

```
environment/   Gymnasium env, phantom geometry, ray-cast slicer, features, clinical constants, rendering
training/      DQN / REINFORCE (custom) / A2C / PPO training + sweep runner + configs
evaluation/    evaluation harness, generalization (held-out transverse lie / severe IUGR), plots
scripts/       day-1 separability gate, optional live Three.js viz server
tests/         env API, spaces, reward shaping, separability, reproducibility
assets/        static Three.js frontend for the optional live visualization
logs/, models/ training logs (TensorBoard + CSV) and saved models
```

## Install & run (uv only)

This project uses [`uv`](https://docs.astral.sh/uv/) for all dependency and
environment management. From a clean clone:

```bash
uv sync
uv run main.py
```

`main.py` loads the best saved model it can find under `models/` (or falls
back to a randomly-initialized policy, and says so clearly) and runs a
matplotlib-rendered demo episode. This works headless (`Agg` backend) --
no display required for anything except the interactive `render_mode="human"`
window.

### Day-1 separability gate

Before trusting any RL result, run the feature-separability check (renders
sample slices, fits a logistic regression on/off target, reports accuracy):

```bash
uv run python scripts/separability_check.py
```

Outputs a montage + accuracy report to `logs/separability/`.

### Tests

```bash
uv run pytest
```

### Smoke-testing the training pipeline

**Do not run full sweeps yourself** -- each algorithm's real sweep is ~16-24
grid combinations x 2 seeds and is meant to be run by the project owner over
several days. Use `--smoke` to prove the pipeline works end-to-end with a
tiny step budget:

```bash
uv run python -m training.sweep --algo dqn --smoke
uv run python -m training.sweep --algo reinforce --smoke
uv run python -m training.sweep --algo a2c --smoke
uv run python -m training.sweep --algo ppo --smoke
```

Full sweeps read grids from `training/configs/*.yaml` and support:

```bash
# grid runs (short "grid_timesteps" budget, single_target curriculum, the
# 10+-run hyperparameter tables):
uv run python -m training.sweep --algo ppo --curriculum single_target --budget grid --n-envs 4

# headline runs (longer "headline_timesteps" budget, for report-quality
# curves -- run a handful of these with each algorithm's best grid config):
uv run python -m training.sweep --algo ppo --curriculum single_target --budget headline --n-envs 4

# the full graded task (head -> abdomen -> femur + AGA/SGA), reserved for
# a small number of headline runs, e.g. capped with a wall-clock limit for
# a bounded probe of whether it's learning anything at all:
uv run python -m training.sweep --algo ppo --curriculum full_task --budget headline \
    --n-envs 4 --max-wall-clock-seconds 1800
```

`--n-envs` uses `SubprocVecEnv` for DQN/A2C/PPO (REINFORCE is single-env
only -- it's a from-scratch full-episode-at-a-time implementation, not
VecEnv-based; `--n-envs` is accepted but ignored for it, see
`training/pg_training.py::train_reinforce`).

### Evaluation & plots

```bash
uv run python -m evaluation.evaluate --model-path models/ppo/best --n-episodes 50
uv run python -m evaluation.generalization --model-path models/ppo/best
uv run python -m evaluation.plots
```

All plotting reads real data from `logs/` -- nothing is fabricated. Any
placeholder plot generated from a smoke test is labeled as such in its
title.

### Optional: live Three.js visualization

A FastAPI/WebSocket bridge can stream probe pose + slice + reward to a
static Three.js page for a nicer live demo. This is entirely optional,
never imported by training code, and launched separately:

```bash
uv run python scripts/serve_viz.py
# then open assets/index.html in a browser
```

See `environment/rendering.py` for the documented JSON message schema sent
over the WebSocket.

## Hard constraints followed in this build

- No hyperparameter sweeps were run by the assistant that built this --
  only short `--smoke` runs to validate the pipeline.
- No fabricated metrics, reward curves, or results appear anywhere in this
  repo. Any figure present was generated from a real (smoke-scale) run and
  is labeled as such.
- REINFORCE is implemented from scratch in PyTorch (`training/reinforce.py`)
  since Stable-Baselines3 has no REINFORCE; it exposes `.learn()` /
  `.predict()` so it shares the training/eval harness with the SB3 algorithms.
- The target plane's pose is never part of the observation vector.
- Training uses derived features only, never raw pixels (no CNN).
