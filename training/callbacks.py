"""Shared SB3 callback(s) used by the sweep runner."""
from __future__ import annotations

import os
import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class WallClockLimitCallback(BaseCallback):
    """Stops `.learn()` once `max_seconds` of wall-clock time have elapsed,
    regardless of `total_timesteps`. Used to cap the Phase-2 bounded
    full-task probe run (see training/sweep.py `--max-wall-clock-seconds`)
    so a slow config can't silently run for hours."""

    def __init__(self, max_seconds: float, verbose: int = 0):
        super().__init__(verbose)
        self.max_seconds = max_seconds
        self._start = None

    def _on_training_start(self) -> None:
        self._start = time.monotonic()

    def _on_step(self) -> bool:
        return (time.monotonic() - self._start) < self.max_seconds


class StartCurriculumPushScheduleCallback(BaseCallback):
    """Staged, SUCCESS-GATED schedule for
    `UltrasoundProbeEnv.start_curriculum_push_prob` (see that attribute's
    docstring / status.md "navigation-skill gap" pass). Unlike the earlier
    `StartCurriculumWideningCallback` (scripts/generalization_check.py),
    which advanced on a blind timestep trigger and caused catastrophic
    forgetting on its final (large) jump, this callback:

      (a) requires a MINIMUM number of timesteps at each stage before
          being eligible to advance (so the rolling online success
          estimate reflects a real sample, not 2-3 lucky/unlucky
          episodes),
      (b) only advances once the rolling online success rate meets that
          stage's OWN floor (floors are expected to decrease at harder
          stages -- genuinely harder episodes succeeding less often is
          expected, not a failure; the floor is a "not actively
          collapsing" check, not a demand for near-100% everywhere), and
      (c) if a stage's max timestep budget is exhausted without meeting
          its floor, advancement STOPS ENTIRELY and training continues at
          the last stage that did meet its floor, instead of barreling
          forward into a regime the policy has shown it isn't ready for.

    Also saves a checkpoint at the END of every stage successfully
    reached (the earlier attempt's documented methodological gap: only
    the final checkpoint was saved, so a later collapse couldn't be
    rolled back from -- see status.md).

    Requires the training env's `Monitor(info_keywords=...)` to include
    "success" (see `custom_env.py` step() info dict) -- that is what
    populates `self.model.ep_info_buffer` entries with a "success" key,
    which is where the rolling success estimate comes from."""

    def __init__(self, schedule: list[dict], checkpoint_dir: str, verbose: int = 0):
        """schedule: list of dicts, each with keys:
          push_prob (float), min_timesteps (int), max_timesteps (int),
          success_floor (float) -- see class docstring."""
        super().__init__(verbose)
        self.schedule = schedule
        self.checkpoint_dir = checkpoint_dir
        self._stage_idx = 0
        self._stage_start_step = 0
        self._advancing_stopped = False
        self._applied_first = False
        self.stage_log: list[dict] = []

    def _rolling_success(self):
        buf = self.model.ep_info_buffer
        if not buf:
            return None
        vals = [ep["success"] for ep in buf if "success" in ep]
        if not vals:
            return None
        return float(np.mean(vals))

    def _apply_stage(self, idx: int):
        push_prob = self.schedule[idx]["push_prob"]
        self.training_env.env_method("set_start_curriculum", True, None, push_prob)
        if self.verbose:
            print(f"[push schedule] timestep={self.num_timesteps}: stage {idx} -> push_prob={push_prob}")

    def _save_checkpoint(self, idx: int):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, f"stage{idx}.zip")
        self.model.save(path)
        if self.verbose:
            print(f"[push schedule] saved checkpoint: {path}")

    def _on_step(self) -> bool:
        if not self._applied_first:
            self._apply_stage(0)
            self._applied_first = True
            self._stage_start_step = self.num_timesteps

        if self._advancing_stopped or self._stage_idx >= len(self.schedule) - 1:
            return True

        elapsed = self.num_timesteps - self._stage_start_step
        stage = self.schedule[self._stage_idx]
        if elapsed < stage["min_timesteps"]:
            return True

        success = self._rolling_success()
        ready = success is not None and success >= stage["success_floor"]

        if ready:
            self.stage_log.append(dict(stage=self._stage_idx, timestep=self.num_timesteps,
                                        rolling_success=success, advanced=True))
            self._save_checkpoint(self._stage_idx)
            self._stage_idx += 1
            self._stage_start_step = self.num_timesteps
            self._apply_stage(self._stage_idx)
        elif elapsed >= stage["max_timesteps"]:
            if self.verbose:
                print(f"[push schedule] stage {self._stage_idx} did not meet success_floor="
                      f"{stage['success_floor']} within max_timesteps={stage['max_timesteps']} "
                      f"(last rolling success={success}) -- STOPPING advancement, holding here.")
            self.stage_log.append(dict(stage=self._stage_idx, timestep=self.num_timesteps,
                                        rolling_success=success, advanced=False, stopped=True))
            self._advancing_stopped = True
        return True
