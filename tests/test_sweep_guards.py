"""Guards a known reliability footgun in `training/sweep.py`: combining a
wall-clock cap (`--max-wall-clock-seconds`) with `n_envs > 1` was observed
to hang indefinitely on Windows (a `SubprocVecEnv` shutdown deadlock after
the cap fires -- see status.md Phase 2 "Bug #3"). `run_sweep` refuses this
combination outright rather than warning, so it can't be silently
rediscovered hours into an unattended real sweep.
"""
import pytest

from training.sweep import run_sweep, UnsafeWallClockVecEnvCombo, _check_wall_clock_vecenv_combo


@pytest.mark.parametrize("n_envs,cap", [(2, 100.0), (4, 1800.0), (8, 1.0)])
def test_wall_clock_cap_with_multi_env_is_refused(n_envs, cap):
    with pytest.raises(UnsafeWallClockVecEnvCombo):
        _check_wall_clock_vecenv_combo(n_envs, cap)


@pytest.mark.parametrize("n_envs,cap", [(1, 100.0), (1, None), (4, None), (8, None)])
def test_safe_combos_are_not_refused(n_envs, cap):
    _check_wall_clock_vecenv_combo(n_envs, cap)  # must not raise


def test_run_sweep_refuses_before_touching_the_filesystem(tmp_path, monkeypatch):
    """The guard must fire before run_sweep does any real work (log/model
    dir creation, training) -- verified by pointing LOGS_DIR/MODELS_DIR at
    an empty tmp_path and confirming nothing gets created."""
    import training.sweep as sweep_module

    monkeypatch.setattr(sweep_module, "LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(sweep_module, "MODELS_DIR", str(tmp_path / "models"))

    with pytest.raises(UnsafeWallClockVecEnvCombo):
        run_sweep("ppo", None, smoke=True, n_envs=2, max_wall_clock_seconds=60.0)

    assert not (tmp_path / "logs").exists()
    assert not (tmp_path / "models").exists()
