"""Guards against a real bug found during Phase 2/3 validation: PyYAML's
`safe_load` parses bare scientific notation like `3e-4` as a STRING, not a
float (it requires a decimal point in the mantissa, e.g. `3.0e-4`). This
was silently wrong in all four config files' `learning_rate` grids --
`--smoke` runs never caught it because they don't read the configs at all,
and it only surfaced when running a real (non-smoke) sweep, where
Stable-Baselines3 raises `AssertionError: The learning rate schedule must
be a float or a callable, not 3e-4`. This test loads every real config the
way `training/sweep.py` does and asserts every grid value has the type
the corresponding trainer function expects.
"""
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "training" / "configs"

NUMERIC_KEYS = {
    "learning_rate", "gamma", "entropy_coef", "gae_lambda", "clip_range",
    "exploration_fraction",
}
INT_KEYS = {"buffer_size", "batch_size", "target_update_interval", "n_steps", "n_epochs"}


def test_all_configs_parse_numeric_grid_values_as_numbers_not_strings():
    for path in CONFIG_DIR.glob("*.yaml"):
        with open(path) as f:
            cfg = yaml.safe_load(f)
        grid = cfg.get("grid", {})
        for key, values in grid.items():
            if key not in NUMERIC_KEYS and key not in INT_KEYS:
                continue
            values = values if isinstance(values, list) else [values]
            for v in values:
                assert isinstance(v, (int, float)), (
                    f"{path.name}: grid[{key!r}] contains {v!r} ({type(v).__name__}), "
                    f"not a number -- likely a bare scientific-notation literal "
                    f"(e.g. '3e-4') that PyYAML parsed as a string; use '3.0e-4' instead"
                )


def test_all_configs_have_tiered_budgets():
    for path in CONFIG_DIR.glob("*.yaml"):
        with open(path) as f:
            cfg = yaml.safe_load(f)
        assert "grid_timesteps" in cfg, f"{path.name}: missing grid_timesteps"
        assert "headline_timesteps" in cfg, f"{path.name}: missing headline_timesteps"
        assert cfg["headline_timesteps"] >= cfg["grid_timesteps"]
