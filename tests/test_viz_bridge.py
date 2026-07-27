"""Automated de-risking for the optional Three.js/WebSocket viz layer
(`environment/rendering.py::VizBridge`, `assets/index.html`,
`scripts/serve_viz.py`). This is the largest untested surface identified in
status.md -- Three.js itself can't be exercised headlessly, but the FastAPI
app, the WebSocket endpoint, and the JSON payload schema all can be, and
the static frontend asset can be checked for gross wiring mistakes (wrong
port, missing script tags) without a browser.

What this does NOT cover (see README manual-test steps): whether the
Three.js scene actually renders correctly, whether the probe/monitor mesh
positions look right, whether the delivery-room scene is visually sane.
That still requires a human opening assets/index.html in a browser while
scripts/serve_viz.py runs.
"""
import asyncio
import re
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from environment.custom_env import UltrasoundProbeEnv
from environment.rendering import VizBridge

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_INDEX = REPO_ROOT / "assets" / "index.html"
SERVE_VIZ = REPO_ROOT / "scripts" / "serve_viz.py"

REQUIRED_TOP_KEYS = {"probe", "target", "acquired", "reward", "alpha_deg", "d_m", "flag", "slice_png_base64", "step"}
REQUIRED_PROBE_KEYS = {"position", "forward", "right", "up"}


def _validate_schema(payload: dict):
    assert REQUIRED_TOP_KEYS.issubset(payload.keys()), (
        f"missing keys: {REQUIRED_TOP_KEYS - payload.keys()}"
    )
    assert REQUIRED_PROBE_KEYS.issubset(payload["probe"].keys())
    for vec_name in REQUIRED_PROBE_KEYS:
        vec = payload["probe"][vec_name]
        assert isinstance(vec, list) and len(vec) == 3
        assert all(isinstance(x, (int, float)) for x in vec)

    assert payload["target"] is None or payload["target"] in ("head", "abdomen", "femur")
    assert isinstance(payload["acquired"], list)
    assert all(a in ("head", "abdomen", "femur") for a in payload["acquired"])
    assert isinstance(payload["reward"], (int, float))
    assert isinstance(payload["alpha_deg"], (int, float)) and 0.0 <= payload["alpha_deg"] <= 180.0
    assert isinstance(payload["d_m"], (int, float)) and payload["d_m"] >= 0.0
    assert payload["flag"] is None or payload["flag"] in ("AGA", "SGA")
    assert isinstance(payload["slice_png_base64"], str)
    assert isinstance(payload["step"], int) and payload["step"] >= 0


def test_ws_endpoint_accepts_and_closes_connection():
    bridge = VizBridge()
    client = TestClient(bridge.app)
    with client.websocket_connect("/ws") as websocket:
        websocket.close()


def test_publish_payload_matches_documented_schema_across_real_steps():
    """Random actions for 10 steps -- must reset() on termination/truncation,
    same reasoning as test_publish_payload_after_freeze_action_has_valid_flag_or_none:
    under the LOCKED environment's default single_target=True (see status.md
    "make single_target the environment default" pass), a random
    freeze_and_measure can plausibly succeed and end the episode well
    within 10 steps -- stepping a terminated episode without resetting is
    undefined behavior (same as any other Gym env), not something this
    test should assume can't happen."""
    bridge = VizBridge()
    env = UltrasoundProbeEnv(seed=0)
    env.reset(seed=0)

    for _ in range(10):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        payload = asyncio.run(bridge.publish(env))
        _validate_schema(payload)
        if terminated or truncated:
            env.reset(seed=0)


def test_publish_payload_after_freeze_action_has_valid_flag_or_none():
    """Repeated freeze_and_measure without resetting between calls --
    correct Gymnasium usage requires reset() once an episode terminates
    (stepping a terminated episode is undefined behavior, same as any
    other Gym env). Under the LOCKED environment (start_curriculum=True,
    alpha_tol_deg=18 -- see status.md "lock the environment" pass), the
    FIRST freeze from a curriculum-near-optimal start often succeeds
    immediately (that's the point of the lock), so this loop must reset
    on termination rather than assuming (as it implicitly did under the
    old, harder-to-succeed defaults) that 5 blind freezes would never
    terminate the episode."""
    bridge = VizBridge()
    env = UltrasoundProbeEnv(seed=1, single_target=True)
    env.reset(seed=1)
    freeze_action = env.action_space.n - 1  # "freeze_and_measure" is the last action
    for _ in range(5):
        _, _, terminated, truncated, _ = env.step(freeze_action)
        if terminated or truncated:
            env.reset(seed=1)
    payload = asyncio.run(bridge.publish(env))
    _validate_schema(payload)


def test_assets_index_html_is_well_formed_and_wired_correctly():
    html = ASSETS_INDEX.read_text(encoding="utf-8")
    assert "<script" in html and "three" in html.lower(), "no CDN three.js script reference found"
    assert "importmap" in html, "expected an import map for the three.js CDN module"

    ws_urls = re.findall(r'ws://[^"\']+', html)
    assert ws_urls, "no WebSocket URL found in assets/index.html"

    serve_viz_src = SERVE_VIZ.read_text(encoding="utf-8")
    port_match = re.search(r'port=(\d+)', serve_viz_src)
    assert port_match, "could not find the port serve_viz.py binds to"
    port = port_match.group(1)
    assert any(port in url for url in ws_urls), (
        f"assets/index.html's WebSocket URL(s) {ws_urls} don't reference "
        f"serve_viz.py's port {port} -- viz would silently fail to connect"
    )


def test_serve_viz_module_never_imported_by_training_code():
    """VizBridge/serve_viz must stay fully optional -- a grep-level guard
    against training code accidentally importing the viz layer and making
    FastAPI/uvicorn a hard training dependency."""
    training_dir = REPO_ROOT / "training"
    for path in training_dir.glob("*.py"):
        content = path.read_text(encoding="utf-8")
        assert "serve_viz" not in content and "VizBridge" not in content, (
            f"{path} references the optional viz layer -- it must stay decoupled from training"
        )
