"""Analytic fetal + maternal geometry and the growth/IUGR sampling model.

Everything is built from closed-form primitives (ellipsoids, capsules,
spheres, boxes/slabs) so the ray-caster in `slicer.py` can test ray/primitive
intersection and inside-tests analytically and vectorized with numpy -- no
external meshes needed.

Frames
------
World frame: origin at maternal umbilicus. x = maternal left (+),
y = cranial (+), z = anterior / out of belly (+).

The maternal abdominal surface is an ellipsoid restricted to z > 0 with
semi-axes MATERNAL_ABDOMEN_SEMI_AXES. The probe is parameterized by two
surface angles (theta, phi) on that ellipsoid.

The fetus is a rigid assembly of primitives defined in a *fetal-local*
frame, then placed in the world/uterus frame by a single rigid transform
(fetal_position, fetal_rotation) that encodes presentation (cephalic/
breech), lie (longitudinal/transverse), and spine rotation
(anterior/posterior). Growth simply rescales primitive sizes/positions
before the rigid placement -- it does not change topology.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Literal

from environment import clinical_constants as cc

MATERNAL_ABDOMEN_SEMI_AXES = np.array([0.14, 0.18, 0.10])  # m


TissueKind = Literal[
    "bone", "bright_line", "hypoechoic", "anechoic", "midgrey",
    "moderate", "clutter", "fat_band",
]

# Base acoustic properties per tissue kind: (brightness in [0,1],
# attenuation multiplier relative to soft tissue, casts_shadow)
TISSUE_ACOUSTICS = {
    "bone": dict(brightness=0.95, attenuation=10.0, shadow=True),
    "bright_line": dict(brightness=0.85, attenuation=1.5, shadow=False),
    "hypoechoic": dict(brightness=0.25, attenuation=0.8, shadow=False),
    "anechoic": dict(brightness=0.03, attenuation=0.3, shadow=False),
    "midgrey": dict(brightness=0.45, attenuation=1.0, shadow=False),
    "moderate": dict(brightness=0.55, attenuation=1.1, shadow=False),
    "clutter": dict(brightness=0.4, attenuation=1.0, shadow=False),
    "fat_band": dict(brightness=0.5, attenuation=1.0, shadow=False),
}


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@dataclass
class Primitive:
    name: str
    kind: str  # "ellipsoid" | "capsule" | "sphere" | "box"
    tissue: str
    center: np.ndarray  # (3,) world frame
    rotation: np.ndarray  # (3,3) world frame, local axes as columns
    params: dict  # shape-specific, in local units (meters)
    measured_group: str | None = None  # "head" | "abdomen" | "femur" | None
    required_for: str | None = None  # target this primitive is REQUIRED visible for
    excluder_for: str | None = None  # target this primitive must NOT be visible for


@dataclass
class PlaneTarget:
    """Ground-truth ideal scan plane for a target structure, world frame."""
    name: str
    normal: np.ndarray  # unit normal, world frame
    point: np.ndarray  # a point on the plane, world frame


@dataclass
class FetalPhantom:
    ga_weeks: float
    presentation: str  # "cephalic" | "breech"
    lie: str  # "longitudinal" | "transverse"
    spine_orientation: str  # "anterior" | "posterior"
    g_head: float
    g_abdo: float
    is_iugr: bool
    fetal_position: np.ndarray  # world frame, m
    fetal_rotation: np.ndarray  # 3x3, world frame

    biometry_mm: dict = field(default_factory=dict)
    primitives: list = field(default_factory=list)
    plane_targets: dict = field(default_factory=dict)

    def to_world(self, local_point: np.ndarray) -> np.ndarray:
        return self.fetal_position + self.fetal_rotation @ local_point

    def to_world_dir(self, local_dir: np.ndarray) -> np.ndarray:
        return self.fetal_rotation @ local_dir


def sample_growth(rng: np.random.Generator) -> tuple[float, float, bool]:
    """Sample (g_head, g_abdo, is_iugr) growth factors."""
    if rng.random() < cc.IUGR_TRAINING_PREVALENCE:
        g_head = cc.IUGR_HEAD_GROWTH_FACTOR + rng.normal(0, 0.02)
        g_abdo = cc.IUGR_ABDO_GROWTH_FACTOR + rng.normal(0, 0.02)
        return float(g_head), float(g_abdo), True
    g_head = float(rng.normal(cc.NORMAL_GROWTH_MEAN, cc.NORMAL_GROWTH_SD))
    g_abdo = float(rng.normal(cc.NORMAL_GROWTH_MEAN, cc.NORMAL_GROWTH_SD))
    return g_head, g_abdo, False


def sample_biometry(ga_weeks: float, g_head: float, g_abdo: float) -> dict:
    bpd = (cc.BPD_SLOPE_MM_PER_WEEK * ga_weeks + cc.BPD_INTERCEPT_MM) * g_head
    hc = (cc.HC_SLOPE_MM_PER_WEEK * ga_weeks + cc.HC_INTERCEPT_MM) * g_head
    ac = (cc.AC_SLOPE_MM_PER_WEEK * ga_weeks + cc.AC_INTERCEPT_MM) * g_abdo
    fl = (cc.FL_SLOPE_MM_PER_WEEK * ga_weeks + cc.FL_INTERCEPT_MM) * g_abdo
    return dict(BPD=bpd, HC=hc, AC=ac, FL=fl)


def build_phantom(
    rng: np.random.Generator,
    allow_transverse: bool = False,
    force_ga: float | None = None,
    force_iugr: bool | None = None,
) -> FetalPhantom:
    """Sample a full fetal phantom (pose + growth + primitives)."""
    ga = force_ga if force_ga is not None else float(rng.uniform(cc.GA_MIN_WEEKS, cc.GA_MAX_WEEKS))
    presentation = "cephalic" if rng.random() < cc.CEPHALIC_PRESENTATION_PROB else "breech"
    spine_orientation = "anterior" if rng.random() < cc.SPINE_ANTERIOR_PROB else "posterior"

    if allow_transverse:
        lie = str(rng.choice(["longitudinal", "transverse"]))
    else:
        lie = "longitudinal"

    g_head, g_abdo, is_iugr = sample_growth(rng)
    if force_iugr is True and not is_iugr:
        g_head, g_abdo, is_iugr = cc.IUGR_HEAD_GROWTH_FACTOR, cc.IUGR_ABDO_GROWTH_FACTOR, True

    biometry = sample_biometry(ga, g_head, g_abdo)

    # Rigid fetal placement roughly centered inside the uterine cavity,
    # below the maternal surface (z smaller than probe contact point).
    fetal_position = np.array([
        rng.uniform(-0.02, 0.02),
        rng.uniform(-0.03, 0.03),
        rng.uniform(-0.06, -0.03),
    ])

    if lie == "longitudinal":
        yaw = rng.uniform(-0.2, 0.2)
        head_up = presentation == "breech"  # breech: head cranial; cephalic: head caudal
        base_rot = _rot_y(yaw) @ (_rot_x(np.pi) if head_up else np.eye(3))
    else:  # transverse
        base_rot = _rot_z(np.pi / 2 * float(rng.choice([-1, 1]))) @ _rot_y(rng.uniform(-0.2, 0.2))

    spine_flip = _rot_x(np.pi) if spine_orientation == "posterior" else np.eye(3)
    fetal_rotation = base_rot @ spine_flip

    phantom = FetalPhantom(
        ga_weeks=ga, presentation=presentation, lie=lie,
        spine_orientation=spine_orientation, g_head=g_head, g_abdo=g_abdo,
        is_iugr=is_iugr, fetal_position=fetal_position, fetal_rotation=fetal_rotation,
        biometry_mm=biometry,
    )
    _build_primitives(phantom)
    _build_plane_targets(phantom)
    return phantom


def _build_primitives(p: FetalPhantom) -> None:
    """Populate p.primitives in the fetal-local frame -> stored in world frame.

    Fetal-local frame (before rigid placement): local +y points from feet
    toward head (crown), local +z points fetal-ventral (belly-forward),
    local +x points fetal-left. This local frame is what all the offsets
    below are defined in; `spine_flip`/presentation/lie rotate it into world.
    """
    hc_mm = p.biometry_mm["HC"]
    bpd_mm = p.biometry_mm["BPD"]
    ac_mm = p.biometry_mm["AC"]
    fl_mm = p.biometry_mm["FL"]

    head_r = (hc_mm / (2 * np.pi)) / 1000.0  # approx head radius, m
    abdo_r = (ac_mm / (2 * np.pi)) / 1000.0
    fem_len = (fl_mm / 1000.0)

    # Local anatomical layout along local-y (feet=-y .. head=+y):
    head_center_local = np.array([0.0, 0.09, 0.0])
    abdo_center_local = np.array([0.0, 0.0, 0.0])
    femur_center_local = np.array([0.03, -0.08, 0.0])

    prims: list[Primitive] = []

    def add(name, kind, tissue, center_local, rot_local, params,
            measured_group=None, required_for=None, excluder_for=None):
        c_world = p.to_world(center_local)
        r_world = p.fetal_rotation @ rot_local
        prims.append(Primitive(name, kind, tissue, c_world, r_world, params,
                                measured_group, required_for, excluder_for))

    # --- Head structures ---
    skull_axes = np.array([head_r * (bpd_mm / hc_mm * 2 * np.pi / 2 / head_r if head_r else 1.0),
                            head_r, head_r]) if head_r > 0 else np.array([0.04, 0.045, 0.045])
    skull_axes = np.clip(skull_axes, 0.02, 0.08)
    add("skull", "ellipsoid_shell", "bone", head_center_local, np.eye(3),
        dict(semi_axes=skull_axes, thickness=0.002),
        measured_group="head", required_for="head")
    add("falx", "box", "bright_line", head_center_local, np.eye(3),
        dict(half_extents=np.array([0.001, skull_axes[1] * 0.7, skull_axes[2] * 0.7])),
        measured_group="head", required_for="head")
    thal_off = skull_axes[0] * 0.25
    add("thalamus_L", "ellipsoid", "hypoechoic",
        head_center_local + np.array([thal_off, 0, 0]), np.eye(3),
        dict(semi_axes=skull_axes * 0.18), measured_group="head", required_for="head")
    add("thalamus_R", "ellipsoid", "hypoechoic",
        head_center_local + np.array([-thal_off, 0, 0]), np.eye(3),
        dict(semi_axes=skull_axes * 0.18), measured_group="head", required_for="head")
    add("csp", "box", "anechoic",
        head_center_local + np.array([0, 0, skull_axes[2] * 0.3]), np.eye(3),
        dict(half_extents=skull_axes * 0.08), measured_group="head", required_for="head")
    # Cerebellum sits posterior/inferior to the correct head plane -> excluder.
    add("cerebellum", "ellipsoid", "hypoechoic",
        head_center_local + np.array([0, -skull_axes[1] * 0.6, -skull_axes[2] * 0.5]), np.eye(3),
        dict(semi_axes=skull_axes * 0.35), measured_group="head", excluder_for="head")

    # --- Abdomen structures ---
    abdo_axes = np.clip(np.array([abdo_r, abdo_r * 0.9, abdo_r]), 0.02, 0.09)
    add("abdomen_body", "ellipsoid", "midgrey", abdo_center_local, np.eye(3),
        dict(semi_axes=abdo_axes), measured_group="abdomen")
    add("stomach", "sphere", "anechoic",
        abdo_center_local + np.array([abdo_axes[0] * 0.4, 0, abdo_axes[2] * 0.2]), np.eye(3),
        dict(radius=abdo_axes[0] * 0.22), measured_group="abdomen", required_for="abdomen")
    add("umbilical_vein", "capsule", "anechoic",
        abdo_center_local + np.array([-abdo_axes[0] * 0.15, 0, abdo_axes[2] * 0.55]),
        _rot_z(np.pi / 5),
        dict(radius=abdo_axes[0] * 0.08, half_length=abdo_axes[0] * 0.35),
        measured_group="abdomen", required_for="abdomen")
    add("spine_abdo", "capsule", "bone",
        abdo_center_local + np.array([0, 0, -abdo_axes[2] * 0.85]), np.eye(3),
        dict(radius=abdo_axes[0] * 0.15, half_length=abdo_axes[1] * 0.9),
        measured_group="abdomen", required_for="abdomen")
    # Heart & kidneys are outside the correct AC plane -> excluders.
    add("heart", "ellipsoid", "clutter",
        abdo_center_local + np.array([0, abdo_axes[1] * 0.9, abdo_axes[2] * 0.1]), np.eye(3),
        dict(semi_axes=abdo_axes * 0.3), measured_group="abdomen", excluder_for="abdomen")
    add("kidney_L", "ellipsoid", "clutter",
        abdo_center_local + np.array([abdo_axes[0] * 0.5, -abdo_axes[1] * 0.7, -abdo_axes[2] * 0.3]),
        np.eye(3), dict(semi_axes=abdo_axes * 0.2), measured_group="abdomen", excluder_for="abdomen")
    add("kidney_R", "ellipsoid", "clutter",
        abdo_center_local + np.array([-abdo_axes[0] * 0.5, -abdo_axes[1] * 0.7, -abdo_axes[2] * 0.3]),
        np.eye(3), dict(semi_axes=abdo_axes * 0.2), measured_group="abdomen", excluder_for="abdomen")

    # --- Femur ---
    add("femur", "capsule", "bone", femur_center_local, _rot_z(np.pi / 2),
        dict(radius=0.004, half_length=fem_len / 2), measured_group="femur", required_for="femur")

    # --- Spine chain (clutter elsewhere), placenta, fluid, maternal layers ---
    for i, t in enumerate(np.linspace(-0.09, 0.05, 6)):
        add(f"vertebra_{i}", "sphere", "bone",
            np.array([0, t, -abdo_axes[2] * 0.7]), np.eye(3),
            dict(radius=0.006), measured_group=None)

    add("placenta", "box", "moderate", np.array([0.10, 0.0, 0.02]), np.eye(3),
        dict(half_extents=np.array([0.02, 0.07, 0.01])), measured_group=None)

    add("limb_1", "capsule", "clutter", np.array([0.06, 0.02, 0.03]), _rot_x(0.6),
        dict(radius=0.006, half_length=0.03), measured_group=None)
    add("limb_2", "capsule", "clutter", np.array([-0.06, -0.02, 0.02]), _rot_x(-0.6),
        dict(radius=0.006, half_length=0.03), measured_group=None)

    p.primitives = prims


def _build_plane_targets(p: FetalPhantom) -> None:
    """Ideal cut plane (normal + point), world frame, per target structure.

    The head/abdomen planes are axial (local-y = normal, through the
    respective center); the femur plane's normal is perpendicular to the
    femur's long axis-*and* the imaging depth axis, i.e. the plane containing
    the femur's long axis (its "long-axis view").
    """
    def find(name):
        return next(pr for pr in p.primitives if pr.name == name)

    head = find("skull")
    abdo = find("abdomen_body")
    femur = find("femur")

    head_normal = p.to_world_dir(np.array([0.0, 1.0, 0.0]))
    head_normal /= np.linalg.norm(head_normal)
    p.plane_targets["head"] = PlaneTarget("head", head_normal, head.center)

    abdo_normal = p.to_world_dir(np.array([0.0, 1.0, 0.0]))
    abdo_normal /= np.linalg.norm(abdo_normal)
    p.plane_targets["abdomen"] = PlaneTarget("abdomen", abdo_normal, abdo.center)

    # Femur long axis is local x (after the pi/2 z-rotation applied at
    # construction, its local capsule axis points along local-x); the plane
    # containing that axis and the local depth axis has normal = local "y".
    femur_normal = p.to_world_dir(np.array([0.0, 0.0, 1.0]))
    # rotate normal to be perpendicular to the femur long axis specifically
    femur_axis_world = femur.rotation @ np.array([0.0, 0.0, 1.0])
    femur_axis_world /= np.linalg.norm(femur_axis_world)
    femur_normal = femur_normal - np.dot(femur_normal, femur_axis_world) * femur_axis_world
    if np.linalg.norm(femur_normal) < 1e-6:
        femur_normal = p.to_world_dir(np.array([1.0, 0.0, 0.0]))
    femur_normal /= np.linalg.norm(femur_normal)
    p.plane_targets["femur"] = PlaneTarget("femur", femur_normal, femur.center)
