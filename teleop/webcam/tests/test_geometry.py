"""Landmark -> wrist math, keypoint mapping, gimbal compensation (pure numpy)."""
import math

import numpy as np
import pytest

from teleop.webcam import geometry as g


def rand_rot(rng):
    return g.so3_exp(rng.normal(size=3))


# ---------------------------------------------------------------- keypoint mapping 21 -> 25
def test_mp21_to_xr25_indices():
    p21 = np.arange(63, dtype=float).reshape(21, 3)
    p25 = g.mediapipe21_to_xr25(p21)
    assert p25.shape == (25, 3)
    np.testing.assert_array_equal(p25[0], p21[0])                  # wrist
    for xr, mp in zip(g.XR_TIPS, g.MP_TIPS):                       # finger tips
        np.testing.assert_array_equal(p25[xr], p21[mp])
    np.testing.assert_array_equal(p25[1:5], p21[1:5])              # thumb chain cmc..tip
    # index: proximal=MCP(5), intermediate=PIP(6), distal=DIP(7)
    np.testing.assert_array_equal(p25[6:9], p21[5:8])
    np.testing.assert_array_equal(p25[21:24], p21[17:20])          # pinky
    # metacarpals interpolated between wrist and MCP
    for xr, mcp in g.XR_METACARPALS.items():
        np.testing.assert_allclose(p25[xr], p21[0] + g.METACARPAL_ALPHA * (p21[mcp] - p21[0]))


def test_dexpilot_indices_hit_wrist_and_tips():
    # brainco.yml DexPilot uses human indices {0, 4, 9, 14, 19, 24}: wrist and the five tips
    used = {9, 14, 19, 24, 0, 4}
    assert used == {0} | set(g.XR_TIPS)


# ---------------------------------------------------------------- palm / hand frames
@pytest.mark.parametrize("side", ["left", "right"])
def test_palm_frame_neutral_is_identity(side):
    # neutral pose (fingers forward, thumbs up, palms facing each other) = televuer/robot
    # neutral wrist orientation = identity in the Unitree arm convention
    R = g.palm_frame(g.synthetic_open_hand21(side))
    np.testing.assert_allclose(R, np.eye(3), atol=0.15)
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-9)
    assert np.linalg.det(R) > 0.999


@pytest.mark.parametrize("side", ["left", "right"])
def test_palm_frame_tracks_rotation(side):
    rng = np.random.default_rng(1)
    p = g.synthetic_open_hand21(side)
    R0 = g.palm_frame(p)
    for _ in range(10):
        Q = rand_rot(rng)
        R = g.palm_frame(p @ Q.T + rng.normal(size=3))
        np.testing.assert_allclose(R, Q @ R0, atol=1e-9)


def test_arm_frame_semantics():
    """left: y palm->back; right: y back->palm; both: x wrist->middle, z pinky->index."""
    for side, palm_normal in (("left", np.array([0, -1.0, 0])), ("right", np.array([0, 1.0, 0]))):
        R = g.palm_frame(g.synthetic_open_hand21(side))
        x, y, z = R.T
        assert x @ np.array([1.0, 0, 0]) > 0.95            # fingers forward
        assert z @ np.array([0, 0, 1.0]) > 0.95            # index above pinky (thumb up)
        palm_to_back = -palm_normal
        if side == "left":
            assert y @ palm_to_back > 0.95
        else:
            assert y @ (-palm_to_back) > 0.95


@pytest.mark.parametrize("side", ["left", "right"])
def test_hand_keypoints_in_unitree_hand_frame(side):
    p21 = g.synthetic_open_hand21(side)
    R_hand = g.hand_frame_from_arm_frame(g.palm_frame(p21), side)
    assert np.linalg.det(R_hand) > 0.999
    k = g.hand_keypoints_in_hand_frame(g.mediapipe21_to_xr25(p21), R_hand)
    np.testing.assert_allclose(k[0], 0.0)
    # y: middle -> wrist, so all finger tips have y < 0
    assert np.all(k[g.XR_TIPS[1:], 1] < -0.05)
    # z: pinky->index (left) / index->pinky (right)
    if side == "left":
        assert k[9, 2] > k[24, 2]
    else:
        assert k[9, 2] < k[24, 2]
    # invariant to where/how the hand is held
    rng = np.random.default_rng(0)
    Q = rand_rot(rng)
    p2 = p21 @ Q.T + 0.3
    R2 = g.hand_frame_from_arm_frame(g.palm_frame(p2), side)
    np.testing.assert_allclose(g.hand_keypoints_in_hand_frame(g.mediapipe21_to_xr25(p2), R2), k, atol=1e-9)


# ---------------------------------------------------------------- wrist position mapping
def test_map_wrist_scales_and_clamps():
    sh = np.array([0.0, 0.1, 0.29])
    v = np.array([0.2, 0.05, -0.1])
    p = g.map_wrist_position(v, sh, 0.6, np.zeros(3), max_reach=0.4)
    np.testing.assert_allclose(p, sh + 0.6 * v)
    far = g.map_wrist_position(np.array([2.0, 0, 0]), sh, 1.0, np.zeros(3), max_reach=0.4)
    assert abs(np.linalg.norm(far - sh) - 0.4) < 1e-9
    dg = g.map_wrist_position(v, sh, 1.0, np.zeros(3), max_reach=1.0, depth_gain=0.5)
    np.testing.assert_allclose(dg - sh, [0.1, 0.05, -0.1])


# ---------------------------------------------------------------- base <-> body
def test_body_axes_from_camera():
    """Operator faces the camera: moving a hand up (image up, -y) is body +z, towards the camera
    (-z optical) is body +x, and the operator's left is the image right (+x)."""
    R = g.R_BODY_BASE
    np.testing.assert_allclose(R @ [0, -1, 0], [0, 0, 1])
    np.testing.assert_allclose(R @ [0, 0, -1], [1, 0, 0])
    np.testing.assert_allclose(R @ [1, 0, 0], [0, 1, 0])
    assert abs(np.linalg.det(R) - 1) < 1e-12


# ---------------------------------------------------------------- gimbal compensation
INTR = (975.0, 975.0, 640.0, 360.0)


def test_R_base_cam_conventions():
    # yaw right: the optical axis turns towards +x_base; pitch up: towards -y_base
    np.testing.assert_allclose(g.R_base_cam(90, 0) @ [0, 0, 1], [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(g.R_base_cam(0, 90) @ [0, 0, 1], [0, -1, 0], atol=1e-12)
    # a point that is straight ahead in the base frame appears on the image LEFT after yawing right
    u, v = g.project(np.array([0, 0, 1.0]), 10, 0, *INTR)
    assert u < INTR[2] and abs(v - INTR[3]) < 1e-6
    u, v = g.project(np.array([0, 0, 1.0]), 0, 10, *INTR)     # pitch up -> appears lower
    assert v > INTR[3]


def test_ray_rotation_points_at_pixel():
    for (u, v) in [(640, 360), (100, 50), (1200, 700), (300, 600)]:
        R = g.ray_rotation(u, v, *INTR)
        d = R @ [0, 0, 1]
        uu, vv = INTR[0] * d[0] / d[2] + INTR[2], INTR[1] * d[1] / d[2] + INTR[3]
        assert abs(uu - u) < 1e-6 and abs(vv - v) < 1e-6
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)


@pytest.mark.parametrize("ray", [False, True])
def test_gimbal_compensation_static_operator(ray):
    """The operator does not move; the gimbal pans/tilts. MediaPipe-style observations
    (camera-aligned, or aligned with the ray through the person when `ray`) are rotated back
    with the gimbal angles: the body-frame arm vector must not change."""
    rng = np.random.default_rng(3)
    person_base = np.array([0.1, 0.05, 0.9])               # torso position in the base frame
    arm_vec_body = np.array([0.25, 0.05, -0.2])            # wrist - shoulder, body frame
    arm_vec_base = g.R_BODY_BASE.T @ arm_vec_body
    outs = []
    for yaw, pitch in [(0, 0), (12, -5), (-20, 8), (30, 15), (-3, -20)]:
        R_bc = g.R_base_cam(yaw, pitch)
        u, v = g.project(person_base, yaw, pitch, *INTR)
        R_co = g.ray_rotation(u, v, *INTR) if ray else np.eye(3)
        obs = (R_bc @ R_co).T @ arm_vec_base               # what the estimator reports
        R_bo = g.R_base_obs(yaw, pitch, (u, v) if ray else None, INTR if ray else None)
        outs.append(g.observation_to_body(obs, R_bo))
    for o in outs:
        np.testing.assert_allclose(o, arm_vec_body, atol=1e-9)


def test_ray_frame_nearly_invariant_to_gimbal():
    """With ray-aligned observations, the observation frame (camera rotation composed with the
    ray to the person) is what stays fixed in the base frame: exactly for its viewing direction,
    exactly for pure pan, and up to a few degrees of roll when pan and tilt are combined
    (documented limit of the yaw-then-pitch crop-frame model)."""
    person_base = np.array([0.15, -0.1, 1.0])
    ref = None
    for yaw, pitch, tol in [(0, 0, 1e-6), (8, 0, 1e-6), (-12, 0, 1e-6), (0, 8, 2.0), (15, -10, 2.0), (-15, 12, 6.0)]:
        u, v = g.project(person_base, yaw, pitch, *INTR)
        R_bo = g.R_base_obs(yaw, pitch, (u, v), INTR)
        z = R_bo @ [0, 0, 1]
        np.testing.assert_allclose(z, person_base / np.linalg.norm(person_base), atol=1e-9)
        if ref is None:
            ref = R_bo
        ang = np.degrees(np.linalg.norm(g.so3_log(ref.T @ R_bo)))
        assert ang < tol, (yaw, pitch, ang)


def test_so3_log_exp_roundtrip():
    rng = np.random.default_rng(5)
    for _ in range(50):
        w = rng.normal(size=3)
        w *= min(1.0, 3.0 / np.linalg.norm(w))
        np.testing.assert_allclose(g.so3_log(g.so3_exp(w)), w, atol=1e-7)
    np.testing.assert_allclose(g.so3_log(np.eye(3)), 0)
    R = g.slerp(np.eye(3), g.rot_z(1.0), 0.5)
    np.testing.assert_allclose(R, g.rot_z(0.5), atol=1e-12)


def test_pinch_and_curl():
    p = g.synthetic_open_hand21("left")
    assert g.finger_curl(p) < 0.15
    fist = p.copy()
    for mcp in (5, 9, 13, 17):          # fold fingers back towards the wrist
        for k in range(1, 4):
            fist[mcp + k] = p[mcp] + (p[0] - p[mcp]) * 0.25 * k + np.array([0, -0.02, 0])
    assert g.finger_curl(fist) > 0.7
    assert g.pinch_distance(p) > 0.03
