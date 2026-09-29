"""Mapper: calibration, filtering, loss handling, gimbal-compensated end-to-end mapping."""
import numpy as np
import pytest

from teleop.webcam import geometry as g
from teleop.webcam.mapper import ArmHandMapper, MapperConfig, RobotArmModel, SideObservation
from teleop.webcam.one_euro import OneEuroFilter, OneEuroRotation

INTR = (975.0, 975.0, 640.0, 360.0)
SHOULDER_BODY = {"left": np.array([0.0, 0.18, 0.0]), "right": np.array([0.0, -0.18, 0.0])}


def human_arm(side, wrist_rel=(0.28, 0.02, -0.22), elbow_rel=(0.02, 0.02, -0.28)):
    """Seated operator in the neutral pose (body frame): upper arm down, forearm forward."""
    sh = SHOULDER_BODY[side]
    s = 1 if side == "left" else -1
    el = sh + np.array([elbow_rel[0], s * elbow_rel[1], elbow_rel[2]])
    wr = sh + np.array([wrist_rel[0], s * wrist_rel[1], wrist_rel[2]])
    return np.stack([sh, el, wr])


def observe(body_pts, yaw=0.0, pitch=0.0, ray_pixel=None, R_body_base=g.R_BODY_BASE):
    """Body-frame points -> what MediaPipe would report for a camera at (yaw, pitch)."""
    R_bo = g.R_base_obs(yaw, pitch, ray_pixel, INTR if ray_pixel else None)
    obs = body_pts @ (R_body_base @ R_bo)          # inverse of observation_to_body
    return obs, R_bo


def make_obs(yaw=0.0, pitch=0.0, arms=True, hands=True, hand_rot=None, wrist_rel=(0.28, 0.02, -0.22)):
    obs = {}
    for side in ("left", "right"):
        o = SideObservation()
        if arms:
            o.arm_obs, o.arm_R_base_obs = observe(human_arm(side, wrist_rel), yaw, pitch)
            o.arm_visibility = 0.95
        if hands:
            h = g.synthetic_open_hand21(side)
            if hand_rot is not None:
                h = h @ hand_rot.T
            o.hand21_obs, o.hand_R_base_obs = observe(h, yaw, pitch)
            o.hand_score = 0.95
        obs[side] = o
    return obs


def test_neutral_pose_after_calibration_hits_robot_neutral():
    rm = RobotArmModel()
    m = ArmHandMapper(rm, MapperConfig())
    m.update(0.0, make_obs())
    ok, msg = m.calibrate(0.0)
    assert ok, msg
    t = 0.0
    for _ in range(40):
        t += 1 / 30
        out = m.update(t, make_obs())
    for side in ("left", "right"):
        np.testing.assert_allclose(out[side].wrist[:3, 3], rm.neutral_pos[side], atol=1e-6)
        np.testing.assert_allclose(out[side].wrist[:3, :3], rm.neutral_rot[side], atol=1e-6)
        assert out[side].arm_state == "live" and out[side].hand_state == "live"
    assert abs(m.calib.human_arm_length["left"] - (np.linalg.norm([0.02, 0.02, -0.28]) + np.linalg.norm([0.26, 0.0, 0.06]))) < 1e-9


def test_motion_is_scaled_to_robot_arm():
    rm = RobotArmModel()
    m = ArmHandMapper(rm, MapperConfig())
    m.update(0.0, make_obs())
    m.calibrate(0.0)
    L_h = m.calib.human_arm_length["left"]
    t = 0.0
    for _ in range(60):
        t += 1 / 30
        out = m.update(t, make_obs(wrist_rel=(0.28, 0.02, -0.12)))     # raise hands by 10 cm
    dz = out["left"].wrist[2, 3] - rm.neutral_pos["left"][2]
    assert abs(dz - 0.10 * rm.arm_length / L_h) < 2e-3


def test_gimbal_motion_does_not_move_targets():
    m = ArmHandMapper(RobotArmModel(), MapperConfig())
    t = 0.0
    ref = None
    for yaw, pitch in [(0, 0)] * 30 + [(15, -8)] * 30 + [(-25, 10)] * 30:
        t += 1 / 30
        out = m.update(t, make_obs(yaw, pitch))
        if ref is None and t > 0.9:
            ref = {s: out[s].wrist.copy() for s in ("left", "right")}
    for s in ("left", "right"):
        np.testing.assert_allclose(out[s].wrist, ref[s], atol=1e-6)


def test_wrist_orientation_follows_hand():
    m = ArmHandMapper(RobotArmModel(), MapperConfig())
    m.update(0.0, make_obs())
    m.calibrate(0.0)
    Q = g.rot_y(np.radians(-40))            # pitch the hands (fingers point up by 40 deg)
    t = 0.0
    for _ in range(90):
        t += 1 / 30
        out = m.update(t, make_obs(hand_rot=Q))
    for s in ("left", "right"):
        np.testing.assert_allclose(out[s].wrist[:3, :3], Q, atol=1e-3)


def test_loss_hold_then_ease_to_neutral_then_reacquire():
    cfg = MapperConfig(hold_s=0.5, ease_s=1.0, reacquire_s=0.4)
    rm = RobotArmModel()
    m = ArmHandMapper(rm, cfg)
    t = 0.0
    for _ in range(30):
        t += 1 / 30
        out = m.update(t, make_obs(wrist_rel=(0.35, 0.05, -0.05)))
    live = out["left"].wrist[:3, 3].copy()
    assert np.linalg.norm(live - rm.neutral_pos["left"]) > 0.05
    t_lost = t
    states = []
    while t < t_lost + 2.0:
        t += 1 / 30
        out = m.update(t, make_obs(arms=False, hands=False))
        states.append(out["left"].arm_state)
        if t - t_lost < 0.45:
            np.testing.assert_allclose(out["left"].wrist[:3, 3], live, atol=1e-9)     # hold
    assert "hold" in states and "easing" in states and states[-1] == "neutral"
    np.testing.assert_allclose(out["left"].wrist[:3, 3], rm.neutral_pos["left"], atol=1e-9)
    np.testing.assert_allclose(out["left"].hand25, m._neutral_hand25("left"), atol=1e-9)
    # reacquire: no jump, blends to the live target over reacquire_s
    t += 1 / 30
    out = m.update(t, make_obs(wrist_rel=(0.35, 0.05, -0.05)))
    assert np.linalg.norm(out["left"].wrist[:3, 3] - rm.neutral_pos["left"]) < 0.01
    for _ in range(20):
        t += 1 / 30
        out = m.update(t, make_obs(wrist_rel=(0.35, 0.05, -0.05)))
    np.testing.assert_allclose(out["left"].wrist[:3, 3], live, atol=2e-3)


def test_low_confidence_counts_as_lost():
    m = ArmHandMapper(RobotArmModel(), MapperConfig())
    obs = make_obs()
    for o in obs.values():
        o.arm_visibility = 0.2
        o.hand_score = 0.1
    out = m.update(0.0, obs)
    assert out["left"].arm_state == "neutral" and out["left"].hand_state == "neutral"


def test_calibration_removes_operator_yaw():
    """Operator sits rotated 20 deg relative to the camera; after calibration the neutral pose
    still maps to the robot neutral and 'forward' means the operator's forward."""
    rm = RobotArmModel()
    m = ArmHandMapper(rm, MapperConfig())
    Rz = g.rot_z(np.radians(20))
    def obs_rot(wrist_rel=(0.28, 0.02, -0.22)):
        obs = {}
        for side in ("left", "right"):
            o = SideObservation()
            o.arm_obs, o.arm_R_base_obs = observe(human_arm(side, wrist_rel) @ Rz.T)
            o.arm_visibility = 0.9
            obs[side] = o
        return obs
    m.update(0.0, obs_rot())
    ok, _ = m.calibrate(0.0)
    assert ok
    t = 0.0
    for _ in range(60):
        t += 1 / 30
        out = m.update(t, obs_rot((0.38, 0.02, -0.22)))       # reach 10 cm forward (operator frame)
    d = out["left"].wrist[:3, 3] - rm.neutral_pos["left"]
    assert d[0] > 0.05 and abs(d[1]) < 2e-3 and abs(d[2]) < 2e-3


def test_calibration_needs_both_arms():
    m = ArmHandMapper(RobotArmModel(), MapperConfig())
    obs = make_obs()
    obs["right"].arm_obs = None
    m.update(0.0, obs)
    ok, msg = m.calibrate(0.0)
    assert not ok and "right" in msg


def test_one_euro_smooths_and_converges():
    rng = np.random.default_rng(0)
    f = OneEuroFilter(1.0, 0.0)
    t, ys = 0.0, []
    for _ in range(300):
        t += 1 / 30
        ys.append(f(np.array([1.0]) + rng.normal(scale=0.02, size=1), t)[0])
    ys = np.array(ys[100:])
    assert abs(ys.mean() - 1.0) < 0.01 and ys.std() < 0.01
    fr = OneEuroRotation(1.0, 0.0)
    t = 0.0
    for _ in range(200):
        t += 1 / 30
        R = fr(g.rot_z(0.5) @ g.so3_exp(rng.normal(scale=0.02, size=3)), t)
    assert np.degrees(np.linalg.norm(g.so3_log(R.T @ g.rot_z(0.5)))) < 1.0
