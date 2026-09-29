"""Contract ordering, BrainCo normalisation/retargeting and the CasADi FK shim used by the IK."""
import os

import numpy as np
import pytest

from teleop.robot_control.brainco_retarget import (BRAINCO_RETARGET_RANGES, CONTRACT_FROM_RETARGET,
                                                   brainco_normalize)
from teleop.webcam import geometry as g

TELEOP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_contract_hand_order_and_names():
    from teleop.robot_control.ros2_backend import HAND_SLOTS, hand_joint_names, UPPER_BODY_JOINTS
    assert HAND_SLOTS == ["thumb", "thumb_aux", "index", "middle", "ring", "pinky"]
    names = hand_joint_names("left")
    # thumb = flexion (thumb_proximal), thumb_aux = rotation (thumb_metacarpal)
    assert names[:2] == ["left_thumb_proximal_joint", "left_thumb_metacarpal_joint"]
    retarget_order = ["metacarpal", "proximal", "index", "middle", "ring", "pinky"]
    assert [retarget_order[i] for i in CONTRACT_FROM_RETARGET] == ["proximal", "metacarpal", "index", "middle", "ring", "pinky"]
    assert len(UPPER_BODY_JOINTS) == 17 and UPPER_BODY_JOINTS[:3] == ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    assert UPPER_BODY_JOINTS[3] == "left_shoulder_pitch_joint" and UPPER_BODY_JOINTS[10] == "right_shoulder_pitch_joint"


def test_brainco_normalize_ranges():
    np.testing.assert_allclose(brainco_normalize(BRAINCO_RETARGET_RANGES), 1.0)
    np.testing.assert_allclose(brainco_normalize(np.zeros(6)), 0.0)
    np.testing.assert_allclose(brainco_normalize(-np.ones(6)), 0.0)
    np.testing.assert_allclose(brainco_normalize(BRAINCO_RETARGET_RANGES / 2), 0.5)


def _curl(p21, amount):
    """Rotate each finger chain about the knuckle axis towards the palm by `amount` (0..1)."""
    p = p21.copy()
    palm_n = np.array([0, -1.0, 0])                  # left hand palm faces -y
    for mcp in (5, 9, 13, 17):
        ang = amount * np.radians(80)
        pivot = p[mcp].copy()
        for k in range(1, 4):
            joint = p[mcp + k - 1]
            ax = np.array([0, 0, 1.0])               # knuckle axis ~ pinky->index (z)
            R = g.so3_exp(-ax * ang)                  # fingers bend towards -y (palm side)
            for j in range(k, 4):
                p[mcp + j] = joint + R @ (p[mcp + j] - joint)
        del pivot
    # thumb across the palm
    p[2:5] = p[2:5] + amount * np.array([0.0, -0.02, -0.03])
    return p


@pytest.fixture
def retargeting():
    pytest.importorskip("dex_retargeting")
    cwd = os.getcwd()
    os.chdir(TELEOP_DIR)                             # upstream config paths are relative to teleop/
    try:
        from teleop.robot_control.hand_retargeting import HandRetargeting, HandType
        yield HandRetargeting(HandType.BRAINCO_HAND)
    finally:
        os.chdir(cwd)


def test_retarget_open_vs_closed_hand(retargeting):
    from teleop.robot_control.brainco_retarget import brainco_retarget_dual

    def keypoints(p21, side):
        R_hand = g.hand_frame_from_arm_frame(g.palm_frame(p21), side)
        return g.hand_keypoints_in_hand_frame(g.mediapipe21_to_xr25(p21), R_hand)

    open_l = keypoints(g.synthetic_open_hand21("left"), "left")
    open_r = keypoints(g.synthetic_open_hand21("right"), "right")
    for _ in range(20):                               # the retargeter low-pass filters its output
        lo, ro = brainco_retarget_dual(retargeting, open_l, open_r)
    closed = _curl(g.synthetic_open_hand21("left"), 1.0)
    closed_l = keypoints(closed, "left")
    for _ in range(40):
        lc, _ = brainco_retarget_dual(retargeting, closed_l, open_r)
    lo_c, lc_c = lo[CONTRACT_FROM_RETARGET], lc[CONTRACT_FROM_RETARGET]
    assert np.all((lo >= 0) & (lo <= 1)) and np.all((lc >= 0) & (lc <= 1))
    assert lo_c[2:].max() < 0.35, lo_c                # open fingers
    assert lc_c[2:].mean() > lo_c[2:].mean() + 0.3, (lo_c, lc_c)   # curled fingers close


def test_casadi_fk_shim_matches_pinocchio():
    casadi = pytest.importorskip("casadi")
    pin = pytest.importorskip("pinocchio")
    from teleop.robot_control import casadi_kinematics as ck
    model = pin.buildSampleModelManipulator()
    data = model.createData()
    cm = ck.Model(model)
    cd = cm.createData()
    q = casadi.SX.sym("q", model.nq)
    ck.framesForwardKinematics(cm, cd, q)
    fid = model.nframes - 1
    f = casadi.Function("f", [q], [cd.oMf[fid].translation, cd.oMf[fid].rotation])
    for _ in range(10):
        qq = pin.randomConfiguration(model)
        pin.framesForwardKinematics(model, data, qq)
        p, R = f(qq)
        np.testing.assert_allclose(np.array(p).ravel(), data.oMf[fid].translation, atol=1e-10)
        np.testing.assert_allclose(np.array(R), data.oMf[fid].rotation, atol=1e-10)
    Rs = casadi.SX.sym("R", 3, 3)
    lf = casadi.Function("l", [Rs], [ck.log3(Rs)])
    rng = np.random.default_rng(0)
    for w in [rng.normal(size=3) * 0.7 for _ in range(10)] + [np.zeros(3), np.array([1e-9, 0, 0])]:
        np.testing.assert_allclose(np.array(lf(pin.exp3(w))).ravel(), w, atol=1e-9)
