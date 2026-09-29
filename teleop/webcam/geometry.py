"""Pure-numpy geometry for webcam teleoperation (unit tested in teleop/webcam/tests).

Frames
------
cam   : OpenCV optical frame of the *current* (moving) OBSBOT view. x right, y down, z forward.
obs   : frame MediaPipe world landmarks are expressed in. Axes are camera-like but, because the
        models run on crops, the view direction is (approximately) the ray through the crop
        centre rather than the optical axis. With `ray_correction` we model obs = cam rotated
        onto that ray; without it obs == cam.
base  : the optical frame of the camera at yaw = pitch = 0 (fixed to the desk).
body  : operator body frame, robot convention (x forward, y left, z up), origin irrelevant
        (only differences between landmarks are used). The operator faces the camera.
robot : G1 IK frame (pelvis, x forward, y left, z up) - what G1_29_ArmIK.solve_ik expects.

Wrist pose convention expected by the IK (televuer "IPunitree_Brobot_waist_arm"):
  left : x wrist->middle finger, y palm->back of hand, z pinky->index
  right: x wrist->middle finger, y back of hand->palm, z pinky->index
Hand keypoints expected by the BrainCo retargeting (25 x 3, OpenXR joint order), expressed in
the Unitree hand frame with the wrist at the origin:
  left : x palm->back, y middle->wrist, z pinky->index
  right: x palm->back, y middle->wrist, z index->pinky
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

# ------------------------------------------------------------------------------------
# rotations
# ------------------------------------------------------------------------------------


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > eps else v * 0.0


def orthonormalize(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    R2 = U @ Vt
    if np.linalg.det(R2) < 0:
        U[:, -1] *= -1
        R2 = U @ Vt
    return R2


def so3_log(R: np.ndarray) -> np.ndarray:
    cos = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = math.acos(cos)
    if th < 1e-7:
        return np.zeros(3)
    if math.pi - th < 1e-4:
        # near pi: axis from the symmetric part
        A = (R + np.eye(3)) / 2
        axis = normalize(A[:, int(np.argmax(np.diag(A)))])
        return axis * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


def so3_exp(w: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(w)
    if th < 1e-9:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def slerp(R0: np.ndarray, R1: np.ndarray, a: float) -> np.ndarray:
    return R0 @ so3_exp(a * so3_log(R0.T @ R1))


# ------------------------------------------------------------------------------------
# gimbal compensation
# ------------------------------------------------------------------------------------

# body (robot convention) <- base (optical at yaw = pitch = 0), operator facing the camera:
#   body x (operator forward) = -z_base (towards the camera)
#   body y (operator left)    = +x_base (operator's left appears on the image right)
#   body z (up)               = -y_base
R_BODY_BASE = np.array([[0.0, 0.0, -1.0],
                        [1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0]])


def R_base_cam(yaw_right_deg: float, pitch_up_deg: float) -> np.ndarray:
    """Rotation of the current camera optical frame in the base frame.

    Pan is a base-fixed vertical axis (optical +y points down, so turning right is a positive
    rotation about +y), tilt is carried by the pan stage (turning up is a positive rotation
    about the camera +x)."""
    return rot_y(math.radians(yaw_right_deg)) @ rot_x(math.radians(pitch_up_deg))


def ray_rotation(u_px: float, v_px: float, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """R_cam_obs: rotation taking the optical axis onto the ray through pixel (u, v), built as
    yaw-then-pitch so a crop 'virtual camera' keeps its x axis horizontal."""
    d = np.array([(u_px - cx) / fx, (v_px - cy) / fy, 1.0])
    yaw = math.atan2(d[0], d[2])
    pitch_up = math.atan2(-d[1], math.hypot(d[0], d[2]))
    return rot_y(yaw) @ rot_x(pitch_up)


def R_base_obs(yaw_right_deg: float, pitch_up_deg: float, pixel: Optional[Tuple[float, float]] = None,
               intrinsics: Optional[Tuple[float, float, float, float]] = None) -> np.ndarray:
    R = R_base_cam(yaw_right_deg, pitch_up_deg)
    if pixel is not None and intrinsics is not None:
        R = R @ ray_rotation(pixel[0], pixel[1], *intrinsics)
    return R


def observation_to_body(vec_obs: np.ndarray, R_base_obs_: np.ndarray,
                        R_body_base: np.ndarray = R_BODY_BASE) -> np.ndarray:
    """Rotate observation-frame vectors (N x 3 or 3) into the operator body frame."""
    R = R_body_base @ R_base_obs_
    return vec_obs @ R.T


def project(p_base: np.ndarray, yaw_right_deg: float, pitch_up_deg: float,
            fx: float, fy: float, cx: float, cy: float) -> Tuple[float, float]:
    """Pinhole projection of a base-frame point with the camera at (yaw, pitch). Test helper."""
    p_cam = R_base_cam(yaw_right_deg, pitch_up_deg).T @ p_base
    return fx * p_cam[0] / p_cam[2] + cx, fy * p_cam[1] / p_cam[2] + cy


# ------------------------------------------------------------------------------------
# MediaPipe 21 -> OpenXR 25 hand keypoints
# ------------------------------------------------------------------------------------
# MediaPipe: 0 wrist | thumb 1 cmc 2 mcp 3 ip 4 tip | index 5 mcp 6 pip 7 dip 8 tip |
#            middle 9..12 | ring 13..16 | pinky 17..20
# OpenXR   : 0 wrist | thumb 1 metacarpal 2 proximal 3 distal 4 tip |
#            index 5 metacarpal 6 proximal 7 intermediate 8 distal 9 tip | middle 10..14 |
#            ring 15..19 | pinky 20..24
# (OpenXR also has a palm joint; the 25-joint WebXR layout used by televuer does not.)
METACARPAL_ALPHA = 0.25   # finger metacarpal joint = wrist + alpha * (MCP - wrist)

MP_TO_XR = np.full(25, -1, dtype=int)
MP_TO_XR[0:5] = [0, 1, 2, 3, 4]
for _f, (_mp, _xr) in enumerate(zip((5, 9, 13, 17), (5, 10, 15, 20))):
    MP_TO_XR[_xr + 1: _xr + 5] = [_mp, _mp + 1, _mp + 2, _mp + 3]
XR_METACARPALS = {5: 5, 10: 9, 15: 13, 20: 17}   # xr index -> mediapipe MCP used for interpolation

XR_TIPS = [4, 9, 14, 19, 24]
MP_TIPS = [4, 8, 12, 16, 20]


def mediapipe21_to_xr25(p21: np.ndarray, alpha: float = METACARPAL_ALPHA) -> np.ndarray:
    p21 = np.asarray(p21, dtype=float)
    assert p21.shape == (21, 3)
    out = np.zeros((25, 3))
    for xr, mp in enumerate(MP_TO_XR):
        if mp >= 0:
            out[xr] = p21[mp]
    for xr, mcp in XR_METACARPALS.items():
        out[xr] = p21[0] + alpha * (p21[mcp] - p21[0])
    return out


# ------------------------------------------------------------------------------------
# palm frame -> wrist orientation, hand keypoints in the Unitree hand frame
# ------------------------------------------------------------------------------------

def palm_frame(p21_body: np.ndarray) -> np.ndarray:
    """Unitree *arm* wrist frame (columns = axes, in body coordinates) from 21 hand points.

    x: wrist -> middle MCP;  z: pinky MCP -> index MCP (orthogonalised);  y = z cross x.
    This single formula yields the left-arm convention (y palm->back) for a left hand and the
    right-arm convention (y back->palm) for a right hand, because the two hands are mirror
    images: in the neutral pose (thumbs up, palms facing each other, fingers forward) both give
    the identity."""
    w, idx, mid, pky = p21_body[0], p21_body[5], p21_body[9], p21_body[17]
    x = normalize(mid - w)
    z = idx - pky
    z = normalize(z - np.dot(z, x) * x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def hand_frame_from_arm_frame(R_arm: np.ndarray, side: str) -> np.ndarray:
    """Unitree *hand* frame axes (columns) from the Unitree arm wrist frame."""
    x_a, y_a, z_a = R_arm[:, 0], R_arm[:, 1], R_arm[:, 2]
    if side == "left":       # x palm->back = y_a, y middle->wrist = -x_a, z pinky->index = z_a
        return np.column_stack([y_a, -x_a, z_a])
    else:                    # palm->back = -y_a, y = -x_a, z index->pinky = -z_a
        return np.column_stack([-y_a, -x_a, -z_a])


def hand_keypoints_in_hand_frame(p25_body: np.ndarray, R_hand: np.ndarray) -> np.ndarray:
    """(25,3) keypoints relative to the wrist, expressed in the hand frame."""
    return (p25_body - p25_body[0]) @ R_hand


# ------------------------------------------------------------------------------------
# arm mapping
# ------------------------------------------------------------------------------------

def clamp_reach(p: np.ndarray, shoulder: np.ndarray, max_reach: float, min_reach: float = 0.08) -> np.ndarray:
    d = p - shoulder
    n = np.linalg.norm(d)
    if n > max_reach:
        return shoulder + d * (max_reach / n)
    if n < min_reach and n > 1e-9:
        return shoulder + d * (min_reach / n)
    return p


def map_wrist_position(wrist_rel_shoulder_body: np.ndarray, robot_shoulder: np.ndarray, scale: float,
                       offset: np.ndarray, max_reach: float, depth_gain: float = 1.0) -> np.ndarray:
    v = np.array(wrist_rel_shoulder_body, dtype=float).copy()
    v[0] *= depth_gain                      # x (forward) comes from monocular depth: optional gain
    p = robot_shoulder + scale * v + offset
    return clamp_reach(p, robot_shoulder, max_reach)


def se3(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def pinch_distance(p21: np.ndarray) -> float:
    return float(np.linalg.norm(p21[4] - p21[8]))


def finger_curl(p21: np.ndarray) -> float:
    """0 (open) .. 1 (fist) from tip-to-wrist distance relative to MCP-to-wrist distance."""
    w = p21[0]
    vals = []
    for mcp, tip in ((5, 8), (9, 12), (13, 16), (17, 20)):
        base = np.linalg.norm(p21[mcp] - w) + 1e-6
        r = np.linalg.norm(p21[tip] - w) / base       # ~1.9 open, ~1.0 fist
        vals.append(np.clip((1.9 - r) / 0.9, 0.0, 1.0))
    return float(np.mean(vals))


def synthetic_open_hand21(side: str = "left", scale: float = 1.0) -> np.ndarray:
    """A flat open hand in body coordinates in the neutral pose (fingers +x, thumb up +z,
    palm facing the body midline). Used as loss-of-tracking neutral and in tests."""
    s = 1.0 if side == "left" else -1.0          # palm normal: left palm faces -y, right faces +y
    pts = np.zeros((21, 3))
    # finger MCP heights (z) index high .. pinky low; segment lengths (m)
    fingers = {5: 0.030, 9: 0.010, 13: -0.010, 17: -0.028}
    seg = {5: (0.040, 0.025, 0.020), 9: (0.045, 0.028, 0.022), 13: (0.042, 0.026, 0.021),
           17: (0.034, 0.020, 0.018)}
    for mcp, z in fingers.items():
        base = np.array([0.085 if mcp != 17 else 0.078, 0.0, z])
        pts[mcp] = base
        acc = base.copy()
        for k, L in enumerate(seg[mcp]):
            acc = acc + np.array([L, 0.0, 0.0])
            pts[mcp + 1 + k] = acc
    # thumb goes up and forward, slightly towards the palm side
    pts[1] = np.array([0.025, -s * 0.010, 0.025])
    pts[2] = np.array([0.045, -s * 0.015, 0.048])
    pts[3] = np.array([0.065, -s * 0.015, 0.062])
    pts[4] = np.array([0.085, -s * 0.015, 0.072])
    return pts * scale
