"""Landmarks -> robot wrist SE(3) targets + 25-point hand keypoints, with calibration,
One-Euro filtering, per-side confidence and loss handling. No MediaPipe / OpenCV imports,
so it is unit-testable with synthetic landmarks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from . import geometry as g
from .one_euro import OneEuroFilter, OneEuroRotation

SIDES = ("left", "right")


@dataclass
class RobotArmModel:
    """Robot-side numbers in the IK frame (pelvis). Filled from the IK model by
    `robot_arm_model_from_ik`; the defaults are the G1 29-DoF values from that computation."""
    shoulder: Dict[str, np.ndarray] = field(default_factory=lambda: {
        "left": np.array([0.0, 0.1002, 0.2918]), "right": np.array([0.0, -0.1002, 0.2918])})
    arm_length: float = 0.427         # shoulder pitch joint -> elbow -> L_ee/R_ee (arm straight)
    # neutral wrist targets = FK of the zero arm pose (G1 elbows at 90 deg, forearms forward),
    # which is also televuer's constant arm pose (0.25, +-0.15, 0.08 in the IK frame)
    neutral_pos: Dict[str, np.ndarray] = field(default_factory=lambda: {
        "left": np.array([0.2498, 0.1487, 0.0952]), "right": np.array([0.2498, -0.1487, 0.0952])})
    neutral_rot: Dict[str, np.ndarray] = field(default_factory=lambda: {
        "left": np.eye(3), "right": np.eye(3)})


@dataclass
class MapperConfig:
    human_arm_length: float = 0.55     # used until calibration measures it
    max_reach_frac: float = 0.97       # of robot arm length
    depth_gain: float = 1.0            # scale on the (noisy, monocular) forward component
    min_arm_visibility: float = 0.5
    min_hand_score: float = 0.5
    hold_s: float = 0.5                # hold the last target this long after losing tracking
    ease_s: float = 2.0                # then ease to neutral over this long
    reacquire_s: float = 0.4           # blend from the held/eased target back to live tracking
    # One-Euro parameters
    pos_min_cutoff: Tuple[float, float, float] = (0.4, 1.0, 1.0)   # x (depth) filtered harder
    pos_beta: float = 0.6
    rot_min_cutoff: float = 1.0
    rot_beta: float = 0.4
    hand_min_cutoff: float = 1.5
    hand_beta: float = 2.0


@dataclass
class SideObservation:
    """Everything the tracker measured for one side in one frame (observation frames)."""
    arm_obs: Optional[np.ndarray] = None       # (3,3) shoulder, elbow, wrist; pose world landmarks
    arm_visibility: float = 0.0
    arm_R_base_obs: Optional[np.ndarray] = None
    hand21_obs: Optional[np.ndarray] = None    # (21,3) hand world landmarks
    hand_score: float = 0.0
    hand_R_base_obs: Optional[np.ndarray] = None


@dataclass
class Calibration:
    calibrated: bool = False
    R_align: np.ndarray = field(default_factory=lambda: np.eye(3))     # body <- raw body (yaw/roll)
    human_arm_length: Dict[str, float] = field(default_factory=dict)
    pos_offset: Dict[str, np.ndarray] = field(default_factory=lambda: {s: np.zeros(3) for s in SIDES})
    R_human_neutral: Dict[str, np.ndarray] = field(default_factory=dict)
    t: float = 0.0


@dataclass
class SideOutput:
    wrist: np.ndarray            # 4x4 in the IK frame
    hand25: np.ndarray           # 25x3 in the Unitree hand frame
    hand21_body: Optional[np.ndarray]
    arm_conf: float
    hand_conf: float
    arm_state: str               # live | hold | easing | neutral
    hand_state: str
    pinch: float = 0.1           # m
    curl: float = 0.0


class _Channel:
    """Hold -> ease-to-neutral -> reacquire blending for one value (vector or rotation)."""

    def __init__(self, cfg: MapperConfig, is_rot: bool, neutral):
        self.cfg = cfg
        self.is_rot = is_rot
        self.neutral = neutral
        self.out = neutral.copy()
        self.t_good: Optional[float] = None
        self.held = neutral.copy()
        self.state = "neutral"
        self.blend_from = None
        self.t_reacq = None

    def _mix(self, a, b, w):
        if self.is_rot:
            return g.slerp(a, b, w)
        return a + (b - a) * w

    def update(self, t: float, live) -> np.ndarray:
        c = self.cfg
        if live is not None:
            if self.state != "live":           # (re)acquired: blend from wherever we are
                self.blend_from, self.t_reacq = self.out.copy(), t
            self.t_good = t
            self.state = "live"
            if self.blend_from is not None:
                w = min(1.0, (t - self.t_reacq) / max(c.reacquire_s, 1e-6))
                self.out = self._mix(self.blend_from, live, w * w * (3 - 2 * w))
                if w >= 1.0:
                    self.blend_from = None
            else:
                self.out = np.array(live, dtype=float)
            self.held = self.out.copy()
            return self.out
        self.blend_from = None
        if self.t_good is None:
            self.state = "neutral"
            self.out = self.neutral.copy()
            return self.out
        lost = t - self.t_good
        if lost < c.hold_s:
            self.state = "hold"
            self.out = self.held.copy()
        else:
            w = min(1.0, (lost - c.hold_s) / max(c.ease_s, 1e-6))
            self.state = "easing" if w < 1.0 else "neutral"
            self.out = self._mix(self.held, self.neutral, w * w * (3 - 2 * w))
        return self.out


class ArmHandMapper:
    def __init__(self, robot: RobotArmModel = RobotArmModel(), cfg: MapperConfig = MapperConfig(),
                 R_body_base: np.ndarray = g.R_BODY_BASE):
        self.robot = robot
        self.cfg = cfg
        self.R_body_base = R_body_base
        self.calib = Calibration()
        self.neutral_hand21 = {s: g.synthetic_open_hand21(s) for s in SIDES}
        self._last_raw: Dict[str, dict] = {s: {} for s in SIDES}
        self._reset_filters()
        self.ch_pos = {s: _Channel(cfg, False, robot.neutral_pos[s]) for s in SIDES}
        self.ch_rot = {s: _Channel(cfg, True, robot.neutral_rot[s]) for s in SIDES}
        self.ch_hand = {s: _Channel(cfg, False, self._neutral_hand25(s)) for s in SIDES}

    # ------------------------------------------------------------------------------
    def _reset_filters(self):
        c = self.cfg
        self.f_pos = {s: OneEuroFilter(np.array(c.pos_min_cutoff), c.pos_beta) for s in SIDES}
        self.f_rot = {s: OneEuroRotation(c.rot_min_cutoff, c.rot_beta) for s in SIDES}
        self.f_hand = {s: OneEuroFilter(c.hand_min_cutoff, c.hand_beta) for s in SIDES}

    def _neutral_hand25(self, side: str) -> np.ndarray:
        p21 = self.neutral_hand21[side]
        R_arm = g.palm_frame(p21)
        R_hand = g.hand_frame_from_arm_frame(R_arm, side)
        return g.hand_keypoints_in_hand_frame(g.mediapipe21_to_xr25(p21), R_hand)

    def to_body(self, v_obs: np.ndarray, R_base_obs: np.ndarray) -> np.ndarray:
        R = self.calib.R_align @ self.R_body_base @ R_base_obs
        return np.asarray(v_obs) @ R.T

    # ------------------------------------------------------------------------------
    def _raw_side(self, side: str, ob: SideObservation) -> dict:
        """Unfiltered body-frame quantities for one side."""
        raw = {}
        c = self.cfg
        if ob.arm_obs is not None and ob.arm_visibility >= c.min_arm_visibility:
            sh, el, wr = self.to_body(ob.arm_obs, ob.arm_R_base_obs)
            raw["wrist_rel"] = wr - sh
            raw["arm_len"] = float(np.linalg.norm(el - sh) + np.linalg.norm(wr - el))
            raw["shoulder"], raw["elbow"], raw["wrist"] = sh, el, wr
        if ob.hand21_obs is not None and ob.hand_score >= c.min_hand_score:
            h = self.to_body(ob.hand21_obs, ob.hand_R_base_obs)
            raw["hand21"] = h - h[0]
            raw["R_palm"] = g.palm_frame(h)
        return raw

    def update(self, t: float, obs: Dict[str, SideObservation]) -> Dict[str, SideOutput]:
        out = {}
        for side in SIDES:
            ob = obs.get(side, SideObservation())
            raw = self._raw_side(side, ob)
            self._last_raw[side] = raw
            out[side] = self._update_side(side, t, raw, ob)
        return out

    def _update_side(self, side, t, raw, ob) -> SideOutput:
        r, cal, c = self.robot, self.calib, self.cfg
        live_pos = live_rot = live_hand = None
        if "wrist_rel" in raw:
            v = self.f_pos[side](raw["wrist_rel"], t)
            L_h = cal.human_arm_length.get(side, c.human_arm_length)
            scale = r.arm_length / max(L_h, 0.2)
            live_pos = g.map_wrist_position(v, r.shoulder[side], scale, cal.pos_offset[side],
                                            c.max_reach_frac * r.arm_length, c.depth_gain)
        else:
            self.f_pos[side].reset()
        pinch, curl = 0.1, 0.0
        hand21 = None
        if "R_palm" in raw:
            R = self.f_rot[side](raw["R_palm"], t)
            if cal.calibrated and side in cal.R_human_neutral:
                R = (R @ cal.R_human_neutral[side].T) @ r.neutral_rot[side]
            live_rot = g.orthonormalize(R)
            hand21 = raw["hand21"]
            R_arm_raw = g.palm_frame(hand21)                 # hand shape uses the unfiltered palm frame
            R_hand = g.hand_frame_from_arm_frame(R_arm_raw, side)
            p25 = g.hand_keypoints_in_hand_frame(g.mediapipe21_to_xr25(hand21), R_hand)
            live_hand = self.f_hand[side](p25.reshape(-1), t).reshape(25, 3)
            pinch, curl = g.pinch_distance(hand21), g.finger_curl(hand21)
        else:
            self.f_rot[side].reset()
            self.f_hand[side].reset()
        p = self.ch_pos[side].update(t, live_pos)
        R = self.ch_rot[side].update(t, live_rot)
        h25 = self.ch_hand[side].update(t, live_hand)
        return SideOutput(g.se3(R, p), h25, hand21,
                          float(ob.arm_visibility if ob.arm_obs is not None else 0.0),
                          float(ob.hand_score if ob.hand21_obs is not None else 0.0),
                          self.ch_pos[side].state, self.ch_hand[side].state, pinch, curl)

    # ------------------------------------------------------------------------------
    def calibrate(self, t: float = 0.0, position: bool = True) -> Tuple[bool, str]:
        """Operator holds the neutral pose (upper arms relaxed, elbows ~90 deg, forearms
        forward, palms facing each other, thumbs up) - the robot's neutral target is that pose."""
        raws = self._last_raw
        missing = [s for s in SIDES if "wrist_rel" not in raws[s]]
        if missing:
            return False, f"calibration needs both arms visible (missing: {', '.join(missing)})"
        # 1) yaw/roll alignment: make the shoulder line (right -> left) the body +y axis,
        #    keeping body z as close to 'up' as possible.
        R_prev = self.calib.R_align
        sh_l = raws["left"]["shoulder"]
        sh_r = raws["right"]["shoulder"]
        y_raw = R_prev.T @ g.normalize(sh_l - sh_r)            # in un-aligned body coordinates
        z_ref = np.array([0.0, 0.0, 1.0])
        x_new = g.normalize(np.cross(y_raw, z_ref))
        z_new = np.cross(x_new, y_raw)
        R_align = np.column_stack([x_new, y_raw, z_new]).T     # rows = new axes in raw coords
        cal = Calibration(calibrated=True, R_align=R_align, t=t)
        for s in SIDES:
            raw = raws[s]
            rel = R_align @ (R_prev.T @ raw["wrist_rel"])
            cal.human_arm_length[s] = float(np.clip(raw["arm_len"], 0.35, 0.85))
            scale = self.robot.arm_length / cal.human_arm_length[s]
            if position:
                mapped = self.robot.shoulder[s] + scale * rel * np.array([self.cfg.depth_gain, 1, 1])
                cal.pos_offset[s] = self.robot.neutral_pos[s] - mapped
            if "R_palm" in raw:
                cal.R_human_neutral[s] = R_align @ R_prev.T @ raw["R_palm"]
        self.calib = cal
        self._reset_filters()
        hands = [s for s in SIDES if s in cal.R_human_neutral]
        return True, (f"calibrated: arm length L={cal.human_arm_length['left']:.2f} "
                      f"R={cal.human_arm_length['right']:.2f} m, palm zero for {hands or 'none'}")

    def reset_calibration(self):
        self.calib = Calibration()
        self._reset_filters()
