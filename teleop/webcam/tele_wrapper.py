"""Drop-in replacement for televuer.TeleVuerWrapper driven by a desk webcam (OBSBOT Tiny 3).

The main loop (teleop_hand_and_arm.py) only needs `get_tele_data()`, `render_to_xr()` and
`close()`. `get_tele_data()` returns a TeleData with the same fields and conventions as
televuer's hand-tracking path:
  head_pose        4x4, fixed (seated operator, head not teleoperated)
  left/right_wrist_pose  4x4 in the IK (pelvis) frame, Unitree arm wrist convention
  left/right_hand_pos    25x3 OpenXR keypoints in the Unitree hand frame (wrist at origin)
  *_pinchValue / *_squeezeValue, motion_data_ready
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .mapper import MapperConfig, RobotArmModel, SIDES
from .tracker import TrackerConfig, WebcamTracker

log = logging.getLogger("webcam_tele")

try:  # identical dataclass when televuer is installed (native XR setup)
    from televuer import TeleData  # type: ignore
except Exception:  # noqa: BLE001
    @dataclass
    class TeleData:  # mirror of televuer.tv_wrapper.TeleData
        head_pose: np.ndarray
        left_wrist_pose: np.ndarray
        right_wrist_pose: np.ndarray
        left_hand_pos: np.ndarray = None
        right_hand_pos: np.ndarray = None
        left_hand_rot: np.ndarray = None
        right_hand_rot: np.ndarray = None
        left_hand_pinch: bool = False
        left_hand_pinchValue: float = 10.0
        left_hand_squeeze: bool = False
        left_hand_squeezeValue: float = 0.0
        right_hand_pinch: bool = False
        right_hand_pinchValue: float = 10.0
        right_hand_squeeze: bool = False
        right_hand_squeezeValue: float = 0.0
        motion_data_ready: bool = False
        left_ctrl_trigger: bool = False
        left_ctrl_triggerValue: float = 10.0
        left_ctrl_squeeze: bool = False
        left_ctrl_squeezeValue: float = 0.0
        left_ctrl_aButton: bool = False
        left_ctrl_bButton: bool = False
        left_ctrl_thumbstick: bool = False
        left_ctrl_thumbstickValue: np.ndarray = field(default_factory=lambda: np.zeros(2))
        right_ctrl_trigger: bool = False
        right_ctrl_triggerValue: float = 10.0
        right_ctrl_squeeze: bool = False
        right_ctrl_squeezeValue: float = 0.0
        right_ctrl_aButton: bool = False
        right_ctrl_bButton: bool = False
        right_ctrl_thumbstick: bool = False
        right_ctrl_thumbstickValue: np.ndarray = field(default_factory=lambda: np.zeros(2))

# Seated operator: fixed head pose in the robot convention (x fwd, y left, z up). televuer's
# CONST_HEAD_POSE converted to the robot basis; only used for logging/recording.
CONST_HEAD_POSE_ROBOT = np.array([[1, 0, 0, 0.2], [0, 1, 0, 0.0], [0, 0, 1, 1.5], [0, 0, 0, 1.0]])


def robot_arm_model_from_ik(arm_ik) -> RobotArmModel:
    """Shoulder positions, arm length and neutral wrist poses from the IK's reduced model
    (neutral = all 14 arm joints at 0: G1 elbows at 90 deg, forearms forward)."""
    import pinocchio as pin
    m = arm_ik.reduced_robot.model
    d = m.createData()
    q0 = np.zeros(m.nq)
    pin.framesForwardKinematics(m, d, q0)
    rm = RobotArmModel()
    for side, ee in (("left", "L_ee"), ("right", "R_ee")):
        sh = d.oMi[m.getJointId(f"{side}_shoulder_pitch_joint")].translation.copy()
        el = d.oMi[m.getJointId(f"{side}_elbow_joint")].translation.copy()
        T = d.oMf[m.getFrameId(ee)]
        rm.shoulder[side] = sh
        rm.neutral_pos[side] = T.translation.copy()
        rm.neutral_rot[side] = T.rotation.copy()
        if side == "left":
            rm.arm_length = float(np.linalg.norm(el - sh) + np.linalg.norm(T.translation - el))
    return rm


class IKPreview:
    """Draws the solved arm configuration (front + top view stick figure) into the overlay."""

    def __init__(self, arm_ik):
        import pinocchio as pin
        self.pin = pin
        self.model = arm_ik.reduced_robot.model
        self.data = self.model.createData()
        self.q = np.zeros(self.model.nq)
        self.targets = None
        self.hand_values_fn = None      # -> (left6, right6) contract order, 0..1
        self.chain = {s: [self.model.getJointId(f"{s}_{j}_joint") for j in
                          ("shoulder_pitch", "elbow", "wrist_roll")] for s in SIDES}
        self.ee = {"left": self.model.getFrameId("L_ee"), "right": self.model.getFrameId("R_ee")}

    def set(self, q, left_T=None, right_T=None):
        self.q = np.asarray(q, dtype=float).copy()
        self.targets = (left_T, right_T)

    def __call__(self, img):
        import cv2
        pin = self.pin
        pin.framesForwardKinematics(self.model, self.data, self.q)
        h, w = img.shape[:2]
        pw, ph = 220, 170
        x0, y0 = w - 2 * pw - 12, h - ph - 8
        cv2.rectangle(img, (x0 - 4, y0 - 4), (w - 4, h - 4), (30, 30, 30), -1)
        # viewer faces the robot, like the operator faces the camera: robot left = image right
        views = (("front", lambda p: (p[1], -p[2])), ("top", lambda p: (p[1], p[0])))
        for k, (name, proj) in enumerate(views):
            ox, oy = x0 + k * pw + pw // 2, y0 + (ph // 2 if name == "front" else 30)
            s = 260.0

            def P(p):
                u, v = proj(p)
                return int(ox + s * u), int(oy + s * (v + (0.3 if name == "front" else 0.0)))
            cv2.putText(img, f"IK {name}", (x0 + k * pw + 4, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            for side, col in (("left", (255, 160, 0)), ("right", (0, 140, 255))):
                pts = [self.data.oMi[j].translation for j in self.chain[side]] + [self.data.oMf[self.ee[side]].translation]
                for a, b in zip(pts[:-1], pts[1:]):
                    cv2.line(img, P(a), P(b), col, 3)
                if self.targets is not None:
                    T = self.targets[0 if side == "left" else 1]
                    if T is not None:
                        cv2.circle(img, P(T[:3, 3]), 5, (0, 255, 0), 1)
        if self.hand_values_fn is not None:
            vals = self.hand_values_fn()
            labels = ("T", "Ta", "I", "M", "R", "P")
            bx, by = x0, y0 - 72
            cv2.rectangle(img, (bx - 4, by - 18), (w - 4, y0 - 6), (30, 30, 30), -1)
            for k, (side, col) in enumerate((("left", (255, 160, 0)), ("right", (0, 140, 255)))):
                cv2.putText(img, f"{side} hand 0=open 1=closed", (bx + k * pw, by - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                for i, v in enumerate(vals[k]):
                    x = bx + k * pw + 8 + i * 34
                    cv2.rectangle(img, (x, y0 - 12), (x + 20, y0 - 12 - int(40 * float(v))), col, -1)
                    cv2.putText(img, labels[i], (x + 2, y0 - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)


class WebcamTeleWrapper:
    def __init__(self, tracker_cfg: TrackerConfig = TrackerConfig(), arm_ik=None,
                 mapper_cfg: MapperConfig = MapperConfig(), framing_cfg=None, show_window: bool = True,
                 window_name: str = "G1 webcam teleop"):
        robot = robot_arm_model_from_ik(arm_ik) if arm_ik is not None else RobotArmModel()
        self.robot = robot
        self.tracker = WebcamTracker(tracker_cfg, robot, mapper_cfg, framing_cfg)
        self.preview = IKPreview(arm_ik) if arm_ik is not None else None
        self.tracker.preview_fn = self.preview
        self.show_window = show_window and bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        self.window_name = window_name
        self._shown_seq = -1
        self.tracker.start()
        log.info(f"webcam teleop: source={tracker_cfg.source} ptz={self.tracker.ptz_note or 'OBSBOT UVC'}")

    # ---- televuer interface ------------------------------------------------------
    def get_tele_data(self) -> TeleData:
        snap = self.tracker.latest()
        if snap is None:
            rm = self.robot
            Ts = {}
            for s in SIDES:
                T = np.eye(4)
                T[:3, :3] = rm.neutral_rot[s]
                T[:3, 3] = rm.neutral_pos[s]
                Ts[s] = T
            return TeleData(head_pose=CONST_HEAD_POSE_ROBOT.copy(), left_wrist_pose=Ts["left"],
                            right_wrist_pose=Ts["right"], left_hand_pos=np.zeros((25, 3)),
                            right_hand_pos=np.zeros((25, 3)), motion_data_ready=False)
        L, R = snap.outputs["left"], snap.outputs["right"]
        return TeleData(
            head_pose=CONST_HEAD_POSE_ROBOT.copy(),
            left_wrist_pose=L.wrist.copy(), right_wrist_pose=R.wrist.copy(),
            left_hand_pos=L.hand25.copy(), right_hand_pos=R.hand25.copy(),
            left_hand_pinch=L.pinch < 0.03, left_hand_pinchValue=L.pinch * 100.0,
            left_hand_squeeze=L.curl > 0.7, left_hand_squeezeValue=L.curl,
            right_hand_pinch=R.pinch < 0.03, right_hand_pinchValue=R.pinch * 100.0,
            right_hand_squeeze=R.curl > 0.7, right_hand_squeezeValue=R.curl,
            motion_data_ready=snap.motion_data_ready,
        )

    def render_to_xr(self, img):
        pass

    def close(self):
        self.tracker.stop()
        if self.show_window:
            try:
                import cv2
                cv2.destroyWindow(self.window_name)
            except Exception:  # noqa: BLE001
                pass

    # ---- webcam extras -------------------------------------------------------------
    def set_ik_preview(self, sol_q, left_T=None, right_T=None):
        if self.preview is not None:
            self.preview.set(sol_q, left_T, right_T)

    def calibrate(self):
        self.tracker.request_calibration()

    def set_mode_text(self, text: str, idle: bool = False):
        self.tracker.mode_text = text
        self.tracker.idle = idle

    def status(self) -> dict:
        """Contract JSON for /g1/teleop/tracking_status (+ extra keys)."""
        snap = self.tracker.latest()
        if snap is None:
            return {"hands_visible": [False, False], "arms_visible": [False, False], "fps": 0.0,
                    "framing": "off", "calibrated": False}
        return {
            "hands_visible": list(map(bool, snap.hands_visible)),
            "arms_visible": list(map(bool, snap.arms_visible)),
            "fps": round(float(snap.fps), 1),
            "framing": snap.framing,
            "framing_detail": snap.framing_detail,
            "calibrated": bool(self.tracker.mapper.calib.calibrated),
            "latency_ms": round(float(snap.latency_ms), 1),
            "confidence": {s: {"arm": round(snap.outputs[s].arm_conf, 2), "hand": round(snap.outputs[s].hand_conf, 2),
                               "arm_state": snap.outputs[s].arm_state, "hand_state": snap.outputs[s].hand_state}
                           for s in SIDES},
            "ptz": {"yaw_right_deg": snap.ptz[0], "pitch_up_deg": snap.ptz[1], "zoom": snap.ptz[2]},
            "message": self.tracker.messages[-1][1] if self.tracker.messages else "",
        }

    def annotated_frame(self):
        return self.tracker.latest_annotated()

    def handle_key(self, key: str) -> Optional[str]:
        """Webcam-specific keys. Returns the key if the main loop should handle it."""
        if key == "c":
            self.calibrate()
        elif key == "f":
            fr = self.tracker.framer
            self.tracker.set_framing(not (fr is not None and fr.enabled))
        elif key == "h":
            self.tracker.recenter_camera()
        elif key == "u":
            self.tracker.mapper.reset_calibration()
            self.tracker.note("calibration cleared")
        else:
            return key
        return None

    def spin_gui(self, wait_ms: int = 1) -> Optional[str]:
        """Show the annotated frame (main thread only). Returns a key the main loop should act on."""
        if not self.show_window:
            return None
        import cv2
        img, seq = self.annotated_frame()
        if img is not None and seq != self._shown_seq:
            cv2.imshow(self.window_name, img)
            self._shown_seq = seq
        k = cv2.waitKey(wait_ms) & 0xFF
        if k == 255:
            return None
        return self.handle_key(chr(k).lower() if k < 128 else "")
