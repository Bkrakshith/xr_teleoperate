"""Webcam capture + MediaPipe (Tasks API, CPU) pose and hand tracking + OBSBOT auto-framing.

This process is the only one that opens the camera. A capture thread keeps only the newest
frame (no driver-queue latency); a processing thread runs pose + hands, gimbal compensation,
the landmark->robot mapper and the auto-framer, and publishes a snapshot that
`WebcamTeleWrapper.get_tele_data()` reads.
"""
from __future__ import annotations

import logging
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import geometry as g
from .mapper import ArmHandMapper, MapperConfig, RobotArmModel, SideObservation, SideOutput, SIDES

log = logging.getLogger("webcam_tracker")

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")

# MediaPipe pose indices
P_SH = {"left": 11, "right": 12}
P_EL = {"left": 13, "right": 14}
P_WR = {"left": 15, "right": 16}

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


def _import_obsbot():
    """teleop/obsbot lives next to this fork in Hospital_G1; allow OBSBOT_PY_PATH override."""
    candidates = [os.environ.get("OBSBOT_PY_PATH", "")]
    root = os.environ.get("HG1_ROOT")
    if root:
        candidates.append(os.path.join(root, "teleop", "obsbot"))
    candidates.append(os.path.abspath(os.path.join(HERE, "..", "..", "..", "obsbot")))
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "obsbot_ptz")) and c not in sys.path:
            sys.path.insert(0, c)
    import obsbot_ptz  # noqa: F401
    return obsbot_ptz


@dataclass
class TrackerConfig:
    source: str = "auto"                 # "auto" (find OBSBOT), /dev/videoN, index, video file or image
    width: int = 1280
    height: int = 720
    fps: int = 30
    mjpg: bool = True
    pose_model: str = "full"             # lite | full | heavy (file must exist in models/)
    num_hands: int = 2
    min_hand_detection: float = 0.5
    min_hand_presence: float = 0.5
    min_pose_detection: float = 0.5
    infer_width: int = 960               # frames are resized to this width before inference
    ray_correction: bool = True          # see geometry.py (MediaPipe estimates 3D on crops)
    frame_latency_s: float = 0.04        # capture timestamp -> exposure (for the gimbal estimate)
    framing: bool = True                 # OBSBOT auto-framing (needs PTZ controls)
    allow_simulated_ptz: bool = True     # no PTZ controls -> framing runs against a model only
    idle_stride: int = 3                 # when idle (robot not in MANIP) process every Nth frame
    loop_source: bool = True             # loop video files / repeat still images
    mirrored_input: bool = False         # True if the frames are already horizontally flipped
    recenter_on_exit: bool = True        # put the OBSBOT back to pan/tilt/zoom 0 when stopping


@dataclass
class TrackSnapshot:
    t: float
    t_capture: float
    outputs: Dict[str, SideOutput]
    hands_visible: List[bool]
    arms_visible: List[bool]
    fps: float
    latency_ms: float
    framing: str
    framing_detail: str
    ptz: Tuple[float, float, float]
    motion_data_ready: bool


class _Capture(threading.Thread):
    """Grabs frames continuously and keeps only the newest (camera, video file or still image)."""

    def __init__(self, cfg: TrackerConfig):
        super().__init__(daemon=True, name="webcam-capture")
        self.cfg = cfg
        self.lock = threading.Condition()
        self.frame = None
        self.t_frame = 0.0
        self.seq = 0
        self.running = True
        self.still = None
        self.is_camera = False
        self.error = None
        src = cfg.source
        if isinstance(src, str) and os.path.isfile(src) and src.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            self.still = cv2.imread(src)
            if self.still is None:
                raise RuntimeError(f"cannot read image {src}")
            self.cap = None
        else:
            if isinstance(src, str) and src.isdigit():
                src = int(src)
            self.is_camera = isinstance(src, int) or (isinstance(src, str) and src.startswith("/dev/video"))
            self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2) if self.is_camera else cv2.VideoCapture(src)
            if not self.cap.isOpened():
                raise RuntimeError(f"cannot open video source {src}")
            if self.is_camera:
                if cfg.mjpg:
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
                self.cap.set(cv2.CAP_PROP_FPS, cfg.fps)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.src_fps = self.cap.get(cv2.CAP_PROP_FPS) or cfg.fps

    def run(self):
        period = 1.0 / max(1, self.cfg.fps)
        while self.running:
            if self.still is not None:
                time.sleep(period)
                f = self.still.copy()
            else:
                ok, f = self.cap.read()
                if not ok:
                    if not self.is_camera and self.cfg.loop_source:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    self.error = "capture read failed"
                    time.sleep(0.05)
                    continue
                if not self.is_camera:
                    time.sleep(1.0 / max(1.0, self.src_fps))
            with self.lock:
                self.frame, self.t_frame = f, time.monotonic()
                self.seq += 1
                self.lock.notify_all()

    def latest(self, last_seq: int, timeout: float = 0.5):
        with self.lock:
            if self.seq == last_seq:
                self.lock.wait(timeout)
            return self.frame, self.t_frame, self.seq

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()


class WebcamTracker:
    def __init__(self, cfg: TrackerConfig = TrackerConfig(), robot: RobotArmModel = RobotArmModel(),
                 mapper_cfg: MapperConfig = MapperConfig(), framing_cfg=None):
        self.cfg = cfg
        if cfg.source == "auto":
            cfg.source = self._find_obsbot() or "/dev/video0"
        self.messages: List[Tuple[float, str]] = []
        self.capture = _Capture(cfg)
        self.mapper = ArmHandMapper(robot, mapper_cfg)
        self._init_mediapipe()
        self._init_ptz(framing_cfg)
        self.lock = threading.Lock()
        self.snapshot: Optional[TrackSnapshot] = None
        self.annotated: Optional[np.ndarray] = None
        self.annot_seq = 0
        self.idle = False
        self.mode_text = "mode: ?"
        self._calib_req = False
        self._calib_pos = True
        self.preview_fn = None           # callable(img) -> None, draws the IK preview panel
        self._fps = 0.0
        self._lat = 0.0
        self.motion_data_ready = False
        self.running = False
        self.thread = threading.Thread(target=self._loop, daemon=True, name="webcam-track")

    # ------------------------------------------------------------------------------
    @staticmethod
    def _find_obsbot() -> Optional[str]:
        try:
            ob = _import_obsbot()
            from obsbot_ptz.v4l2 import find_device
            return find_device()
        except Exception:  # noqa: BLE001
            return None

    def _init_mediapipe(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self.mp = mp
        pose_path = os.path.join(MODEL_DIR, f"pose_landmarker_{self.cfg.pose_model}.task")
        hand_path = os.path.join(MODEL_DIR, "hand_landmarker.task")
        for p in (pose_path, hand_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"{p} missing - run teleop/webcam/fetch_models.sh")
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=pose_path, delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=self.cfg.min_pose_detection,
            min_pose_presence_confidence=0.5, min_tracking_confidence=0.5))
        self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=hand_path, delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO, num_hands=self.cfg.num_hands,
            min_hand_detection_confidence=self.cfg.min_hand_detection,
            min_hand_presence_confidence=self.cfg.min_hand_presence, min_tracking_confidence=0.5))
        self._ts_ms = 0

    def _init_ptz(self, framing_cfg):
        self.ptz = None
        self.framer = None
        self.ptz_note = ""
        try:
            ob = _import_obsbot()
        except Exception as e:  # noqa: BLE001
            self.ptz_note = f"obsbot_ptz unavailable: {e}"
            return
        dev = self.cfg.source if (isinstance(self.cfg.source, str) and self.cfg.source.startswith("/dev/video")) else None
        if self.capture.is_camera and dev is not None:
            try:
                self.ptz = ob.ObsbotPTZ(dev)
            except Exception as e:  # noqa: BLE001
                self.ptz_note = f"no PTZ on {dev}: {e}"
        if self.ptz is None and self.cfg.allow_simulated_ptz:
            self.ptz = ob.SimulatedPTZ()
            self.ptz_note = (self.ptz_note + "; " if self.ptz_note else "") + "simulated PTZ"
        if self.ptz is not None:
            fc = framing_cfg or ob.FramingConfig()
            fc.enabled = self.cfg.framing and fc.enabled
            self.framer = ob.AutoFramer(self.ptz, fc)
            if isinstance(self.ptz, ob.ObsbotPTZ):
                ok, msg = self.ptz.disable_ai_tracking()
                if not ok:
                    self.note(msg)

    # ------------------------------------------------------------------------------
    def start(self):
        self.running = True
        self.capture.start()
        self.thread.start()
        return self

    def stop(self):
        self.running = False
        self.capture.stop()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        try:
            if self.ptz is not None and self.cfg.recenter_on_exit and self.framer is not None:
                self.ptz.center(0.0)
            if self.ptz is not None and hasattr(self.ptz, "close"):
                self.ptz.close()
            self.pose.close()
            self.hands.close()
        except Exception:  # noqa: BLE001
            pass

    def note(self, msg: str):
        log.warning(msg) if ("fail" in msg or "error" in msg) else log.info(msg)
        print(f"[webcam] {msg}", flush=True)
        self.messages.append((time.monotonic(), msg))
        self.messages = self.messages[-6:]

    def request_calibration(self, position: bool = True):
        self._calib_req, self._calib_pos = True, position

    def set_framing(self, on: bool):
        if self.framer is not None:
            self.framer.set_enabled(on)
            self.note(f"auto-framing {'on' if on else 'off'}")

    def recenter_camera(self):
        if self.framer is not None:
            self.framer.recenter()
            self.note("camera re-centred")

    def latest(self) -> Optional[TrackSnapshot]:
        with self.lock:
            return self.snapshot

    def latest_annotated(self) -> Tuple[Optional[np.ndarray], int]:
        with self.lock:
            return self.annotated, self.annot_seq

    # ------------------------------------------------------------------------------
    def _intrinsics(self, w: int, h: int, zoom: float):
        f = self.ptz.optics.focal_px(w, zoom) if self.ptz is not None else 975.0 * w / 1280.0
        return (f, f, w / 2.0, h / 2.0)

    def _loop(self):
        seq = 0
        n = 0
        t_prev = None
        while self.running:
            frame, t_cap, new_seq = self.capture.latest(seq)
            if frame is None or new_seq == seq:
                continue
            seq = new_seq
            n += 1
            if self.idle and (n % max(1, self.cfg.idle_stride)) != 0:
                continue
            try:
                self._process(frame, t_cap)
            except Exception as e:  # noqa: BLE001
                log.exception("tracker step failed")
                self.note(f"tracker error: {e}")
                time.sleep(0.05)
            now = time.monotonic()
            if t_prev is not None:
                inst = 1.0 / max(1e-3, now - t_prev)
                self._fps = inst if self._fps == 0 else 0.9 * self._fps + 0.1 * inst
            t_prev = now

    def _process(self, frame: np.ndarray, t_cap: float):
        cfg = self.cfg
        H, W = frame.shape[:2]
        scale = cfg.infer_width / W if W > cfg.infer_width else 1.0
        small = cv2.resize(frame, (int(W * scale), int(H * scale))) if scale < 1.0 else frame
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        self._ts_ms = max(self._ts_ms + 1, int(t_cap * 1000))
        pose_res = self.pose.detect_for_video(mp_img, self._ts_ms)
        hand_res = self.hands.detect_for_video(mp_img, self._ts_ms)

        t_exp = t_cap - cfg.frame_latency_s
        yaw, pitch, zoom = self.ptz.estimate(t_exp) if self.ptz is not None else (0.0, 0.0, 0.0)
        intr = self._intrinsics(W, H, zoom)

        obs = {s: SideObservation() for s in SIDES}
        img_pose = None
        pose_px = {}
        if pose_res.pose_landmarks:
            lm = pose_res.pose_landmarks[0]
            wl = pose_res.pose_world_landmarks[0]
            img_pose = np.array([[p.x, p.y, (p.visibility if p.visibility is not None else 1.0)] for p in lm])
            world = np.array([[p.x, p.y, p.z] for p in wl])
            # operator faces the camera: their left shoulder is on the image right. Assign sides
            # by image position so a mirrored label can never swap the arms.
            sides = dict(left="left", right="right")
            if img_pose[P_SH["left"], 0] < img_pose[P_SH["right"], 0]:
                sides = dict(left="right", right="left")
            mid = 0.5 * (img_pose[11, :2] + img_pose[12, :2]) * [W, H]
            R_bo = g.R_base_obs(yaw, pitch, tuple(mid) if cfg.ray_correction else None, intr)
            for s in SIDES:
                src = sides[s]
                ids = [P_SH[src], P_EL[src], P_WR[src]]
                vis = float(np.min(img_pose[ids, 2]))
                obs[s].arm_obs = world[ids]
                obs[s].arm_visibility = vis
                obs[s].arm_R_base_obs = R_bo
                pose_px[s] = img_pose[ids, :2] * [W, H]
        hands_px = {}
        if hand_res.hand_landmarks:
            cands = []
            for i, lms in enumerate(hand_res.hand_landmarks):
                px = np.array([[p.x * W, p.y * H] for p in lms])
                world = np.array([[p.x, p.y, p.z] for p in hand_res.hand_world_landmarks[i]])
                cat = hand_res.handedness[i][0]
                cands.append((px, world, float(cat.score), cat.category_name))
            assigned = self._assign_hands(cands, pose_px, W)
            for s, (px, world, score, _) in assigned.items():
                R_bo = g.R_base_obs(yaw, pitch, tuple(px.mean(axis=0)) if cfg.ray_correction else None, intr)
                obs[s].hand21_obs = world
                obs[s].hand_score = score
                obs[s].hand_R_base_obs = R_bo
                hands_px[s] = px

        t = time.monotonic()
        if self._calib_req:
            self._calib_req = False
            self.mapper.update(t, obs)          # refresh raw values with this frame
            ok, msg = self.mapper.calibrate(t, self._calib_pos)
            self.note(msg if ok else "calibration failed: " + msg)
        outputs = self.mapper.update(t, obs)
        hands_visible = [obs[s].hand21_obs is not None and obs[s].hand_score >= self.mapper.cfg.min_hand_score for s in SIDES]
        arms_visible = [obs[s].arm_obs is not None and obs[s].arm_visibility >= self.mapper.cfg.min_arm_visibility for s in SIDES]
        if any(hands_visible) or any(arms_visible):
            self.motion_data_ready = True

        # ---- auto-framing: hands + forearms ---------------------------------------
        framing, detail, fo = "off", "off", None
        if self.framer is not None:
            boxes, fingers = [], []
            for s, px in hands_px.items():
                n = px / [W, H]
                boxes.append((n[:, 0].min() - 0.02, n[:, 1].min() - 0.02, n[:, 0].max() + 0.02, n[:, 1].max() + 0.02))
                fingers.extend([tuple(p) for p in n])
            for s, px in pose_px.items():
                if obs[s].arm_visibility >= 0.5:
                    el, wr = px[1] / [W, H], px[2] / [W, H]
                    boxes.append((min(el[0], wr[0]), min(el[1], wr[1]), max(el[0], wr[0]), max(el[1], wr[1])))
            fo = self.framer.update(t, (W, H), boxes, fingers)
            framing, detail = fo.contract_state, fo.state
        ptz_now = self.ptz.commanded() if self.ptz is not None else (0.0, 0.0, 0.0)
        latency = (time.monotonic() - t_cap) * 1000.0
        self._lat = latency if self._lat == 0 else 0.9 * self._lat + 0.1 * latency
        snap = TrackSnapshot(t, t_cap, outputs, hands_visible, arms_visible, self._fps, self._lat,
                             framing, detail, ptz_now, self.motion_data_ready)
        annotated = self._draw(frame, img_pose, hands_px, fo, snap)
        with self.lock:
            self.snapshot = snap
            self.annotated = annotated
            self.annot_seq += 1

    def _assign_hands(self, cands, pose_px, W):
        """Match detected hands to the operator's left/right arm: nearest pose wrist first,
        then image position (operator's left hand is on the image right), then label."""
        out = {}
        if not cands:
            return out
        if pose_px:
            pairs = []
            for i, c in enumerate(cands):
                for s, px in pose_px.items():
                    pairs.append((float(np.linalg.norm(c[0][0] - px[2])), i, s))
            used_c, used_s = set(), set()
            for d, i, s in sorted(pairs):
                if i in used_c or s in used_s or d > 0.25 * W:
                    continue
                out[s] = cands[i]
                used_c.add(i)
                used_s.add(s)
            rest = [c for i, c in enumerate(cands) if i not in used_c]
        else:
            rest = list(cands)
        # No pose match: MediaPipe's handedness label assumes a mirrored (selfie) image, so for the
        # raw webcam image the label is flipped. Fall back to image position (operator's left hand
        # appears on the image right) if the labels collide.
        for c in sorted(rest, key=lambda c: -c[2]):
            label = c[3].lower()
            s = label if self.cfg.mirrored_input else {"left": "right", "right": "left"}.get(label, label)
            if s in ("left", "right") and s not in out:
                out[s] = c
        for c in sorted(rest, key=lambda c: -c[0][0, 0]):
            if any(c is v for v in out.values()):
                continue
            for s in ("left", "right"):
                if s not in out:
                    out[s] = c
                    break
        return out

    # ------------------------------------------------------------------------------
    def _draw(self, frame, img_pose, hands_px, fo, snap: TrackSnapshot):
        H, W = frame.shape[:2]
        k = 960.0 / W if W > 960 else 1.0
        img = cv2.resize(frame, (int(W * k), int(H * k))) if k < 1.0 else frame.copy()
        h, w = img.shape[:2]
        col = {"left": (255, 160, 0), "right": (0, 140, 255)}
        if img_pose is not None:
            for a, b in ((11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)):
                if img_pose[a, 2] > 0.3 and img_pose[b, 2] > 0.3:
                    pa = (int(img_pose[a, 0] * w), int(img_pose[a, 1] * h))
                    pb = (int(img_pose[b, 0] * w), int(img_pose[b, 1] * h))
                    cv2.line(img, pa, pb, (200, 200, 200), 2)
        for s, px in hands_px.items():
            p = (px * k).astype(int)
            for a, b in HAND_EDGES:
                cv2.line(img, tuple(p[a]), tuple(p[b]), col[s], 2)
            for q in p:
                cv2.circle(img, tuple(q), 3, (255, 255, 255), -1)
            cv2.putText(img, s.upper(), tuple(p[0] + [5, 15]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col[s], 2)
        if fo is not None and fo.union_box is not None:
            b = fo.union_box
            cv2.rectangle(img, (int(b[0] * w), int(b[1] * h)), (int(b[2] * w), int(b[3] * h)), (0, 255, 0), 2)
        cv2.drawMarker(img, (w // 2, h // 2), (0, 255, 0), cv2.MARKER_CROSS, 20, 1)
        # status bar
        bar = [f"{self.mode_text}",
               f"fps {snap.fps:4.1f}  lat {snap.latency_ms:4.0f}ms",
               f"framing {snap.framing_detail}  yaw(R+) {snap.ptz[0]:+.0f} pitch(up+) {snap.ptz[1]:+.0f} zoom {snap.ptz[2]:.0f}",
               "calibrated" if self.mapper.calib.calibrated else "NOT calibrated (hold neutral pose, press c)"]
        for s_i, s in enumerate(SIDES):
            o = snap.outputs[s]
            bar.append(f"{s[0].upper()}: arm {o.arm_state:7s} {o.arm_conf:.2f}  hand {o.hand_state:7s} {o.hand_conf:.2f}")
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 22 * len(bar) + 8), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
        for i, line in enumerate(bar):
            cv2.putText(img, line, (8, 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        now = time.monotonic()
        msgs = [m for t0, m in self.messages if now - t0 < 6.0]
        for i, m in enumerate(msgs[-3:]):
            cv2.putText(img, m[:110], (8, h - 12 - 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        if self.preview_fn is not None:
            try:
                self.preview_fn(img)
            except Exception as e:  # noqa: BLE001
                log.debug(f"preview failed: {e}")
        return img
