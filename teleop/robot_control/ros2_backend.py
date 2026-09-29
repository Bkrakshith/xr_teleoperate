"""ROS 2 output backend (Hospital_G1 contract, docs/interfaces.md) and a no-robot preview
backend, both drop-in replacements for the DDS arm / BrainCo hand controllers used by
teleop_hand_and_arm.py.

Published (ROS 2 backend):
  /g1/teleop/upper_body_target_raw     sensor_msgs/JointState  waist(3, zero) + 14 arm joints
  /g1/teleop/hand/{left,right}/cmd_raw sensor_msgs/JointState  6 values 0..1, contract order
                                       [thumb, thumb_aux, index, middle, ring, pinky]
  /g1/teleop/camera/annotated/compressed sensor_msgs/CompressedImage (jpeg)
  /g1/teleop/tracking_status           std_msgs/String (JSON)
Subscribed:
  /g1/mode (g1_interfaces/Mode, transient local)  -> shown in the overlay; optional idling
  /g1/teleop/key (std_msgs/String)                -> 'c' calibrates the webcam tracker
"""
from __future__ import annotations

import json
import threading
import time
from typing import Callable, Optional

import numpy as np

import logging_mp
logger_mp = logging_mp.getLogger(__name__)

from teleop.robot_control.brainco_retarget import CONTRACT_FROM_RETARGET, brainco_retarget_dual

WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
UPPER_BODY_JOINTS = WAIST_JOINTS + ARM_JOINTS
HAND_SLOTS = ["thumb", "thumb_aux", "index", "middle", "ring", "pinky"]


def hand_joint_names(side: str):
    """Names on the cmd_raw JointState, contract slot order (= /joint_states hand names)."""
    return [f"{side}_thumb_proximal_joint", f"{side}_thumb_metacarpal_joint", f"{side}_index_proximal_joint",
            f"{side}_middle_proximal_joint", f"{side}_ring_proximal_joint", f"{side}_pinky_proximal_joint"]


MODE_NAMES = {0: "DAMP", 1: "STAND", 2: "LOCO", 3: "MANIP", 4: "NAV"}


class Ros2TeleopNode:
    """Owns rclpy, the node, publishers/subscribers and a spinning executor thread."""

    def __init__(self, node_name: str = "g1_webcam_teleop", use_sim_time: bool = False,
                 image_hz: float = 15.0, status_hz: float = 10.0, jpeg_quality: int = 70):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.parameter import Parameter
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, HistoryPolicy
        from sensor_msgs.msg import CompressedImage, JointState
        from std_msgs.msg import String
        self.rclpy = rclpy
        self.JointState, self.CompressedImage, self.String = JointState, CompressedImage, String
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node(node_name)
        if use_sim_time:
            self.node.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        n = self.node
        self.pub_upper = n.create_publisher(JointState, "/g1/teleop/upper_body_target_raw", 10)
        self.pub_hand = {s: n.create_publisher(JointState, f"/g1/teleop/hand/{s}/cmd_raw", 10) for s in ("left", "right")}
        self.pub_img = n.create_publisher(CompressedImage, "/g1/teleop/camera/annotated/compressed", 2)
        self.pub_status = n.create_publisher(String, "/g1/teleop/tracking_status", 10)
        self.mode: Optional[int] = None
        self.mode_source = ""
        self.on_mode: Optional[Callable[[int], None]] = None
        self.on_key: Optional[Callable[[str], None]] = None
        try:
            from g1_interfaces.msg import Mode
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST)
            n.create_subscription(Mode, "/g1/mode", self._mode_cb, qos)
            self.has_mode = True
        except ImportError:
            logger_mp.warning("[ros2] g1_interfaces not found (build ros2_ws); /g1/mode not shown")
            self.has_mode = False
        n.create_subscription(String, "/g1/teleop/key", self._key_cb, 10)
        self.image_period = 1.0 / image_hz
        self.status_period = 1.0 / status_hz
        self.jpeg_quality = jpeg_quality
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(n)
        self._spin = threading.Thread(target=self._spin_loop, daemon=True, name="ros2-spin")
        self._spin.start()
        self._webcam = None
        self._aux = None
        logger_mp.info(f"[ros2] node {node_name} up (use_sim_time={use_sim_time})")

    def _spin_loop(self):
        while self.rclpy.ok():
            try:
                self.executor.spin_once(timeout_sec=0.1)
            except Exception:  # noqa: BLE001  (shutdown race)
                break

    def _mode_cb(self, msg):
        self.mode = int(msg.mode)
        self.mode_source = msg.source
        if self.on_mode:
            self.on_mode(self.mode)

    def _key_cb(self, msg):
        if self.on_key and msg.data:
            self.on_key(msg.data.strip().lower())

    def stamp(self):
        return self.node.get_clock().now().to_msg()

    def mode_name(self) -> str:
        return MODE_NAMES.get(self.mode, "?") if self.mode is not None else "unknown"

    # ---- publishing -----------------------------------------------------------------
    def publish_upper_body(self, arm_q_by_name: dict):
        if not self.rclpy.ok():
            return
        msg = self.JointState()
        msg.header.stamp = self.stamp()
        msg.name = list(UPPER_BODY_JOINTS)
        msg.position = [0.0, 0.0, 0.0] + [float(arm_q_by_name[j]) for j in ARM_JOINTS]
        self.pub_upper.publish(msg)

    def publish_hand(self, side: str, values_contract):
        if not self.rclpy.ok():
            return
        msg = self.JointState()
        msg.header.stamp = self.stamp()
        msg.name = hand_joint_names(side)
        msg.position = [float(np.clip(v, 0.0, 1.0)) for v in values_contract]
        self.pub_hand[side].publish(msg)

    def attach_webcam(self, wrapper):
        """Start the image + status publisher thread for a WebcamTeleWrapper."""
        self._webcam = wrapper
        self._aux = threading.Thread(target=self._aux_loop, daemon=True, name="ros2-webcam-pub")
        self._aux.start()

    def _aux_loop(self):
        import cv2
        t_img = t_status = 0.0
        last_seq = -1
        while self.rclpy.ok() and self._webcam is not None:
            now = time.monotonic()
            if now - t_status >= self.status_period:
                t_status = now
                st = self._webcam.status()
                st["mode"] = self.mode_name()
                self.pub_status.publish(self.String(data=json.dumps(st)))
            if now - t_img >= self.image_period:
                img, seq = self._webcam.annotated_frame()
                if img is not None and seq != last_seq:
                    t_img, last_seq = now, seq
                    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
                    if ok:
                        msg = self.CompressedImage()
                        msg.header.stamp = self.stamp()
                        msg.header.frame_id = "teleop_camera"
                        msg.format = "jpeg"
                        msg.data = buf.tobytes()
                        self.pub_img.publish(msg)
            time.sleep(0.01)

    def shutdown(self):
        self._webcam = None
        try:
            self.executor.shutdown()
            self.node.destroy_node()
            if self.rclpy.ok():
                self.rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass


class _ArmBackendBase:
    """Same methods as teleop.robot_control.robot_arm.G1_29_ArmController that the main loop uses."""

    def __init__(self, arm_ik):
        names = [arm_ik.reduced_robot.model.names[i] for i in range(1, arm_ik.reduced_robot.model.njoints)]
        missing = [j for j in ARM_JOINTS if j not in names]
        if missing:
            raise ValueError(f"IK model lacks arm joints {missing}")
        self.ik_names = names
        self.q = np.zeros(len(names))
        self.tau = np.zeros(len(names))
        self.t_cmd = 0.0

    def get_current_dual_arm_q(self):
        return self.q.copy()          # IK warm start from the last solution (no joint feedback needed)

    def get_current_dual_arm_dq(self):
        return np.zeros_like(self.q)

    def get_current_motor_q(self):
        return np.zeros(29)

    def ctrl_dual_arm(self, q_target, tauff_target):
        self.q = np.asarray(q_target, dtype=float).copy()
        self.tau = np.asarray(tauff_target, dtype=float).copy()
        self.t_cmd = time.monotonic()
        self._send(dict(zip(self.ik_names, self.q)))

    def _send(self, q_by_name):
        pass

    def ctrl_dual_arm_go_home(self):
        # The mode manager blends the arms back to the default pose when MANIP ends.
        pass


class Ros2ArmBackend(_ArmBackendBase):
    def __init__(self, arm_ik, ros: Ros2TeleopNode):
        super().__init__(arm_ik)
        self.ros = ros

    def _send(self, q_by_name):
        self.ros.publish_upper_body(q_by_name)


class PreviewArmBackend(_ArmBackendBase):
    """--dry-run: IK runs and is drawn in the overlay; nothing is sent anywhere."""


class RetargetHandBackend:
    """Replacement for Brainco_Controller_hand: same shared-array inputs, retargeting runs in a
    thread of this process and the result (contract order, 0..1) goes to `sink(left, right)`."""

    def __init__(self, left_hand_array, right_hand_array, dual_hand_data_lock=None, dual_hand_state_array=None,
                 dual_hand_action_array=None, fps: float = 50.0, xr_motion_data_ready_in=None,
                 sink: Optional[Callable] = None):
        from teleop.robot_control.hand_retargeting import HandRetargeting, HandType
        self.hand_retargeting = HandRetargeting(HandType.BRAINCO_HAND)
        self.left_in, self.right_in = left_hand_array, right_hand_array
        self.lock, self.state_arr, self.action_arr = dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array
        self.ready_in = xr_motion_data_ready_in
        self.fps = fps
        self.sink = sink
        self.latest = (np.zeros(6), np.zeros(6))
        self._prev = None
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True, name="hand-retarget")
        self.thread.start()
        logger_mp.info("[hand] BrainCo retargeting (in-process) running")

    def _loop(self):
        period = 1.0 / self.fps
        while self.running:
            t0 = time.time()
            with self.left_in.get_lock():
                left = np.array(self.left_in[:]).reshape(25, 3)
            with self.right_in.get_lock():
                right = np.array(self.right_in[:]).reshape(25, 3)
            ready = True
            if self.ready_in is not None:
                with self.ready_in.get_lock():
                    ready = bool(self.ready_in.value)
            if ready and np.any(left) and np.any(right):
                try:
                    if self._prev is None or not (np.array_equal(left, self._prev[0]) and np.array_equal(right, self._prev[1])):
                        # new keypoints -> retarget (nlopt); otherwise republish the last result
                        l_r, r_r = brainco_retarget_dual(self.hand_retargeting, left, right)
                        self.latest = (l_r[CONTRACT_FROM_RETARGET], r_r[CONTRACT_FROM_RETARGET])
                        self._prev = (left, right)
                    l_c, r_c = self.latest
                    if self.state_arr is not None and self.action_arr is not None and self.lock is not None:
                        with self.lock:
                            self.action_arr[:] = np.concatenate([l_c, r_c])
                    if self.sink is not None:
                        self.sink(l_c, r_c)
                except Exception as e:  # noqa: BLE001
                    logger_mp.warning(f"[hand] retarget failed: {e}")
            time.sleep(max(0.0, period - (time.time() - t0)))

    def stop(self):
        self.running = False
