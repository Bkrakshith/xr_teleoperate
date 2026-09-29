"""Desk-webcam input for xr_teleoperate (Hospital_G1): MediaPipe arm + hand tracking with
OBSBOT Tiny 3 auto-framing, standing in for televuer's TeleVuerWrapper."""
from .tele_wrapper import TeleData, WebcamTeleWrapper, robot_arm_model_from_ik
from .tracker import TrackerConfig, WebcamTracker
from .mapper import ArmHandMapper, MapperConfig, RobotArmModel

__all__ = ["WebcamTeleWrapper", "TeleData", "TrackerConfig", "WebcamTracker", "ArmHandMapper",
           "MapperConfig", "RobotArmModel", "robot_arm_model_from_ik"]
