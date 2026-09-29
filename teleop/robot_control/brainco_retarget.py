"""BrainCo Revo2 retargeting helpers without DDS dependencies (used by the native DDS hand
controller in robot_hand_brainco.py and by the ROS 2 backend)."""
import numpy as np

# Normalisation of the retargeted joint angles to BrainCo's 0 (open) .. 1 (closed) command.
# Retarget output order (hand_retargeting.py): thumb_metacarpal (abduction/rotation),
# thumb_proximal (flexion), index, middle, ring, pinky proximal.
# Revo2 ranges from BrainCo's documentation (Hospital_G1 robot/robot_spec.json):
# thumb_metacarpal 0-1.553, thumb_proximal 0-1.047, fingers 0-1.41 rad.
# (Upstream used 1.52 / 1.05 / 1.47, the joint limits of the older BrainCo hand URDF.)
BRAINCO_RETARGET_RANGES = np.array([1.553, 1.047, 1.41, 1.41, 1.41, 1.41])

# NOTE on motor order: robot_hand_brainco.py sends the retarget output order straight to motor ids 0..5,
# i.e. motor 0 <- thumb_metacarpal (rotation) and motor 1 <- thumb_proximal (flexion), which
# contradicts BrainCo's motor table (0 = thumb flexion, 1 = thumb aux/rotation). Kept as upstream
# for the native DDS path; must be checked on real hardware. The ROS 2 backend publishes the
# Hospital_G1 contract order instead: [thumb(flexion), thumb_aux(rotation), index, middle, ring, pinky].
CONTRACT_FROM_RETARGET = [1, 0, 2, 3, 4, 5]


def brainco_normalize(q_retarget):
    """Retargeted joint angles (rad, retarget order) -> 0..1 commands (same order)."""
    return np.clip(np.asarray(q_retarget, dtype=float) / BRAINCO_RETARGET_RANGES, 0.0, 1.0)


def brainco_retarget_dual(hand_retargeting, left_hand_data, right_hand_data):
    """(25,3) OpenXR hand keypoints in the Unitree hand frame -> normalised 0..1 commands
    (retarget order) for both hands. Shared by the DDS controller and the ROS 2 backend."""
    ref_left_value = left_hand_data[hand_retargeting.left_indices[1,:]] - left_hand_data[hand_retargeting.left_indices[0,:]]
    ref_right_value = right_hand_data[hand_retargeting.right_indices[1,:]] - right_hand_data[hand_retargeting.right_indices[0,:]]
    left_q = hand_retargeting.left_retargeting.retarget(ref_left_value)[hand_retargeting.left_dex_retargeting_to_hardware]
    right_q = hand_retargeting.right_retargeting.retarget(ref_right_value)[hand_retargeting.right_dex_retargeting_to_hardware]
    return brainco_normalize(left_q), brainco_normalize(right_q)
