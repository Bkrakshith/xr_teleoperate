"""Virtual-gimbal experiment: which gimbal-compensation model fits MediaPipe?

A still image is re-rendered as the camera would see it after pan/tilt (pure-rotation
homography K R^T K^-1), MediaPipe runs on every view and the arm vectors (pose: wrist - shoulder)
or palm orientations (hands) are rotated into the body frame with three models:
  none : ignore the gimbal
  cam  : rotate by the gimbal pan/tilt only (landmarks assumed camera-axis aligned)
  ray  : gimbal pan/tilt composed with the rotation onto the ray through the person/hand
         (landmarks assumed aligned with the crop the model saw) - the tracker's default.
A good model keeps the body-frame result constant across views.

  cd teleop && python -m webcam.virtual_gimbal_check --mode pose person.jpg
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from teleop.webcam import geometry as g  # noqa: E402

MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--mode", choices=["pose", "hands"], default="pose")
    ap.add_argument("--hfov", type=float, default=67.0, help="assumed horizontal FOV of the still")
    a = ap.parse_args()
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    if a.mode == "pose":
        det = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=os.path.join(MODELS, "pose_landmarker_full.task")),
            running_mode=vision.RunningMode.IMAGE))
    else:
        det = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=os.path.join(MODELS, "hand_landmarker.task")),
            num_hands=2, running_mode=vision.RunningMode.IMAGE))
    img = cv2.imread(a.image)
    h0, w0 = img.shape[:2]
    pad = int(0.4 * max(h0, w0))
    big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    H, W = big.shape[:2]
    f = 0.5 * w0 / np.tan(np.radians(a.hfov) / 2)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    intr = (f, f, W / 2, H / 2)
    res = {}
    for yaw in range(-20, 21, 5):
        for pitch in range(-10, 11, 5):
            Hm = K @ g.R_base_cam(yaw, pitch).T @ np.linalg.inv(K)
            view = cv2.cvtColor(cv2.warpPerspective(big, Hm, (W, H)), cv2.COLOR_BGR2RGB)
            r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=view))
            models = lambda px: {"none": np.eye(3), "cam": g.R_base_obs(yaw, pitch),
                                 "ray": g.R_base_obs(yaw, pitch, px, intr)}
            out = {}
            if a.mode == "pose" and r.pose_landmarks:
                lm = np.array([[p.x * W, p.y * H] for p in r.pose_landmarks[0]])
                wl = np.array([[p.x, p.y, p.z] for p in r.pose_world_landmarks[0]])
                vec = np.stack([wl[15] - wl[11], wl[16] - wl[12]])
                for m, Rb in models(tuple(0.5 * (lm[11] + lm[12]))).items():
                    out[("arms", m)] = g.observation_to_body(vec, Rb)
            elif a.mode == "hands":
                for i, lms in enumerate(r.hand_landmarks):
                    px = np.array([[p.x * W, p.y * H] for p in lms])
                    wl = np.array([[p.x, p.y, p.z] for p in r.hand_world_landmarks[i]])
                    wb = np.linalg.inv(Hm) @ np.array([px[0, 0], px[0, 1], 1.0])
                    key = "upper" if wb[1] / wb[2] < H / 2 else "lower"
                    for m, Rb in models(tuple(px.mean(axis=0))).items():
                        out[(key, m)] = g.palm_frame(g.observation_to_body(wl, Rb))
            res[(yaw, pitch)] = out
    ref = res.get((0, 0), {})
    for m in ("none", "cam", "ray"):
        e = []
        for k, out in res.items():
            if k == (0, 0):
                continue
            for (part, mm), v in out.items():
                if mm != m or (part, mm) not in ref:
                    continue
                if a.mode == "pose":
                    e.append(np.linalg.norm(v - ref[(part, mm)], axis=1).mean() * 100)
                else:
                    e.append(np.degrees(np.linalg.norm(g.so3_log(ref[(part, mm)].T @ v))))
        unit = "cm (wrist-shoulder vector)" if a.mode == "pose" else "deg (palm orientation)"
        if e:
            print(f"{m:5s}: n={len(e):3d}  mean {np.mean(e):5.1f}  p90 {np.percentile(e, 90):5.1f} {unit}")
    det.close()


if __name__ == "__main__":
    main()
