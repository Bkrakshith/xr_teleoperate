"""Headless tracker benchmark: runs capture + MediaPipe + mapping (+ framing) for N seconds and
prints FPS, capture->output latency, detection rates and PTZ activity. No frames are written
to disk unless --save-annotated is given.

  cd teleop && python -m webcam.bench --seconds 20 [--source /dev/video0|clip.mp4|hand.jpg]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from teleop.webcam.tracker import TrackerConfig, WebcamTracker  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="auto")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--pose-model", default="full")
    ap.add_argument("--no-framing", action="store_true")
    ap.add_argument("--save-annotated", default="", help="write the last annotated frame here")
    a = ap.parse_args()
    w, h = (int(v) for v in a.size.split("x"))
    tr = WebcamTracker(TrackerConfig(source=a.source, width=w, height=h, pose_model=a.pose_model,
                                     framing=not a.no_framing)).start()
    t0 = time.monotonic()
    seen = set()
    n = 0
    hands = np.zeros(2)
    arms = np.zeros(2)
    lat = []
    states = {}
    while time.monotonic() - t0 < a.seconds:
        time.sleep(0.02)
        s = tr.latest()
        if s is None or s.t in seen:
            continue
        seen.add(s.t)
        n += 1
        hands += s.hands_visible
        arms += s.arms_visible
        lat.append((s.t - s.t_capture) * 1000)
        states[s.framing_detail] = states.get(s.framing_detail, 0) + 1
    snap = tr.latest()
    img, _ = tr.latest_annotated()
    if a.save_annotated and img is not None:
        import cv2
        cv2.imwrite(a.save_annotated, img)
    tr.stop()
    out = {
        "source": tr.cfg.source, "frames_processed": n, "seconds": a.seconds,
        "fps": round(n / a.seconds, 1), "fps_ema": round(snap.fps, 1) if snap else 0,
        "latency_ms_capture_to_output": {"median": round(float(np.median(lat)), 1) if lat else None,
                                          "p95": round(float(np.percentile(lat, 95)), 1) if lat else None},
        "hands_visible_frac": (hands / max(n, 1)).round(2).tolist(),
        "arms_visible_frac": (arms / max(n, 1)).round(2).tolist(),
        "framing_states": states, "ptz": tr.ptz_note or "OBSBOT UVC",
        "ptz_final_cmd": list(tr.ptz.commanded()) if tr.ptz is not None else None,
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
