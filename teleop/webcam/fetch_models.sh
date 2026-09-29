#!/usr/bin/env bash
# Download the MediaPipe Tasks models used by the webcam tracker (CPU, float16).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p models
base=https://storage.googleapis.com/mediapipe-models
for u in pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task \
         pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task \
         pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task \
         hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task; do
    f=models/$(basename "$u")
    [ -s "$f" ] || { echo "fetching $f"; curl -sSfL -o "$f" "$base/$u"; }
done
ls -la models
