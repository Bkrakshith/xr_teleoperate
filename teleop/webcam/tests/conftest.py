import os
import sys

# make `teleop.*` importable (repo root = two levels above teleop/webcam)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
