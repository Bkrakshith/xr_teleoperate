"""One Euro filter (Casiez, Roussel, Vogel, CHI 2012) for vectors and rotations."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .geometry import so3_exp, so3_log


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Element-wise One Euro filter. `min_cutoff` may be a scalar or per-element array."""

    def __init__(self, min_cutoff=1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = np.asarray(min_cutoff, dtype=float)
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: Optional[np.ndarray] = None
        self.dx_prev: Optional[np.ndarray] = None
        self.t_prev: Optional[float] = None

    def reset(self, x: Optional[np.ndarray] = None, t: Optional[float] = None):
        self.x_prev = None if x is None else np.array(x, dtype=float)
        self.dx_prev = None if x is None else np.zeros_like(self.x_prev)
        self.t_prev = t

    def __call__(self, x, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.x_prev is None or self.t_prev is None:
            self.reset(x, t)
            return x.copy()
        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev.copy()
        dx = (x - self.x_prev) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        tau = 1.0 / (2.0 * np.pi * cutoff)
        a = 1.0 / (1.0 + tau / dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat.copy()


class OneEuroRotation:
    """One Euro filtering of a rotation matrix on its tangent space around the last estimate."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.R_prev: Optional[np.ndarray] = None
        self.w_prev = np.zeros(3)
        self.t_prev: Optional[float] = None

    def reset(self, R: Optional[np.ndarray] = None, t: Optional[float] = None):
        self.R_prev = None if R is None else np.array(R, dtype=float)
        self.w_prev = np.zeros(3)
        self.t_prev = t

    def __call__(self, R: np.ndarray, t: float) -> np.ndarray:
        if self.R_prev is None or self.t_prev is None:
            self.reset(R, t)
            return np.array(R, dtype=float)
        dt = t - self.t_prev
        if dt <= 0:
            return self.R_prev.copy()
        delta = so3_log(self.R_prev.T @ R)          # body-frame increment
        w = delta / dt
        a_d = _alpha(self.d_cutoff, dt)
        w_hat = a_d * w + (1 - a_d) * self.w_prev
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(w_hat))
        a = _alpha(cutoff, dt)
        R_hat = self.R_prev @ so3_exp(a * delta)
        self.R_prev, self.w_prev, self.t_prev = R_hat, w_hat, t
        return R_hat.copy()
