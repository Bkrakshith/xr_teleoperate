"""Minimal stand-in for `pinocchio.casadi` (cpin) covering what robot_arm_ik.py uses.

The pip wheels of pinocchio 3.x (`pin`) are built without the CasADi bindings, and the conda
build the upstream README uses cannot be mixed with the ROS 2 Humble system Python. This module
rebuilds symbolic forward kinematics from a numeric `pinocchio.Model` (joint placements, joint
axes, frame placements), so the upstream CasADi/IPOPT IK formulation runs unchanged.

Supported joints: revolute about X/Y/Z, revolute-unaligned, unbounded revolute X/Y/Z
(continuous), prismatic X/Y/Z. Fixed joints are already folded into frames by pinocchio.

API subset:  Model(model) -> cmodel;  cmodel.createData() -> cdata;
             framesForwardKinematics(cmodel, cdata, q_sx);  cdata.oMf[i].translation / .rotation;
             log3(R_sx).
"""
from __future__ import annotations

import casadi
import numpy as np


class _SE3:
    __slots__ = ("rotation", "translation")

    def __init__(self, R, p):
        self.rotation = R
        self.translation = p

    def __mul__(self, other: "_SE3") -> "_SE3":
        return _SE3(casadi.mtimes(self.rotation, other.rotation),
                    self.translation + casadi.mtimes(self.rotation, other.translation))


def _const(M) -> _SE3:
    return _SE3(casadi.SX(np.asarray(M.rotation)), casadi.SX(np.asarray(M.translation).reshape(3, 1)))


def _axis_angle(axis: np.ndarray, q):
    axis = np.asarray(axis, dtype=float).reshape(3)
    axis = axis / np.linalg.norm(axis)
    K = casadi.SX(np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]))
    return casadi.SX.eye(3) + casadi.sin(q) * K + (1 - casadi.cos(q)) * casadi.mtimes(K, K)


_AXES = {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]), "Z": np.array([0, 0, 1.0])}


def _joint_motion(jmodel, q) -> _SE3:
    name = jmodel.shortname()
    iq = jmodel.idx_q
    zero = casadi.SX.zeros(3, 1)
    if name in ("JointModelRX", "JointModelRY", "JointModelRZ"):
        return _SE3(_axis_angle(_AXES[name[-1]], q[iq]), zero)
    if name == "JointModelRevoluteUnaligned":
        return _SE3(_axis_angle(np.asarray(jmodel.axis), q[iq]), zero)
    if name in ("JointModelRUBX", "JointModelRUBY", "JointModelRUBZ"):
        ang = casadi.atan2(q[iq + 1], q[iq])
        return _SE3(_axis_angle(_AXES[name[-1]], ang), zero)
    if name in ("JointModelPX", "JointModelPY", "JointModelPZ"):
        return _SE3(casadi.SX.eye(3), casadi.SX(_AXES[name[-1]].reshape(3, 1)) * q[iq])
    raise NotImplementedError(f"casadi_kinematics: joint type {name} not supported")


class Model:
    def __init__(self, model):
        self.model = model
        self.nq = model.nq

    def createData(self):
        return Data(self)


class Data:
    def __init__(self, cmodel: Model):
        self.cmodel = cmodel
        self.oMi = []
        self.oMf = []


def forwardKinematics(cmodel: Model, cdata: Data, q):
    m = cmodel.model
    oMi = [_SE3(casadi.SX.eye(3), casadi.SX.zeros(3, 1))]
    for i in range(1, m.njoints):
        parent = m.parents[i]
        oMi.append(oMi[parent] * _const(m.jointPlacements[i]) * _joint_motion(m.joints[i], q))
    cdata.oMi = oMi


def framesForwardKinematics(cmodel: Model, cdata: Data, q):
    forwardKinematics(cmodel, cdata, q)
    m = cmodel.model
    oMf = []
    for f in m.frames:
        parent = f.parentJoint if hasattr(f, "parentJoint") else f.parent
        oMf.append(cdata.oMi[parent] * _const(f.placement))
    cdata.oMf = oMf


def log3(R):
    """SO(3) logarithm, smooth at the identity (atan2 form + Taylor branch)."""
    v = casadi.vertcat(R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1])
    s = casadi.sqrt(casadi.sumsqr(v) / 4.0 + 1e-18)
    c = (R[0, 0] + R[1, 1] + R[2, 2] - 1.0) / 2.0
    th = casadi.atan2(s, c)
    factor = casadi.if_else(s < 1e-6, 1.0 + th * th / 6.0, th / s)
    return factor * v / 2.0
