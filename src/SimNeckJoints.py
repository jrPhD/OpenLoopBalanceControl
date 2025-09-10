from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from sympy import Symbol
from sympy.physics.mechanics import System, Vector, WeldJoint
from src.Sim_bodies import SimPelvis, SimTorso
from sympy import Matrix, cos, sin
from sympy.physics.mechanics import (
    PinJoint,
    Point,
    ReferenceFrame,
    SphericalJoint,
    System,
    Torque,
    TorqueActuator,
    dynamicsymbols,
)

with contextlib.suppress(ImportError):
    import numpy as np
    from symbrim.utilities.parametrize import get_inertia_vals_from_yeadon

    import bicycleparameters
    if TYPE_CHECKING:
        from bicycleparameters import Bicycle
from src.Sim_bodies import SimHead, SimTorso
from src.SimJointBase import NeckBase
from symbrim.core import LoadGroupBase, Attachment
from symbrim.core import ConnectionBase, ModelRequirement

__all__ = ["NeckPinJoint", "NeckPinTorque", "NeckPinSpringDamper", "FixedNeck"]


class NeckPinJoint(NeckBase):
    """Pinjoint to connect the head to the torso """

    @property
    def descriptions(self) -> dict[object, str]:
        """return descriptions"""
        return {
            **super().descriptions,
            self.q[0]: "Adduction angle of the neck",
            self.u[0]: "Adduction angular velocity of the neck",
            self.symbols["beta"]: "Angle of the neck lean axis. a.k.a. the flexion angle of the head"
        }

    def _define_objects(self) -> None:
        """Define the objects."""
        super()._define_objects()
        self.q = Matrix([dynamicsymbols(self._add_prefix("q"))])
        self.u = Matrix([dynamicsymbols(self._add_prefix("u"))])
        self._system = System.from_newtonian(self.torso.body)
        beta = Symbol(self._add_prefix("beta"))
        self.symbols["beta"] = beta
        self._torso_neck_lean_axis = ((cos(beta) * self.torso.frame.x -
                                      sin(beta) * self.torso.frame.z))
        self._head_lean_axis = self.head.x
        self._system = System(
            self.torso.system.frame, self.torso.system.fixed_point)

    def _define_kinematics(self) -> None:
        """Define the kinematics."""
        super()._define_kinematics()
        self.system.add_joints(
            PinJoint(
                self._add_prefix("neckjoint"), self.torso.body, self.head.body,
                self.q, self.u, parent_point=self.torso.torso_neck_point, child_point=self.head.neck_point,
                parent_interframe=self.torso.frame, child_interframe=self.head.frame, joint_axis=self._torso_neck_lean_axis))

    def head_lean_axis(self) -> Vector:

        return self._head_lean_axis

class NeckPinTorque(LoadGroupBase):
    """ torque for lateral neck control """
    parent: NeckPinJoint
    required_parent_type = NeckPinJoint

    @property
    def descriptions(self) -> dict[object, str]:
        """Descriptions of the objects."""
        return {
            **super().descriptions,
            self.symbols["T"]: "Adduction torque of the neck.",
        }

    def _define_objects(self) -> None:
        """Define the objects."""
        self.symbols["T"] = dynamicsymbols(self._add_prefix("T"))

    def _define_loads(self) -> None:
        """Define the loads."""
        neck = self.parent.system.joints[0]
        adduction_axis = (cos(neck.coordinates[0]) * neck.parent_interframe.x -
                          sin(neck.coordinates[0]) * neck.parent_interframe.z)

        self.parent.system.add_actuators(
            TorqueActuator(self.symbols["T"], adduction_axis,
                           self.parent.torso.frame, self.parent.head.frame))

class NeckPinSpringDamper(LoadGroupBase):
    parent: NeckPinJoint
    required_parent_type = NeckPinJoint

    @property
    def descriptions(self) -> dict[object, str]:
        """Descriptions of the objects."""
        return {
            **super().descriptions,
            self.symbols["k"]: f"neck stiffness of {self.parent}",
            self.symbols["c"]: f"neck damping of {self.parent}",
            self.symbols["q_ref"]: f"neck reference angle of {self.parent}",
        }

    def _define_objects(self) -> None:
        """Define the objects."""
        self.symbols.update({
            "k": dynamicsymbols(self._add_prefix("k")),
            "c": dynamicsymbols(self._add_prefix("c")),
            "q_ref": dynamicsymbols(self._add_prefix("q_ref")),
        })

    def _define_loads(self) -> None:
        """Define the loads."""
        neck = self.parent.system.joints[0]
        adduction_axis = (cos(neck.coordinates[0]) * neck.parent_interframe.x -
                          sin(neck.coordinates[0]) * neck.parent_interframe.z)
        if isinstance(self.parent, NeckBase):
            rot_dir = -1
        else:
            adduction_axis *= -1
            rot_dir = 1

        self.parent.system.add_actuators(
            TorqueActuator(
                -self.symbols["k"] * (neck.coordinates[0] - self.symbols["q_ref"]) -
                self.symbols["c"] * neck.speeds[0],
                adduction_axis, self.parent.head.frame, self.parent.torso.frame)
        )

class FixedNeck(NeckBase):
    """Fixed connection between the head to the torso """

    @property
    def descriptions(self) -> dict[object, str]:
        """return descriptions"""
        return {
            **super().descriptions,
            self.symbols["yaw"]: "Yaw angle of the head w.r.t. the torso.",
            self.symbols["pitch"]: "Pitch angle of the head w.r.t. the torso.",
            self.symbols["roll"]: "Roll angle of the head w.r.t. the torso.",
        }

    def _define_objects(self) -> None:
        """Define the objects."""
        super()._define_objects()

        self._system = System.from_newtonian(self.torso.body)
        self.symbols.update({
            name: Symbol(self._add_prefix(name)) for name in ("yaw", "pitch", "roll")})
        self._rear_interframe = ReferenceFrame(self._add_prefix("rear_interframe"))

        self._system = System(
            self.torso.system.frame, self.torso.system.fixed_point)

    def _define_kinematics(self) -> None:
        """Define the kinematics."""
        super()._define_kinematics()
        self.system.add_joints(WeldJoint(
            self._add_prefix("neckjoint"), self.torso.body, self.head.body,
            parent_point=self.torso.torso_neck_point, child_point=self.head.neck_point,
            parent_interframe=self.torso.frame, child_interframe=self.head.frame))

    def head_lean_axis(self) -> Vector:
        return self._head_lean_axis