from __future__ import annotations
from typing import Any
from sympy import Matrix, Symbol, cos, sin, symbols, Derivative
from sympy.physics.mechanics import (
    PinJoint,
    PrismaticJoint,
    Point,
    RigidBody,
    inertia,
    ReferenceFrame,
    System,
    TorqueActuator,
    Vector,
    WeldJoint,
    dynamicsymbols,
)
from symbrim.brim import PelvisInterPointMixin
from symbrim.core import LoadGroupBase
from src.SimJointBase import InterSeatJointBase, ShiftingSideLeanSeatBase

class ShiftingSideLeanSeat(PelvisInterPointMixin, ShiftingSideLeanSeatBase):
    """ This Rider seat connection is to be used in conjuction with the Interseat-model and -joint.
     This joint consists of the pinjoint that makes the rider lean on the interseat-model, while the
     interseat-joint makes the rider shift laterally on the seat while leaning. It will be coupled,
     such that a certain degree in lean angle is equal to a certain amount of lateral displacement"""

    @property
    def descriptions(self) -> dict[Any, str]:

        return {
            **super().descriptions,
            self.q[0]: "Lean angle.",
            self.u[0]: "Angular lean velocity.",
            self.symbols["alpha"]: "Angle of the rider lean axis.",
        }

    def _define_objects(self) -> None:

        super()._define_objects()
        self.q = Matrix([dynamicsymbols(self._add_prefix("q"))])
        self.u = Matrix([dynamicsymbols(self._add_prefix("u"))])
        alpha = Symbol(self._add_prefix("alpha"))
        self.symbols["alpha"] = alpha
        self._frame_lean_axis = (cos(alpha) * self.interseat.frame.x -
                                 sin(alpha) * self.interseat.frame.z)
        self._pelvis_lean_axis = self.pelvis.x
        self._pelvis_fixed_point = self.pelvis.body.masscenter
        self._system = System(
            self.interseat.system.frame, self.interseat.system.fixed_point)




    def _define_kinematics(self) -> None:
        super()._define_kinematics()
        self.system.add_joints(
            PinJoint(
                self._add_prefix("lean_joint"),
                self.interseat.body,  # Parent: interseat
                self.pelvis.body,  # Child: pelvis
                self.q[0], self.u[0],  # Lean angle and velocity
                self.interseat.inter_pelvis_point,  # Parent point
                self._pelvis_interpoint,  # Fixed point on the pelvis
                self._frame_lean_axis, self._pelvis_lean_axis,
            ))


    @property
    def frame_lean_axis(self) -> Vector:
        """Return the lean axis of the rear frame."""
        return self._frame_lean_axis

    def saddle_axis(self) -> Vector:
        return self._saddle_axis

    @frame_lean_axis.setter
    def frame_lean_axis(self, value: Vector) -> None:
        """Set the lean axis of the rear frame."""
        try:
            value.express(self.rear_frame.saddle.frame)
        except ValueError as e:
            raise ValueError("Lean axis must be expressable in the rear frame.") from e
        self._frame_lean_axis = value

    @property
    def pelvis_lean_axis(self) -> Vector:
        """Lean axis of the pelvis."""
        return self._pelvis_lean_axis

    @pelvis_lean_axis.setter
    def pelvis_lean_axis(self, value: Vector) -> None:
        try:
            value.express(self.pelvis.frame)
        except ValueError as e:
            raise ValueError(
                "Lean axis must be expressable in the pelvis frame.") from e
        self._pelvis_lean_axis = value

class ShiftingSideLeanSeatTorque(LoadGroupBase):
    """Torque load group for the side lean seat connection."""

    parent: ShiftingSideLeanSeat
    required_parent_type = ShiftingSideLeanSeat

    @property
    def descriptions(self) -> dict[object, str]:
        """Descriptions of the objects."""
        return {
            **super().descriptions,
            self.symbols["T"]: f"Side lean torque of {self.parent}",
        }

    def _define_objects(self) -> None:
        """Define the objects."""
        self.symbols["T"] = dynamicsymbols(self._add_prefix("T"))

    def _define_loads(self) -> None:
        """Define the kinematics."""
        self.system.add_actuators(
            TorqueActuator(
                self.symbols["T"], self.parent.frame_lean_axis,
                self.parent.pelvis.frame, self.parent.interseat.frame)
        )

class ShiftingSideLeanSeatSpringDamper(LoadGroupBase):
    """Torque applied to the side lean connection as linear spring-damper."""

    parent: ShiftingSideLeanSeat
    required_parent_type = ShiftingSideLeanSeat

    @property
    def descriptions(self) -> dict[object, str]:
        """Descriptions of the objects."""
        return {
            **super().descriptions,
            self.symbols["k"]: f"Side lean stiffness of {self.parent}",
            self.symbols["c"]: f"Side lean damping of {self.parent}",
            self.symbols["q_ref"]: f"Side lean reference angle of {self.parent}",
        }

    def _define_objects(self) -> None:
        """Define the objects."""
        self.symbols.update({
            "k": dynamicsymbols(self._add_prefix("k")),
            "c": dynamicsymbols(self._add_prefix("c")),
            "q_ref": dynamicsymbols(self._add_prefix("q_ref")),
        })
        #super().define_objects()
        #self.symbols["k"] = Symbol(self._add_prefix("k"))
        #self.symbols["c"] = Symbol(self._add_prefix("c"))
        #self.symbols["q_ref"] = Symbol(self._add_prefix("q_ref"))

    def _define_loads(self) -> None:
        """Define the kinematics."""
        pin = self.parent.system.joints[0]
        self.system.add_actuators(
            TorqueActuator(
                -self.symbols["k"] * (pin.coordinates[0] - self.symbols["q_ref"]) -
                self.symbols["c"] * pin.speeds[0],
                self.parent.frame_lean_axis, self.parent.pelvis.frame,
                self.parent.interseat.frame)
        )

class InterSeatJoint(InterSeatJointBase):
    """ This Rider seat connection is to be used in conjuction with the Interseat-model and -joint.
     This joint consists of the pinjoint that makes the rider lean on the interseat-model, while the
     interseat-joint makes the rider shift laterally on the seat while leaning. It will be coupled,
     such that a certain degree in lean angle is equal to a certain amount of lateral displacement"""

    @property
    def descriptions(self) -> dict[object, str]:

        return {
            **super().descriptions,
            self.q[0]: "Lateral linear displacement.",
            self.u[0]: "Lateral linear velocity.",
            self.symbols["translation_factor"]: "Relation between the lean angle and lateral translation."
        }

    def _define_objects(self) -> None:
        super()._define_objects()
        self.q = Matrix([dynamicsymbols(self._add_prefix("q"))])
        self.u = Matrix([dynamicsymbols(self._add_prefix("u"))])

        translation_factor = Symbol(self._add_prefix("translation_factor"))
        self.symbols["translation_factor"] = translation_factor

        self._system = System(
            self.interseat.system.frame, self.interseat.system.fixed_point)
        self.translation = self.symbols["translation_factor"]
        self._saddle_axis = self.rear_frame.saddle.frame.y #self.interseat.frame.y

    def _define_kinematics(self) -> None:
        super()._define_kinematics()
        self.system.add_joints(
            PrismaticJoint(
                self._add_prefix("prismatic_joint"),
                self.rear_frame.saddle.to_valid_joint_arg(),  # Parent: saddle
                self.interseat.body,  # Child: interseat
                self.q, self.u,  # Lateral displacement and velocity
                self.rear_frame.saddle.point,  # Parent point
                self.interseat.inter_saddle_point,  # Child point
                self._saddle_axis,  # The prismatic axis defined above
                self.interseat.y  # Axis of the interseat joint
            ))

    def saddle_axis(self) -> Vector:
        return self._saddle_axis

    @property
    def pelvis_lean_axis(self) -> Vector:
        """Lean axis of the pelvis."""
        return self._pelvis_lean_axis

    @pelvis_lean_axis.setter
    def pelvis_lean_axis(self, value: Vector) -> None:
        try:
            value.express(self.pelvis.frame)
        except ValueError as e:
            raise ValueError(
                "Lean axis must be expressable in the pelvis frame.") from e
        self._pelvis_lean_axis = value
