"""Module containing the Whipple bicycle model."""
from __future__ import annotations

import contextlib

from sympy import Matrix, Symbol
from sympy.physics.mechanics import (
    PinJoint,
    TorqueActuator,
    ReferenceFrame,
    System,
    Vector,
    dynamicsymbols,
    Force,
    Torque,
)

from symbrim.bicycle.bicycle_base import BicycleBase
from symbrim.bicycle.front_frames import FrontFrameBase
from symbrim.bicycle.grounds import GroundBase
from symbrim.bicycle.rear_frames import RearFrameBase
from symbrim.bicycle.tires import TireBase
from symbrim.bicycle.wheels import WheelBase
from symbrim.core import ConnectionRequirement, ModelRequirement, set_default_convention

__all__ = ["WhippleBicycleSprungSteering", "WhippleBicycle"]


@set_default_convention("moore")
class WhippleBicycle(BicycleBase):
    """Base class for the Whipple bicycle model."""

    required_models: tuple[ModelRequirement, ...] = (
        ModelRequirement("ground", GroundBase, "Submodel of the ground."),
        ModelRequirement("rear_frame", RearFrameBase, "Submodel of the rear frame."),
        ModelRequirement("front_frame", FrontFrameBase, "Submodel of the front frame."),
        ModelRequirement("rear_wheel", WheelBase, "Submodel of the rear wheel."),
        ModelRequirement("front_wheel", WheelBase, "Submodel of the front wheel."),
    )
    required_connections: tuple[ConnectionRequirement, ...] = (
        ConnectionRequirement("front_tire", TireBase,
                              "Tire model for the front wheel."),
        ConnectionRequirement("rear_tire", TireBase,
                              "Tire model for the rear wheel."),
    )
    ground: GroundBase
    rear_frame: RearFrameBase
    front_frame: FrontFrameBase
    rear_wheel: WheelBase
    front_wheel: WheelBase
    rear_tire: TireBase
    front_tire: TireBase

    def _define_connections(self) -> None:
        """Define the connections between the submodels."""
        super()._define_connections()
        self.rear_tire.ground = self.ground
        self.rear_tire.wheel = self.rear_wheel
        self.front_tire.ground = self.ground
        self.front_tire.wheel = self.front_wheel


class WhippleBicycleSprungSteering(WhippleBicycle):
    """Whipple bicycle model based on Moore's convention."""

    convention: str = "moore"

    @property
    def descriptions(self) -> dict[object, str]:
        """Dictionary of descriptions of the Whipple bicycle's symbols."""
        desc = {
            **super().descriptions,
            self.q[0]: f"Perpendicular distance along ground.x to the rear contact "
                       f"point in the ground plane of {self.name}.",
            self.q[1]: f"Perpendicular distance along ground.y to the rear contact "
                       f"point in the ground plane of {self.name}.",
            self.q[2]: f"Yaw angle of the rear frame of {self.name}.",
            self.q[3]: f"Roll angle of the rear frame of {self.name}.",
            self.q[4]: f"Pitch angle of the rear frame of {self.name}.",
            self.q[5]: f"Front wheel rotation angle of {self.name}.",
            self.q[6]: f"Steering rotation angle of {self.name}.",
            self.q[7]: f"Rear wheel rotation angle of {self.name}.",
            self.symbols["k"]: f"Spring stiffness of {self.parent}",
            self.symbols["c"]: f"Damping of {self.parent}",
            self.symbols["q_ref"]: f"Spring reference angle of {self.parent}",
        }
        desc.update({ui: f"Generalized speed of the {desc[qi].lower()}"
                     for qi, ui in zip(self.q, self.u)})
        if "gear_ratio" in self.symbols:
            desc[self.symbols["gear_ratio"]] = (
                "Ratio between the angle of the rear wheel and the cranks.")
        return desc

    def _define_objects(self) -> None:
        """Define the objects of the Whipple bicycle."""
        super()._define_objects()
        self._system = System(self.ground.frame, self.ground.origin)
        self.rear_tire.define_objects()
        self.rear_tire.on_ground = True
        self.front_tire.define_objects()
        self.q = Matrix(dynamicsymbols(self._add_prefix("q1:9")))
        self.u = Matrix(dynamicsymbols(self._add_prefix("u1:9")))
        if self.cranks is not None:
            self.symbols["gear_ratio"] = Symbol(self._add_prefix("gear_ratio"))

    def _define_kinematics(self) -> None:
        """Define the kinematics of the Whipple bicycle."""
        super()._define_kinematics()
        qd_repl = dict(zip(self.q.diff(dynamicsymbols._t), self.u))
        # Define the location of the rear wheel contact point in the ground frame.
        self.ground.set_pos_point(self.rear_tire.contact_point, self.q[:2])
        self.rear_tire.contact_point.set_vel(
            self.ground.frame,
            self.rear_tire.contact_point.vel(self.ground.frame).xreplace(qd_repl))
        # Define the orientation of the rear frame.
        self.roll_frame = ReferenceFrame("roll_frame")
        
        self.roll_frame.orient_body_fixed(self.ground.frame, (*self.q[2:4], 0), "zxy")
        self.roll_frame.set_ang_vel(self.ground.frame, self.roll_frame.ang_vel_in(
            self.ground.frame).xreplace(qd_repl))
        self.rear_frame.wheel_hub.frame.orient_axis(self.roll_frame, self.roll_frame.y, self.q[4])
        self.rear_frame.wheel_hub.frame.set_ang_vel(
            self.ground.frame,
            self.rear_frame.wheel_hub.frame.ang_vel_in(
                self.ground.frame).xreplace(qd_repl))
        
        # self.roll_frame = roll_frame #JR
        # Define the joints
        self.system.add_joints(
            PinJoint(self._add_prefix("rear_wheel_joint"),
                     self.rear_frame.wheel_hub.to_valid_joint_arg(),
                     self.rear_wheel.body, self.q[5], self.u[5],
                     self.rear_frame.wheel_hub.point, self.rear_wheel.center,
                     self.rear_frame.wheel_hub.axis, self.rear_wheel.rotation_axis),
            PinJoint(self._add_prefix("steer_joint"),
                     self.rear_frame.steer_hub.to_valid_joint_arg(),
                     self.front_frame.steer_hub.to_valid_joint_arg(),
                     self.q[6], self.u[6],
                     self.rear_frame.steer_hub.point, self.front_frame.steer_hub.point,
                     self.rear_frame.steer_hub.axis, self.front_frame.steer_hub.axis),
            PinJoint(self._add_prefix("front_wheel_joint"),
                     self.front_frame.wheel_hub.to_valid_joint_arg(),
                     self.front_wheel.body, self.q[7], self.u[7],
                     self.front_frame.wheel_hub.point, self.front_wheel.center,
                     self.front_frame.wheel_hub.axis,
                     self.front_wheel.rotation_axis),
        )
        # Define the spring which "stabilizes" the steer with respect to the rear frame.
        # The spring stiffness is defined here, because doing it in the model configuration messes things up.
        steerjoint = self.system.joints[1]
        #torque = - self.symbols["k"] * (steerjoint.coordinates[0] - self.symbols["q_ref"]) - self.symbols["c"] * steerjoint.speeds[0]
        torquee = - 10 * (steerjoint.coordinates[0] - self.symbols["q_ref"])
        self.system.add_loads(
            Torque(self.rear_frame.steer_hub.to_valid_joint_arg(), torquee * self.rear_frame.steer_hub.axis),
            Torque(self.front_frame.steer_hub.to_valid_joint_arg(), -torquee * self.front_frame.steer_hub.axis)
        )

        with contextlib.suppress(ValueError):
            normal = self.ground.get_normal(self.rear_tire.contact_point)
            direction = normal.dot(-self.ground.frame.z)
            self.rear_tire.upward_radial_axis = direction * -self.roll_frame.z
            self.rear_tire.longitudinal_axis = direction * self.roll_frame.x
            fake_roll = ReferenceFrame("fake_roll")
            fake_yaw = ReferenceFrame("fake_yaw")
            fake_roll.orient_axis(fake_yaw, fake_yaw.x, self.q[3])
            self.rear_tire.lateral_axis = Vector({
                self.
                roll_frame: fake_yaw.y.to_matrix(fake_roll)})
        self.rear_tire.define_kinematics()
        self.front_tire.define_kinematics()
        # Add the coordinates and speeds to the system.
        self.system.add_coordinates(*self.q[:5])
        self.system.add_speeds(*self.u[:5])
        self.system.add_kdes(*(
            ui - qi.diff(dynamicsymbols._t) for qi, ui in zip(self.q[:5], self.u[:5])))
        if self.cranks:
            self.cranks.center_point.set_pos(self.rear_frame.bottom_bracket, 0)
            self.cranks.frame.orient_axis(
                self.rear_frame.wheel_hub.frame, self.rear_frame.wheel_hub.axis,
                self.q[7] / self.symbols["gear_ratio"])
            self.cranks.frame.set_ang_vel(
                self.rear_frame.wheel_hub.frame,
                self.u[7] / self.symbols["gear_ratio"] * self.rear_frame.wheel_hub.axis)

    def _define_loads(self) -> None:
        """Define the loads of the Whipple bicycle."""
        super()._define_loads()
        self.rear_tire.define_loads()
        self.front_tire.define_loads()

    def _define_constraints(self) -> None:
        """Define the constraints of the Whipple bicycle."""
        super()._define_constraints()
        self.rear_tire.define_constraints()
        self.front_tire.define_constraints()