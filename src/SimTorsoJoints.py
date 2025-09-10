from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from sympy import Symbol
from sympy.physics.mechanics import System, Vector, WeldJoint

from symbrim.rider.base_connections import SacrumBase
from symbrim.core import Attachment, LoadGroupBase
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




__all__ = ["SphericalTorsoJoint", "PinTorsoJoint",
           "PinTorsoJointTorque", "PinTorsoJointSpringDamper",
           "SphericalTorsoJointTorque", "SphericalTorsoJointSpringDamper"]



class SphericalTorsoJoint(SacrumBase):
    """ Spherical joint between pelvis and torso"""
    @property
    def descriptions(self) -> dict[object, str]:
        """Return the descriptions."""
        return {
            **super().descriptions,
            self.q[0]: "Flexion angle of the torso",
            self.q[1]: "Adduction angle of the torso.",
            self.q[2]: "Endorotation/Twisting angle of the torso.",
            self.u[0]: "Flexion angular velocity of the torso.",
            self.u[1]: "Adduction angular velocity of the torso.",
            self.u[2]: "Rotation/Twisting angular velocity of the torso.",
            self.symbols["d_p_t"]: "Distance from the torso center of mass to the "
                                   "pelvis.",
        }

    def _define_objects(self) -> None:
        """Define the objects."""
        super()._define_objects()
        self.q = Matrix(
            dynamicsymbols(self._add_prefix("q_flexion, q_adduction, q_rotation")))
        self.u = Matrix(
            dynamicsymbols(self._add_prefix("u_flexion, u_adduction, u_rotation")))
        self.symbols["d_p_t"] = Symbol(self._add_prefix("d_p_t"))
        self._torso_wrt_pelvis = self.pelvis.body.masscenter#-self.symbols["d_p_t"] * self.pelvis.z
        self._system = System.from_newtonian(self.pelvis.body)

    def _define_kinematics(self) -> None:
        """Define the kinematics."""
        super()._define_kinematics()
        self.system.add_joints(
            SphericalJoint(
                self._add_prefix("joint"), self.pelvis.body, self.torso.body,
                self.q, self.u, parent_point=self._torso_wrt_pelvis, #child_point=self._torso_wrt_pelvis,
                parent_interframe=self.pelvis.frame, child_interframe=self.torso.frame,
                rot_type="BODY", amounts=(self.q[0], self.q[1], self.q[2]), rot_order="YXZ")
        )
    """    
    def get_param_values(self, bicycle_parameters: Bicycle) -> dict[Symbol, float]:
        ""Get a parameters mapping of a model based on a bicycle parameters object.""
        params = super().get_param_values(bicycle_parameters)
        human = bicycle_parameters.human
        if human is None:
            return params
        torso_props = human.combine_inertia(("T", "C"))
        params[self.symbols["d_p_t"]] = np.linalg.norm(
            torso_props[1] - human.P.center_of_mass)
        return params
    """

    @property
    def torso_wrt_pelvis(self) -> Vector:
        """Return the position of the torso w.r.t. the pelvis center of mass."""
        return self._torso_wrt_pelvis

    @torso_wrt_pelvis.setter
    def torso_wrt_pelvis(self, value: Vector) -> None:
        try:
            value.express(self.pelvis.frame)
        except ValueError as e:
            raise ValueError(
                "Torso position must be expressable in the pelvis frame.") from e
        self._torso_wrt_pelvis = value

    def get_param_values(self, bicycle_parameters: Bicycle) -> dict[Symbol, float]:
        """Get a parameters mapping of a model based on a bicycle parameters object."""
        params = super().get_param_values(bicycle_parameters)
        human = bicycle_parameters.human
        if human is None:
            return params
        torso_props = human.combine_inertia(("T", "C"))
        params[self.symbols["d_p_t"]] = np.linalg.norm(
            torso_props[1] - human.P.center_of_mass)
        return params

class SphericalTorsoJointTorque(LoadGroupBase):
    """Torque for the spherical shoulder joints."""

    parent: SphericalTorsoJoint
    required_parent_type = SphericalTorsoJoint

    @property
    def descriptions(self) -> dict[object, str]:
        """Descriptions of the objects."""
        return {
            **super().descriptions,
            self.symbols["T_flexion"]: f"Flexion torque of torso: {self.parent}.",
            self.symbols["T_adduction"]:
                f"Adduction torque of torso: {self.parent}.",
            self.symbols["T_rotation"]:
                f"Endorotation torque of torso: {self.parent}.",
        }

    def _define_objects(self) -> None:
        """Define the objects."""
        self.symbols.update({name: dynamicsymbols(self._add_prefix(name)) for name in (
            "T_flexion", "T_adduction", "T_rotation")})

    def _define_loads(self) -> None:
        """Define the loads."""
        torsojoint = self.parent.system.joints[0]
        adduction_axis = (cos(torsojoint.coordinates[0]) * torsojoint.parent_interframe.x -
                          sin(torsojoint.coordinates[0]) * torsojoint.parent_interframe.z)
        if isinstance(self.parent, SacrumBase):
            rot_dir = -1
        else:
            adduction_axis *= -1
            rot_dir = 1
        torque = (self.symbols["T_flexion"] * torsojoint.parent_interframe.y +
                  self.symbols["T_adduction"] * adduction_axis +
                  self.symbols["T_rotation"] * rot_dir * torsojoint.child_interframe.z)
        self.parent.system.add_loads(
            Torque(torsojoint.child_interframe, torque),
            Torque(torsojoint.parent_interframe, -torque)
        )

class SphericalTorsoJointSpringDamper(LoadGroupBase):
    """Spherical for the spherical shoulder joints."""

    parent: SphericalTorsoJoint
    required_parent_type = SphericalTorsoJoint

    @property
    def descriptions(self) -> dict[object, str]:
        """Descriptions of the objects."""
        desc = {**super().descriptions}
        for tp in ("flexion", "adduction", "rotation"):
            desc.update({
                self.symbols[f"k_{tp}"]:
                    f"{tp.capitalize()} stiffness of torsojoint: {self.parent}.",
                self.symbols[f"c_{tp}"]:
                    f"{tp.capitalize()} damping of:torsojoint {self.parent}.",
                self.symbols[f"q_ref_{tp}"]:
                    f"{tp.capitalize()} reference angle of torsojoint: {self.parent}.",
            })
        return desc

    def _define_objects(self) -> None:
        """Define the objects."""
        for tp in ("flexion", "adduction", "rotation"):
            self.symbols.update({name: dynamicsymbols(self._add_prefix(name))
                                 for name in (f"k_{tp}", f"c_{tp}", f"q_ref_{tp}")})

    def _define_loads(self) -> None:
        """Define the loads."""
        torsojoint = self.parent.system.joints[0]
        adduction_axis = (cos(torsojoint.coordinates[0]) * torsojoint.parent_interframe.x -
                          sin(torsojoint.coordinates[0]) * torsojoint.parent_interframe.z)
        if isinstance(self.parent, SacrumBase):
            rot_dir = -1
        else:
            adduction_axis *= -1
            rot_dir = 1
        torques = []
        for i, tp in enumerate(("flexion", "adduction", "rotation")):
            torques.append(-self.symbols[f"k_{tp}"] * (
                    torsojoint.coordinates[i] - self.symbols[f"q_ref_{tp}"]) -
                           self.symbols[f"c_{tp}"] * torsojoint.speeds[i])
        torque = (torques[0] * torsojoint.parent_interframe.y +
                  torques[1] * adduction_axis +
                  torques[2] * rot_dir * torsojoint.child_interframe.z)
        self.parent.system.add_loads(
            Torque(torsojoint.child_interframe, torque),
            Torque(torsojoint.parent_interframe, -torque)
        )

class PinTorsoJoint(SacrumBase):
    """ Connection between pelvis and torso respresented by a pin joint"""

    @property
    def descriptions(self) -> dict[object, str]:
        """Return the descriptions."""
        return {
            **super().descriptions,
            self.q[0]: "adduction angle of the torsojoint.",
            self.u[0]: "adduction angular velocity of the torsojoint.",
            self.symbols["d_p_t"]: "Distance from the torso center of mass to the "
                                   "pelvis.",
            self.symbols["th_h_p"]: "distance from torso center of mass to the thorax/torsojoint joint",
        }

    def _define_objects(self) -> None:
        """Define the objects."""
        super()._define_objects()
        self.q = Matrix([dynamicsymbols(self._add_prefix("q"))])
        self.u = Matrix([dynamicsymbols(self._add_prefix("u"))])
        self._system = System.from_newtonian(self.pelvis.body)
        self._intermediate = Attachment(ReferenceFrame(self._add_prefix("int_frame")),
                                        Point(self._add_prefix("int_point")))
        self.symbols["d_p_t"] = Symbol(self._add_prefix("d_p_t"))
        self.symbols["th_h_p"] = Symbol(self._add_prefix("th_h_p"))
        theta = Symbol(self._add_prefix("theta"))
        self.symbols["theta"] = theta
        self._torso_wrt_pelvis = -self.symbols["d_p_t"] * self.pelvis.z
        self._pelvis_lean_axis = (cos(theta) * self.pelvis.frame.x -
                                  sin(theta) * self.pelvis.frame.z)
        self._torso_lean_axis = self.torso.x
        self._system = System(
            self.pelvis.system.frame, self.pelvis.system.fixed_point)
        self._thorax_wrt_pelvis = - self.symbols["th_h_p"] * self.pelvis.z



    def _define_kinematics(self) -> None:
        """Define the kinematics."""
        super()._define_kinematics()
        self.system.add_joints(
            PinJoint(
                self._add_prefix("joint"), self.pelvis.body, self.torso.body,
                self.q, self.u, parent_point=self.pelvis.pelvis_top_point, child_point=self.torso.torsojoint_point,
                parent_interframe=self.pelvis.frame, child_interframe=self.torso.frame, joint_axis=self._pelvis_lean_axis))

    @property
    def torso_lean_axis(self) -> Vector:
        return self._torso_lean_axis

    @torso_lean_axis.setter
    def torso_lean_axis(self, value: Vector) -> None:
        try:
            value.express(self.pelvis.frame)
        except ValueError as e:
            raise ValueError("Torso lean axis must be expressable in the pelvis frame") from e
        self._torso_lean_axis = value

    @property
    def torso_wrt_pelvis(self) -> Vector:
        """Return the position of the torso w.r.t. the pelvis center of mass."""
        return self._torso_wrt_pelvis

    @torso_wrt_pelvis.setter
    def torso_wrt_pelvis(self, value: Vector) -> None:
        try:
            value.express(self.pelvis.frame)
        except ValueError as e:
            raise ValueError(
                "Torso position must be expressable in the pelvis frame.") from e
        self._torso_wrt_pelvis = value

    @property
    def thorax_wrt_pelvis(self) -> Vector:
        return self._thorax_wrt_pelvis

    @thorax_wrt_pelvis.setter
    def thorax_wrt_pelvis(self, value: Vector) -> None:
        try:
            value.express(self.pelvis.frame)
        except ValueError as e:
            raise ValueError(
                "Thorax position must be expressable in the pelvis frame.") from e
        self._thorax_wrt_pelvis = value

    def get_param_values(self, bicycle_parameters: Bicycle) -> dict[Symbol, float]:
        """Get a parameters mapping of a model based on a bicycle parameters object."""
        params = super().get_param_values(bicycle_parameters)
        human = bicycle_parameters.human
        if human is None:
            return params
        torso_props = human.combine_inertia(("T", "C"))
        params[self.symbols["d_p_t"]] = np.linalg.norm(
            torso_props[1] - human.P.center_of_mass)
        params[self.symbols["th_h_p"]] = np.linalg.norm(
            human.T.pos - human.P.center_of_mass)

        return params

class PinTorsoJointTorque(LoadGroupBase):
    parent: PinTorsoJoint
    required_parent_type = PinTorsoJoint

    @property
    def descriptions(self) -> dict[object, str]:
        """Descriptions of the objects."""
        return {
            **super().descriptions,
            self.symbols["T"]: f"Adduction torque of torso: {self.parent}.",
        }

    def _define_objects(self) -> None:
        """Define the objects."""
        self.symbols["T"] = dynamicsymbols(self._add_prefix("T"))

    def _define_loads(self) -> None:
        """Define the loads."""
        torsojoint = self.parent.system.joints[0]
        adduction_axis = (cos(torsojoint.coordinates[0]) * torsojoint.parent_interframe.x -
                          sin(torsojoint.coordinates[0]) * torsojoint.parent_interframe.z)
        self.parent.system.add_actuators(
            TorqueActuator(self.symbols["T"], adduction_axis,
                           self.parent.pelvis.frame, self.parent.torso.frame))

class PinTorsoJointSpringDamper(LoadGroupBase):
    parent: PinTorsoJoint
    required_parent_type = PinTorsoJoint

    @property
    def descriptions(self) -> dict[object, str]:
        """Descriptions of the objects."""
        return{
            **super().descriptions,
            self.symbols["k"]: f"TorsoJoint stiffness of {self.parent}",
            self.symbols["c"]: f"TorsoJoint damping of {self.parent}",
            self.symbols["q_ref"]: f"TorsoJoint reference angle of {self.parent}",
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
        torsojoint = self.parent.system.joints[0]
        adduction_axis = (cos(torsojoint.coordinates[0]) * torsojoint.parent_interframe.x -
                          sin(torsojoint.coordinates[0]) * torsojoint.parent_interframe.z)
        if isinstance(self.parent, SacrumBase):
            rot_dir = -1
        else:
            adduction_axis *= -1
            rot_dir = 1

        self.parent.system.add_actuators(
            TorqueActuator(
                -self.symbols["k"] * (torsojoint.coordinates[0] - self.symbols["q_ref"]) -
                self.symbols["c"] * torsojoint.speeds[0],
                adduction_axis, self.parent.torso.frame, self.parent.pelvis.frame)
        )

class FixedTorsoJoint(SacrumBase):
    """A fixed connection between the pelvis/lower torso and upper torso"""
    @property
    def descriptions(self) -> dict[object, str]:
        """Return the descriptions."""
        return {
            **super().descriptions,
            self.symbols["yaw"]: "Yaw angle of the torso w.r.t. the pelvis.",
            self.symbols["pitch"]: "Pitch angle of the torso w.r.t. the pelvis.",
            self.symbols["roll"]: "Roll angle of the torso w.r.t. the pelvis.",
            self.symbols["d_p_t"]: "Distance from the torso center of mass to the pelvis.",
            self.symbols["th_h_p"]: "distance from torso center of mass to the thorax/torsojoint joint",
        }

    def _define_objects(self) -> None:
        """Define the objects."""
        super()._define_objects()
        self._system = System.from_newtonian(self.pelvis.body)
        self.symbols.update({
            name: Symbol(self._add_prefix(name)) for name in ("yaw", "pitch", "roll")})
        self._intermediate = Attachment(ReferenceFrame(self._add_prefix("int_frame")),
                                        Point(self._add_prefix("int_point")))
        self.symbols["d_p_t"] = Symbol(self._add_prefix("d_p_t"))
        self.symbols["th_h_p"] = Symbol(self._add_prefix("th_h_p"))
        self._torso_wrt_pelvis = - self.symbols["d_p_t"] * self.pelvis.z
        self._system = System(
            self.pelvis.system.frame, self.pelvis.system.fixed_point)
        self._thorax_wrt_pelvis = - self.symbols["th_h_p"] * self.pelvis.z

    def define_kinematics(self) -> None:
        """Define the kinematics."""
        super()._define_kinematics()
        self.system.add_joints(WeldJoint(
            self._add_prefix("weld_joint"), self.pelvis.body, self.torso.body,
            parent_point=self.pelvis.pelvis_top_point, child_point=self.torso.torsojoint_point,
            parent_interframe=self.pelvis.frame, child_interframe=self.torso.frame))

    @property
    def torso_wrt_pelvis(self) -> Vector:
        """Return the position of the torso w.r.t. the pelvis center of mass."""
        return self._torso_wrt_pelvis

    @torso_wrt_pelvis.setter
    def torso_wrt_pelvis(self, value: Vector) -> None:
        try:
            value.express(self.pelvis.frame)
        except ValueError as e:
            raise ValueError(
                "Torso position must be expressable in the pelvis frame.") from e
        self._torso_wrt_pelvis = value

    @property
    def thorax_wrt_pelvis(self) -> Vector:
        return self._thorax_wrt_pelvis

    @thorax_wrt_pelvis.setter
    def thorax_wrt_pelvis(self, value: Vector) -> None:
        try:
            value.express(self.pelvis.frame)
        except ValueError as e:
            raise ValueError(
                "Thorax position must be expressable in the pelvis frame.") from e
        self._thorax_wrt_pelvis = value

    def get_param_values(self, bicycle_parameters: Bicycle) -> dict[Symbol, float]:
        """Get a parameters mapping of a model based on a bicycle parameters object."""
        params = super().get_param_values(bicycle_parameters)
        human = bicycle_parameters.human
        if human is None:
            return params
        torso_props = human.combine_inertia(("T", "C"))
        params[self.symbols["d_p_t"]] = np.linalg.norm(
            torso_props[1] - human.P.center_of_mass)
        params[self.symbols["th_h_p"]] = np.linalg.norm(
            human.T.pos - human.P.center_of_mass)

        return params