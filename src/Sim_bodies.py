from __future__ import annotations
from symbrim.rider import PlanarPelvis, PelvisBase, TorsoBase
from symbrim.brim import PelvisInterPointMixin
from src.SimBodyBase import InterSeatBase, HeadBase, SimPelvisBase, SimTorsoBase
import contextlib
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    with contextlib.suppress(ImportError):
        from symbrim.utilities.plotting import PlotModel
        from symbrim.utilities.parametrize import get_inertia_vals_from_yeadon
        from bicycleparameters import Bicycle

from abc import abstractmethod
from sympy import Matrix, Symbol
from sympy.physics.mechanics import (
    PinJoint,
    Point,
    Vector,
    ReferenceFrame,
    RigidBody,
    System,
    TorqueActuator,
    dynamicsymbols,
)

from sympy import Symbol
from sympy.physics.mechanics import Point

from symbrim.core import ModelBase, NewtonianBodyMixin
with contextlib.suppress(ImportError):
    import numpy as np
    from yeadon.inertia import rotate_inertia, rotate3_inertia

    from symbrim.utilities.parametrize import get_inertia_vals_from_yeadon

    if TYPE_CHECKING:
        from bicycleparameters import Bicycle

__all__ = ["SimPelvis", "SimTorso", "SimHead", "SimInterSeat"]

class SimPelvis(PelvisBase):
    """mixin class for a single pendulum pelvis body, i don't even know why i need to do this but okay.
    """

    @property
    def descriptions(self) -> dict[object, str]:
        """Descriptions of the objects."""
        return {
            **super().descriptions,
            self.symbols["com_height"]: "com_height",
            self.symbols["hip_width"]: "hip_width",
            self.symbols["thorax_height"]: "thorax_height"
        }

    @property
    def pelvis_top_point(self) -> Point:
        """ location of the top point of the hip, just for the single pendulum purpose.
        A torsojoint can be connected to this point later on perhapsss """
        return self._pelvis_top_point

    @property
    def left_hip_point(self) -> Point:
        """Location of the left hip.

        Explanation
        -----------
        The left hip point is defined as the point where the left hip joint is located.
        This point is used by connections to connect the left leg to the pelvis.
        """
        return self._left_hip_point

    @property
    def right_hip_point(self) -> Point:
        """Location of the right hip.

        Explanation
        -----------
        The right hip point is defined as the point where the right hip joint is
        located. This point is used by connections to connect the right leg to the
        pelvis.
        """
        return self._right_hip_point

    def _define_objects(self) -> None:
        """Define the objects."""
        super()._define_objects()
        self._pelvis_top_point = Point(self._add_prefix("PTP"))
        self.symbols["com_height"] = Symbol(self._add_prefix("com_height"))
        self.symbols["hip_width"] = Symbol(self._add_prefix("hip_width"))
        self.symbols["thorax_height"] = Symbol(self._add_prefix("thorax_height"))

    def _define_kinematics(self) -> None:
        """Define the kinematics."""
        super()._define_kinematics()
        #h_p = (self.symbols[name] for name in ("h_pelvis"))
        #self.pelvis.masscenter.set_pos(self.),
#        self.hip_top_point.set_pos(self.body.masscenter,
#                                    2 * self.symbols["com_height"] * self.z)
        self.left_hip_point.set_pos(self.body.masscenter,
                                    - self.symbols["hip_width"] * self.y / 2 +
                                    self.symbols["com_height"] * self.z)
        self.right_hip_point.set_pos(self.body.masscenter,
                                     self.symbols["hip_width"] * self.y / 2 +
                                     self.symbols["com_height"] * self.z)
        self.pelvis_top_point.set_pos(self.body.masscenter,
                                      - (self.symbols["thorax_height"] - self.symbols["com_height"]) * self.z)

    def get_param_values(self, bicycle_parameters: Bicycle) -> dict[Symbol, float]:
        """Get the parameter values of the pelvis."""
        params = super().get_param_values(bicycle_parameters)
        human = bicycle_parameters.human
        if human is None:
            return params
        params.update(get_inertia_vals_from_yeadon(self.body, human.P.rel_inertia))
        params[self.symbols["com_height"]] = human.P.rel_center_of_mass[2, 0]
        params[self.symbols["hip_width"]] = np.linalg.norm(human.J1.pos - human.K1.pos)
        params[self.symbols["thorax_height"]] = np.linalg.norm(human.T.pos)
        return params

    def set_plot_objects(self, plot_object: PlotModel) -> None:
        """Set the symmeplot plot objects."""
        super().set_plot_objects(plot_object)
        plot_object.add_line([
            self.left_hip_point, self.body.masscenter, self.right_hip_point,
            self.left_hip_point], self.name)

class SimTorso(TorsoBase):
    """A planar rigid torso.

       Explanation
       -----------
       This is different from the PlanarTorso as it enables the torso as a whole to rotate around the torsojoint joint,
       instead of it just being the shoulders that move

       """

    @property
    @abstractmethod
    def neck_frame(self) -> ReferenceFrame:
        """" The frame of the neck, and thus essentially the head aswell"""

    @property
    def descriptions(self) -> dict[object, str]:
        """Descriptions of the objects."""
        return {
            **super().descriptions,
            self.symbols["shoulder_width"]: "Distance between the left and right "
                                            "shoulder joints.",
            self.symbols["shoulder_height"]: "Distance between the shoulder joints and "
                                             "center of mass of the the torso.",
            self.symbols["torsojoint_shoulder_length"]: "Distance from thorax to shoulder height",
            self.symbols["shoulder_neck_length"]: "Distance from shoulder to neck rotation point" # Ls6L
        }

    def _define_objects(self) -> None:
        """Define the objects."""
        super()._define_objects()
        self._torsojoint_point = Point(self._add_prefix("SP"))
        self._torso_neck_point = Point(self._add_prefix("TNP"))
        self.symbols["shoulder_width"] = Symbol(self._add_prefix("shoulder_width"))
        self.symbols["shoulder_height"] = Symbol(self._add_prefix("shoulder_height"))
        self.symbols["torsojoint_shoulder_length"] = Symbol(self._add_prefix("torsojoint_shoulder_length"))
        self.symbols["shoulder_neck_length"] = Symbol(self._add_prefix("shoulder_neck_length"))


    def _define_kinematics(self) -> None:
        """Define the kinematics."""
        super()._define_kinematics()
        w, h, l, s = (self.symbols["shoulder_width"], self.symbols["shoulder_height"],
                      self.symbols["torsojoint_shoulder_length"], self.symbols["shoulder_neck_length"])
        self.left_shoulder_point.set_pos(self.body.masscenter,
                                         -w / 2 * self.y - h * self.z)
        self.right_shoulder_point.set_pos(self.body.masscenter,
                                          w / 2 * self.y - h * self.z)
        self.torsojoint_point.set_pos(self.body.masscenter, + l * self.z)
        self.torso_neck_point.set_pos(self.body.masscenter, (h + s) * self.z)  #Looks wrong here should be - (h+s)

    #@property
    #def torsojoint_frame(self) -> ReferenceFrame:
    #    return self.body.frame

    def neck_frame(self) -> ReferenceFrame:
        return self.body.frame
    @property
    def left_shoulder_frame(self) -> ReferenceFrame:
        """The left shoulder frame."""
        return self.body.frame

    @property
    def right_shoulder_frame(self) -> ReferenceFrame:
        """The right shoulder frame."""
        return self.body.frame

    @property
    def torsojoint_point(self) -> Point:

        return self._torsojoint_point

    @property
    def torso_neck_point(self) -> Point:

        return self._torso_neck_point

    def set_plot_objects(self, plot_object: PlotModel) -> None:
        """ override symmeplot from the TorsoBase if possible"""
        super().set_plot_objects(plot_object)
        plot_object.add_line([
            self.body.masscenter, self.torsojoint_point, self.right_shoulder_point, self.left_shoulder_point,
            self.torsojoint_point], self.name)
        plot_object.add_line([
            self.body.masscenter, self.torso_neck_point], self.name)

    def get_param_values(self, bicycle_parameters: Bicycle) -> dict[Symbol, float]:
        """Get the parameter values of the torso."""
        params = super().get_param_values(bicycle_parameters)
        human = bicycle_parameters.human
        if human is None:
            return params
        C2 = human.C.inertia / 2
        torso_props = human.combine_inertia(("T", "s4", "s5", "A1", "A2", "B1", "B2"))
        torso_propz = human.combine_inertia(("T", "C"))
        params.update(get_inertia_vals_from_yeadon(
            self.body, rotate_inertia(human.T.rot_mat, (torso_props[2]))))
        print(torso_props)
        # + 2 * (torso_props[2] + torso_props[3])))))
        params[self.symbols["shoulder_height"]] = np.linalg.norm(
            (human.A1.pos + human.B1.pos) / 2 - torso_propz[1])
        params[self.symbols["shoulder_width"]] = np.linalg.norm(
            human.A1.pos - human.B1.pos)
        params[self.symbols["torsojoint_shoulder_length"]] = np.linalg.norm(
            (human.A1.pos + human.B1.pos) / 2 - human.T.pos)
        params[self.symbols["shoulder_neck_length"]] = np.linalg.norm(
            (human.meas["Ls5L"] - human.meas["Ls4L"]))

        return params

class SimHead(HeadBase):


    @property
    def descriptions(self) -> dict[object, str]:
        """Descriptions of the objects."""

        return {
            **super().descriptions,
            self.symbols["head_width"]: "Distance between the left and right ears.",
            self.symbols["head_neck_height"]: "Distance between the neck joint and "
                                              "center of mass of the the head.",
            self.symbols["head_top_point"]: "top point of the head, speaks for itself innit"
        }

    def _define_objects(self) -> None:
        """Define the objects."""
        super()._define_objects()
        self.symbols["head_width"] = Symbol(self._add_prefix("head_width"))
        self.symbols["head_neck_height"] = Symbol(self._add_prefix("head_neck_height"))
        self.symbols["head_top_point"] = Symbol(self._add_prefix("head_top_point"))

    def _define_kinematics(self) -> None:
        """Define the kinematics."""
        super()._define_kinematics()
        w, h, t = self.symbols["head_width"], self.symbols["head_neck_height"], self.symbols["head_top_point"]
        self.left_ear.set_pos(self.body.masscenter, -(w / 2) * self.y)# - h * self.z)
        self.right_ear.set_pos(self.body.masscenter, (w / 2) * self.y) #- h * self.z)
        self.neck_point.set_pos(self.body.masscenter, h * self.z)
        self.head_top_point.set_pos(self.body.masscenter, - t * self.z)
        self.imu_point.set_pos(self.body.masscenter, 0 * self.z) #Assuming the IMU is located at the head center of mass


    def get_param_values(self, bicycle_parameters: Bicycle) -> dict[Symbol, float]:
        """Get the parameter values of the head."""
        params = super().get_param_values(bicycle_parameters)
        human = bicycle_parameters.human
        if human is None:
            return params
        torso_props = human.combine_inertia(("s6", "s7"))
        params.update(get_inertia_vals_from_yeadon(
            self.body, rotate_inertia(human.T.rot_mat, torso_props[2])))
        params[self.symbols["head_neck_height"]] = np.linalg.norm(human.meas["Ls6L"])
        params[self.symbols["head_width"]] = np.linalg.norm(human.meas["Ls7p"] / np.pi)
        params[self.symbols["head_top_point"]] = np.linalg.norm(human.meas["Ls8L"] - human.meas["Ls6L"])
        return params

    def set_plot_objects(self, plot_object: PlotModel) -> None:
        """Set the symmeplot plot objects."""
        super().set_plot_objects(plot_object)
        plot_object.add_line([
            self.body.masscenter, self.neck_point], self.name)
        plot_object.add_line([
            self.body.masscenter, self.left_ear, self.right_ear, self.body.masscenter], self.name)
        plot_object.add_line([
            self.left_ear, self.head_top_point, self.right_ear], self.name)


    """
    copied this from the planartorso for comparison reasonz
        def get_param_values(self, bicycle_parameters: Bicycle) -> dict[Symbol, float]:
        ""Get the parameter values of the pelvis.""
        params = super().get_param_values(bicycle_parameters)
        human = bicycle_parameters.human
        if human is None:
            return params
        torso_props = human.combine_inertia(("T", "C"))
        params.update(get_inertia_vals_from_yeadon(
            self.body, rotate_inertia(human.T.rot_mat, torso_props[2])))
        params[self.symbols["shoulder_height"]] = np.linalg.norm(
            (human.A1.pos + human.B1.pos) / 2 - torso_props[1])
        params[self.symbols["shoulder_width"]] = np.linalg.norm(
            human.A1.pos - human.B1.pos)
        return params
    """

class SimInterSeat(InterSeatBase):
    @property
    def descriptions(self) -> dict[object, str]:
        """Descriptions of the objects."""

        return {
            **super().descriptions,
            self.symbols["interseat"]: "Distance between the saddle and pelvis."}

    def _define_objects(self) -> None:
        """Define the objects."""
        super()._define_objects()
        self.symbols["interseat"] = Symbol(self._add_prefix("interseat"))


    def _define_kinematics(self) -> None:
        """Define the kinematics."""
        super()._define_kinematics()
        l = self.symbols["interseat"]
        self.inter_saddle_point.set_pos(self.body.masscenter, (l / 2) * self.z)
        self.inter_pelvis_point.set_pos(self.body.masscenter, -(l / 2) * self.z)

    def get_param_values(self, bicycle_parameters: Bicycle) -> dict[Symbol, float]:
        """Get the parameter values of the pelvis."""
        params = super().get_param_values(bicycle_parameters)
        human = bicycle_parameters.human
        params[self.symbols["interseat"]] = np.linalg.norm(0.001) ## should basically have zero length
        params.update(get_inertia_vals_from_yeadon(self.body, inertia=0.0001*human.T.rot_mat))
            #get_inertia_vals_from_yeadon(
            #self.body, rotate_inertia(human.T.rot_mat, inertia=inertia_val)))
        return params

    def set_plot_objects(self, plot_object: PlotModel) -> None:
        """Set the symmeplot plot objects."""
        super().set_plot_objects(plot_object)
        plot_object.add_line([
            self.body.masscenter, self.inter_saddle_point, self.inter_pelvis_point], self.name)







