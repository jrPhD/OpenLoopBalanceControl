from __future__ import annotations

import contextlib
from abc import abstractmethod
from typing import TYPE_CHECKING
from symbrim.core import NewtonianBodyMixin
with contextlib.suppress(ImportError):
    import numpy as np
    from yeadon.inertia import rotate_inertia
    if TYPE_CHECKING:
        from bicycleparameters import Bicycle

        with contextlib.suppress(ImportError):
            from symbrim.utilities.plotting import PlotModel

    from symbrim.utilities.parametrize import get_inertia_vals_from_yeadon

from sympy import Symbol
from sympy.physics.mechanics import Point, ReferenceFrame
from symbrim.core import ModelRequirement, ModelBase




__all__ = ["HeadBase", "InterSeatBase", "SimPelvisBase", "SimTorsoBase"]

class HeadBase(NewtonianBodyMixin, ModelBase):
    """ Base class for the head of a rider"""

    @property
    def neck_point(self) -> Point:
        """Location of the neck.

        Explanation
        -----------
        The neck point is defined as the point where the left hip joint is located.
        This point is used by connections to connect the left leg to the pelvis.
        """
        return self._neck_point

    @property
    def left_ear(self) -> Point:
        """Location of the left ear.
        """
        return self._left_ear

    @property
    def right_ear(self) -> Point:
        """Location of the right ear.
                """
        return self._right_ear

    @property
    def head_top_point(self) -> Point:

        return self._head_top_point

    def _define_objects(self) -> None:
        """Define the objects."""
        super()._define_objects()
        self._left_ear = Point(self._add_prefix("LE"))
        self._right_ear = Point(self._add_prefix("RE"))
        self._neck_point = Point(self._add_prefix("NP"))
        self._head_top_point = Point(self._add_prefix("TP"))
    def get_param_values(self, bicycle_parameters: Bicycle) -> dict[Symbol, float]:
        """Get the parameter values of the head."""
        params = super().get_param_values(bicycle_parameters)
        human = bicycle_parameters.human
        if human is None:
            return params
        params[self.body.mass] = 0.5 * human.C.mass
        return params

    def set_plot_objects(self, plot_object: PlotModel) -> None:
        """Set the symmeplot plot objects."""
        super().set_plot_objects(plot_object)
        plot_object.add_line([
            self.body.masscenter, self.neck_point], self.name)
        plot_object.add_line([
            self.body.masscenter, self.left_ear, self.right_ear, self.body.masscenter], self.name)

class SimPelvisBase(NewtonianBodyMixin, ModelBase):
    """Base class for the pelvis of a rider."""

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

    @property
    def pelvis_top_point(self) -> Point:
        """ location of the top point of the hip, just for the single pendulum purpose.
        A sacrum can be connected to this point later on perhapsss """
        return self._pelvis_top_point


    def _define_objects(self) -> None:
        """Define the objects."""
        super()._define_objects()
        self._left_hip_point = Point(self._add_prefix("LHP"))
        self._right_hip_point = Point(self._add_prefix("RHP"))
        self._pelvis_top_point = Point(self._add_prefix("PTP"))

    def get_param_values(self, bicycle_parameters: Bicycle) -> dict[Symbol, float]:
        """Get the parameter values of the pelvis."""
        params = super().get_param_values(bicycle_parameters)
        human = bicycle_parameters.human
        if human is None:
            return params
        params[self.body.mass] = human.P.mass
        return params

    def set_plot_objects(self, plot_object: PlotModel) -> None:
        """Set the symmeplot plot objects."""
        super().set_plot_objects(plot_object)
        plot_object.add_line([
            self.left_hip_point, self.body.masscenter, self.right_hip_point,
            self.left_hip_point], self.name)
        #plot_object.add_line([self.body.masscenter, self.pelvis_top_point], self.name)

class SimTorsoBase(NewtonianBodyMixin, ModelBase):
    """Base class for the torso of a rider."""

    @property
    def left_shoulder_point(self) -> Point:
        """Location of the left shoulder.

        Explanation
        -----------
        The left shoulder point is defined as the point where the left shoulder joint
        is located. This point is used by connections to connect the left arm to the
        torso.
        """
        return self._left_shoulder_point

    @property
    @abstractmethod
    def left_shoulder_frame(self) -> ReferenceFrame:
        """The left shoulder frame.

        Explanation
        -----------
        The left shoulder frame is defined as the frame that is attached to the left
        shoulder point. This frame is used by connections to connect the left arm to
        the torso.
        """

    @property
    def right_shoulder_point(self) -> Point:
        """Location of the right shoulder.

        Explanation
        -----------
        The right shoulder point is defined as the point where the right shoulder joint
        is located. This point is used by connections to connect the right arm to the
        torso.
        """
        return self._right_shoulder_point

    @property
    @abstractmethod
    def right_shoulder_frame(self) -> ReferenceFrame:
        """The right shoulder frame.

        Explanation
        -----------
        The right shoulder frame is defined as the frame that is attached to the right
        shoulder point. This frame is used by connections to connect the right arm to
        the torso.
        """
    """
    @property
    def sacrum_point(self) -> Point:

        return self._sacrum_point


    @property
    def torso_neck_point(self) -> Point:

        return self._torso_neck_point
    """
    @property
    @abstractmethod
    def neck_frame(self) -> ReferenceFrame:
        """" The frame of the neck, and thus essentially the head aswell"""

    def _define_objects(self) -> None:
        """Define the objects."""
        super()._define_objects()
        self._left_shoulder_point = Point(self._add_prefix("LSP"))
        self._right_shoulder_point = Point(self._add_prefix("RSP"))
        self._torsojoint_point = Point(self._add_prefix("SP"))
        self._torso_neck_point = Point(self._add_prefix("TNP"))

    def get_param_values(self, bicycle_parameters: Bicycle) -> dict[Symbol, float]:
        """Get the parameter values of the pelvis."""
        params = super().get_param_values(bicycle_parameters)
        human = bicycle_parameters.human
        if human is None:
            return params
        params[self.body.mass] = human.T.mass + (0.5 * human.C.mass) + 2 * (human.A1.mass + human.A2.mass)
        return params

    def set_plot_objects(self, plot_object: PlotModel) -> None:
        """Set the symmeplot plot objects."""
        super().set_plot_objects(plot_object)
        plot_object.add_line([
            self.body.masscenter, self.left_shoulder_point, self.right_shoulder_point,
            self.body.masscenter], self.name)

class InterSeatBase(NewtonianBodyMixin, ModelBase):
    """ This is a model used to enable the sidelean-shifting mechanism.
        This submodel will basically consist of 2 points, which is connected to the saddle via a prismatic joint,
        and the pelvis connects to this via a pin joint"""
    @property
    def inter_saddle_point(self) -> Point:
        """Location of the left ear.
        """
        return self._inter_saddle_point

    @property
    def inter_pelvis_point(self) -> Point:
        """Location of the right ear.
                """
        return self._inter_pelvis_point

    def _define_objects(self) -> None:
        """Define the objects."""
        super()._define_objects()
        self._inter_saddle_point = Point(self._add_prefix("ISP"))
        self._inter_pelvis_point = Point(self._add_prefix("IPP"))

    def get_param_values(self, bicycle_parameters: Bicycle) -> dict[Symbol, float]:
        """Get the parameter values of the pelvis."""
        params = super().get_param_values(bicycle_parameters)
        human = bicycle_parameters.human
        if human is None:
            return params
        params[self.body.mass] = 0
        return params

