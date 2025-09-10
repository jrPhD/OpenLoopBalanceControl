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
from symbrim.core import ConnectionBase, ModelRequirement, ModelBase
from symbrim.brim import SeatBase
from symbrim.rider import TorsoBase, PelvisBase
from src.SimBodyBase import HeadBase, InterSeatBase, SimPelvisBase, SimTorsoBase
from symbrim.bicycle import RearFrameBase


__all__ = ["NeckBase", "InterSeatJointBase", "ShiftingSideLeanSeatBase"]

class NeckBase(ConnectionBase):
    """Base class for the neck joint."""

    required_models: tuple[ModelRequirement, ...] = (
        ModelRequirement("torso", TorsoBase, "Torso of the rider."),
        ModelRequirement("head", HeadBase, "Head of the rider."),
    )
    torso: TorsoBase
    head: HeadBase

class SimTorsoJointBase(ConnectionBase):
    """Base class for the connection between the pelvis and the torso."""

    required_models: tuple[ModelRequirement, ...] = (
        ModelRequirement("pelvis", (PelvisBase, SimPelvisBase), "Pelvis of the rider."),
        ModelRequirement("torso", (TorsoBase, SimTorsoBase), "Torso of the rider."),
    )
    pelvis: PelvisBase | SimPelvisBase
    torso: TorsoBase | SimTorsoBase

    def set_plot_objects(self, plot_object: PlotModel) -> None:
        """Set the symmeplot plot objects."""
        super().set_plot_objects(plot_object)
        plot_object.add_line([
            self.pelvis.body.masscenter,
            *(joint.parent_point for joint in self.system.joints),
            self.torso.body.masscenter],
            self.name)

class InterSeatJointBase(ConnectionBase):
    """Base class for the connection between the saddle and the interseat part."""

    required_models: tuple[ModelRequirement, ...] = (
        ModelRequirement("rear_frame", RearFrameBase, "Rear frame of the bicycle on which the seat sits."),
        ModelRequirement("interseat", InterSeatBase, "Interseat model to enable the shifting sidelean seat."),
    )
    rear_frame: RearFrameBase
    interseat: InterSeatBase

class ShiftingSideLeanSeatBase(ConnectionBase):
    """ Base class for the connection between the interseat part and the pelvis."""

    required_models: tuple[ModelRequirement, ...] = (
        ModelRequirement("interseat", InterSeatBase, "Interseat model to enable the shifting sidelean seat."),
        ModelRequirement("pelvis", PelvisBase, "Pelvis of the rider."),

    )
    interseat: InterSeatBase
    pelvis: PelvisBase