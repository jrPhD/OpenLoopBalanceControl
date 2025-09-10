from __future__ import annotations
import sympy as sm
import sympy.physics.mechanics as me

from sympy.physics.mechanics import System

from symbrim.bicycle import BicycleBase
from symbrim.brim.base_connections import SeatBase, PedalsBase

from symbrim.core import ConnectionRequirement, ModelBase, ModelRequirement

from symbrim.rider import Rider
from src.simrider import SimRider
from src.SimJointBase import InterSeatJointBase


__all__ = ["SimBicycleRider"]

class SimBicycleRider(ModelBase,): #, ConnectionRequirement):
    """Model of a bicycle and a simple pendulum rider."""

    required_models: tuple[ModelRequirement] = (
        ModelRequirement("bicycle", BicycleBase, "Bicycle model."),
        ModelRequirement("rider", SimRider, "Pendulum Rider model.", False),
    )
    required_connections: tuple[ConnectionRequirement] = (
        ConnectionRequirement(
            "seat", (SeatBase, InterSeatJointBase),
            "Connection between the rider (pelvis or interseat) and the rear frame."),
        ConnectionRequirement("pedals", PedalsBase,
            "Connection between riders feet and bicycle", False)
    )
    bicycle: BicycleBase
    rider: SimRider
    seat: SeatBase | InterSeatJointBase



    def _define_connections(self) -> None:
        """Define the connections."""
        super()._define_connections()
        if self.seat is not None:
            self.seat.rear_frame = self.bicycle.rear_frame
            self.seat.pelvis = self.rider.pelvis
            self.seat.interseat = self.rider.interseat


    def _define_objects(self) -> None:
        """Define the objects."""
        super()._define_objects()
        self._system = System(
            self.bicycle.system.frame, self.bicycle.system.fixed_point)
        if self.seat is not None:
            self.seat.define_objects()

    def _define_kinematics(self) -> None:
        """Define the kinematics."""
        super()._define_kinematics()
        if self.seat is not None:
            self.seat.define_kinematics()

    def _define_loads(self) -> None:
        """Define the loads."""
        super()._define_loads()
        if self.seat is not None:
            self.seat.define_loads()

    def _define_constraints(self) -> None:
        """Define the constraints."""
        super()._define_constraints()
        if self.seat is not None:
            self.seat.define_constraints()