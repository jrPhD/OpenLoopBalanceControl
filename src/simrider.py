"""Module containing stuff for a composable rider"""
from __future__ import annotations

from sympy.physics.mechanics import System

from symbrim.core import ConnectionRequirement, ModelBase, ModelRequirement
from symbrim.rider.base_connections import LeftHipBase, RightHipBase, SacrumBase
from symbrim.rider.legs import LeftLegBase, RightLegBase
from symbrim.rider.pelvis import PelvisBase
from symbrim.rider.torso import TorsoBase
from src.SimJointBase import NeckBase, ShiftingSideLeanSeatBase
from src.SimBodyBase import HeadBase, InterSeatBase
__all__ = ["SimRider"]


class SimRider(ModelBase):
    """Customizable rider model."""

    required_models: tuple[ModelRequirement, ...] = (
        ModelRequirement("pelvis", PelvisBase, "Pelvis of the rider."),
        ModelRequirement("torso", TorsoBase, "Torso of the rider.", False),
        ModelRequirement("left_leg", LeftLegBase, "Left leg of the rider.", False),
        ModelRequirement("right_leg", RightLegBase, "Right leg of the rider.", False),
        ModelRequirement("head", HeadBase, "head of the rider.", False),
        ModelRequirement("interseat", InterSeatBase, "Seat model to enable shifting-lean seat", False)
    )
    required_connections: tuple[ConnectionRequirement] = (
        ConnectionRequirement("torsojoint", SacrumBase,
                              "Connection between the pelvis and the torso.", False),
        ConnectionRequirement("left_hip", LeftHipBase,
                              "Connection between the pelvis and the left leg.", False),
        ConnectionRequirement("right_hip", RightHipBase,
                              "Connection between the pelvis and the right leg.", False),
        ConnectionRequirement("neck", NeckBase,
                              "Connection between the torso and the head.", False),
        ConnectionRequirement("shiftingsideleanseat", ShiftingSideLeanSeatBase,
                              "Connection between the pelvis and the interseat part to enable shifting-lean seat", False)
    )
    pelvis: PelvisBase
    torso: TorsoBase
    left_leg: LeftLegBase
    right_leg: RightLegBase
    head: HeadBase
    interseat: InterSeatBase
    torsojoint: SacrumBase
    left_hip: LeftHipBase
    right_hip: RightHipBase
    neck: NeckBase
    shiftingsideleanseat: ShiftingSideLeanSeatBase

    def _define_connections(self) -> None:
        """Define the connections."""
        super()._define_connections()
        if self.torsojoint:
            self.torsojoint.pelvis = self.pelvis
            self.torsojoint.torso = self.torso
        if self.neck:
            self.neck.torso = self.torso
            self.neck.head = self.head
        if self.left_hip:
            self.left_hip.pelvis = self.pelvis
            self.left_hip.leg = self.left_leg
        if self.right_hip:
            self.right_hip.pelvis = self.pelvis
            self.right_hip.leg = self.right_leg
        if self.shiftingsideleanseat:
            self.shiftingsideleanseat.interseat = self.interseat
            self.shiftingsideleanseat.pelvis = self.pelvis

    def _define_objects(self) -> None:
        """Define the objects."""
        super()._define_objects()
        self._system = System.from_newtonian(self.pelvis.body)
        for conn in self.connections:
            conn.define_objects()

    def _define_kinematics(self) -> None:
        """Define the kinematics."""
        super()._define_kinematics()
        for conn in self.connections:
            conn.define_kinematics()

    def _define_loads(self) -> None:
        """Define the loads."""
        super()._define_loads()
        for conn in self.connections:
            conn.define_loads()

    def _define_constraints(self) -> None:
        """Define the constraints."""
        super()._define_constraints()
        for conn in self.connections:
            conn.define_constraints()




