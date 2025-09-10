from __future__ import annotations

import contextlib
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from sympy import Symbol, symbols
from sympy.physics.mechanics import Point, RigidBody, System, inertia

from symbrim.core import Attachment, Hub, ModelBase, set_default_convention
from symbrim.bicycle import RearFrameBase
with contextlib.suppress(ImportError):
    import numpy as np
    from bicycleparameters.io import remove_uncertainties
    from bicycleparameters.main import calculate_benchmark_from_measured
    from bicycleparameters.rider import yeadon_vec_to_bicycle_vec
    from dtk.bicycle import benchmark_to_moore

    from symbrim.utilities.parametrize import get_inertia_vals

    if TYPE_CHECKING:
        from bicycleparameters import Bicycle

if TYPE_CHECKING:
    with contextlib.suppress(ImportError):
        from symbrim.utilities.plotting import PlotModel

__all__ = ["SimRearFrame"]

class SimRearFrame(RearFrameBase):
    """Rigid rear frame."""

    @property
    def descriptions(self) -> dict[Any, str]:
        """Dictionary of descriptions of the rear frame's symbols."""
        return {
            **super().descriptions,
            self.symbols["l_bbx"]: f"Distance between the rear hub and the bottom "
                                   f"bracket along {self.body.x}.",
            self.symbols["l_bbz"]: f"Distance between the rear hub and the bottom "
                                   f"bracket along {self.body.z}.",
        }

    def _define_objects(self):
        """Define the objects of the rear frame."""
        super()._define_objects()
        self._body = RigidBody(self._add_prefix("body"))
        self.body.central_inertia = inertia(self.body.frame,
                                            *symbols(self._add_prefix("ixx iyy izz")),
                                            izx=Symbol(self._add_prefix("izx")))
        self._system = System.from_newtonian(self.body)
        self._saddle = Attachment(self.body.frame, Point(self._add_prefix("saddle")))
        self._saddle_axis = self.body.x
        self._bottom_bracket = Point(self._add_prefix("bottom_bracket"))
        self.symbols.update({name: Symbol(self._add_prefix(name))
                             for name in ("l_bbx", "l_bbz")})

    def _define_kinematics(self):
        """Define the kinematics of the rear frame."""
        super()._define_kinematics()
        self.bottom_bracket.set_pos(self.wheel_hub.point,
                                    self.symbols["l_bbx"] * self.body.x +
                                    self.symbols["l_bbz"] * self.body.z)
        self.bottom_bracket.set_vel(self.body.frame, 0)

    @property
    def body(self) -> RigidBody:
        """Rigid body representing the rear frame."""
        return self._body

    @property
    def saddle(self) -> Attachment:
        """Attachment representing the saddle."""
        return self._saddle

    @property
    def bottom_bracket(self) -> Point:
        """Point representing the center of the bottom bracket."""
        return self._bottom_bracket

    def get_param_values(self, bicycle_parameters: Bicycle) -> dict[Symbol, float]:
        """Get a parameters mapping of a model based on a bicycle parameters object."""
        params = super().get_param_values(bicycle_parameters)
        if "Benchmark" in bicycle_parameters.parameters:
            bp = remove_uncertainties(bicycle_parameters.parameters["Benchmark"])
            params[self.body.mass] = bp["mB"]
        if "Measured" in bicycle_parameters.parameters:
            mep = remove_uncertainties(bicycle_parameters.parameters["Measured"])
            rr, lcs, hbb, lamht = (mep.get(name) for name in (
                "rR", "lcs", "hbb", "lamht"))
            if "mB" in mep:
                params[self.body.mass] = mep["mB"]
            if rr is None and "Benchmark" in bicycle_parameters.parameters:
                rr = bp["rR"]
            if lamht is None and "Benchmark" in bicycle_parameters.parameters:
                lamht = np.pi / 2 - bp["lam"]
            if not any(value is None for value in (rr, lcs, hbb, lamht)):
                glob_z = rr - hbb
                glob_x = np.sqrt(lcs ** 2 - glob_z ** 2)
                params[self.symbols["l_bbx"]] = (
                        glob_x * np.sin(lamht) - glob_z * np.cos(lamht))
                params[self.symbols["l_bbz"]] = (
                        glob_x * np.cos(lamht) + glob_z * np.sin(lamht))
        return params