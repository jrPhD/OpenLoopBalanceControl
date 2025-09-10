from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto, unique
# from simrider import SimRider
# from sim_bicycle_rider import SimBicycleRider
import symbrim as bm
import numpy as np
import numpy.typing as npt
import sympy as sm
import sympy.physics.mechanics as me
from opty.direct_collocation import Problem
from src.Optimization.opt_simulator import Simulator

@unique
class SteerWith(Enum):
    """Enumeration of options for controlling the bicycle steering."""
    PEDAL_STEER_TORQUE = auto()
    HUMAN_TORQUE = auto()
    SEAT_TORQUE = auto()
    TORSO_TORQUE = auto()
    SEAT_AND_TORSO_TORQUE = auto()
    LEG_TORQUE = auto()
    UPPER_BODY_TORQUE = auto()
    ALL_TORQUES = auto()

@unique
class SeatType(Enum):
    """Enumeration of options for the seat joint."""
    NONE = auto()
    SPHERICAL = auto()
    SIDELEAN = auto()
    SHIFTINGSIDELEAN = auto()
    FIXED = auto()

@unique
class TorsoType(Enum):
    """Enumeration of options for the seat joint."""
    SPHERICAL = auto()
    PIN = auto()
    FIXED = auto()
    

@unique
class Task(Enum):
    """To set the task that will be executed by the model."""
    LANE_SWITCH = auto()
    DOUBLE_LANE_SWITCH = auto()
    STRAIGHT_TURN = auto()
    PERTURBED_CYCLING = auto()
    MATCH_EXP_DATA = auto()

@unique
class Model(Enum):
    """To determine if the model acts as a single, double or triple inverted pendulum."""
    SINGLE_PENDULUM = auto()
    DOUBLE_PENDULUM = auto()
    TRIPLE_PENDULUM = auto()

@unique
class InitGuess(Enum):
    ZEROS = auto()
    ONES = auto()
    INITIAL = auto()
    SIMULATED = auto()
    RANDOM = auto()
    PATH = auto()
    PREVIOUS = auto()

@dataclass(frozen=True)
class Metadata:
    bicycle_only: bool
    model_upper_body: bool
    model_torso: bool
    model_head: bool
    model_legs: bool
    sprung_steering: bool
    model: Model
    seat_type: SeatType
    torso_type: TorsoType
    steer_with: SteerWith
    pedaling_torque: bool
    extra_roll_torque: bool
    task: Task
    init_guess: InitGuess
    parameter_data_dir: str
    bicycle_parametrization: str
    rider_parametrization: str
    part_personalized_constants : dict
    duration: float
    lateral_displacement: float
    straight_length: float
    turn_radius: float
    num_nodes: int
    weight: float
    weight_tr: float
    weight_ct: float
    x_meas_vec: dict

    def __post_init__(self):
        if self.model_upper_body:
            if self.bicycle_only:
                raise ValueError("Cannot have both model_upper_body and bicycle_only.")
            elif self.seat_type is SeatType.NONE:
                raise ValueError("Cannot have model_upper_body and no seat joint.")
        else:
            if self.seat_type is not SeatType.NONE:
                raise ValueError(
                    "Cannot have no model_upper_body and a seat joint.")
            elif self.steer_with is SteerWith.SEAT_TORQUE:
                raise ValueError(
                    "Cannot have no model_upper_body and a seat torque input.")
        if self.weight < 0 or self.weight > 1:
            raise ValueError("Weight must be between 0 and 1.")

    @property
    def interval_value(self):
        return self.duration / (self.num_nodes - 1)

@dataclass(frozen=True)
class ConstraintStorage:
    """Constraint storage object."""
    initial_state_constraints: dict[sm.Basic, float]
    final_state_constraints: dict[sm.Basic, float]
    instance_constraints: tuple[sm.Expr, ...]
    bounds: dict[sm.Basic, tuple[float, float]]

@dataclass
class DataStorage:
    """Data storage object."""
    metadata: Metadata
    #bicycle_rider: bm.BicycleRider | None = None
    bicycle_rider: SimBicycleRider | None = None
    bicycle: bm.WhippleBicycle | None = None
    #rider: bm.Rider | None = None
    rider: SimRider | None = None
    system: me.System | None = None
    eoms: sm.ImmutableMatrix | None = None
    input_vars: sm.ImmutableMatrix | None = None
    constants: dict[sm.Basic, float] | None = None
    simulator: Simulator | None = None
    objective_expr: sm.Expr | None = None
    constraints: ConstraintStorage | None = None
    initial_guess: npt.NDArray[np.float_] | None = None
    problem: Problem | None = None
    solution: npt.NDArray[np.float_] | None = None

    def __getstate__(self):
        # Problem cannot be pickled.
        return {k: v for k, v in self.__dict__.items() if k != 'problem'}

    def __setstate__(self, state):
        for key, value in state.items():
            setattr(self, key, value)

    @property
    def x(self) -> sm.ImmutableMatrix:
        """State variables."""
        return self.system.q.col_join(self.system.u)

    @property
    def r(self) -> sm.ImmutableMatrix:
        """Input variables."""
        return self.input_vars

    @property
    def time_array(self) -> npt.NDArray[np.float_]:
        """Time array."""
        return np.linspace(0, self.metadata.duration, self.metadata.num_nodes)

    @property
    def solution_state(self) -> npt.NDArray[np.float_]:
        """State trajectory from the solution."""
        n, N = self.x.shape[0], self.metadata.num_nodes
        return self.solution[:n * N].reshape((n, N))

    @property
    def solution_input(self) -> npt.NDArray[np.float_]:
        """Input trajectory from the solution."""
        n, q, N = self.x.shape[0], self.r.shape[0], self.metadata.num_nodes
        return self.solution[n * N:(q + n) * N].reshape((q, N))

    @property
    def lane_switch_task(self) -> sm.Expr:
        """Target path of the lane switch."""
        cos_trans = lambda x: (1 - sm.cos(sm.pi * x)) / 2
        s = self.metadata.straight_length
        d_lat = self.metadata.lateral_displacement
        x, y = self.bicycle.q[0], self.bicycle.q[1]
        return y - sm.Piecewise(
            (0, x < s),
            (d_lat * cos_trans((x - s) / (s)), x <= 2 * s),
            (d_lat, True))

    @property
    def double_lane_switch_task(self) -> sm.Expr:
        """Target path of the double lane switch."""
        cos_trans = lambda x: (1 - sm.cos(sm.pi * x)) / 2
        s = self.metadata.straight_length
        d_lat = self.metadata.lateral_displacement
        x, y = self.bicycle.q[0], self.bicycle.q[1]
        return y - sm.Piecewise(
            (0, x < s),
            (d_lat * cos_trans((x - s) / (s)), x <= 2 * s),
            (d_lat, x <= 3 * s),
            (d_lat * (1 - cos_trans((x - s) / (s))), x <= 4 * s),
            (0, x <= 5 * s))

    @property
    def straight_turn_task(self) -> sm.Expr:
        """Target path of the 90 degree turn."""
        s = self.metadata.straight_length
        r = self.metadata.turn_radius
        x, y = self.bicycle.q[0], self.bicycle.q[1]
        turn_direction = sm.sign(r)
        # Define the components of the path
        straight_1 = sm.Piecewise((0, x < s))
        turning = sm.Piecewise(
            (turn_direction * (r - sm.sqrt(r**2 - (x - s)**2)),
                (x >= s) & (x < s + sm.Abs(r))
            ))
        straight_2 = sm.Piecewise(
            (turn_direction * r,
                x >= s + sm.Abs(r)
            ))
        # Combine the components for the full path
        path = sm.Piecewise(
            (straight_1, x < s),
            (turning, (x >= s) & (x < s + sm.Abs(r))),
            (straight_2, x >= s + sm.Abs(r)))
        return y - path

    @property
    def straight_line_task(self) -> sm.Expr:
        """ Straight line cycling task to be combined with a wind perturbation. """
        s = 5 * self.metadata.straight_length
        x, y = self.bicycle.q[0], self.bicycle.q[1]
        path = sm.Piecewise(
            (0, x <= s),  # y = 0 for all points along the straight path up to length s
            (sm.nan, x > s)  # Path undefined beyond length s
        )

        return y - path
    
    
    @property
    def match_exp_data_task(self) -> sm.Expr:
        """ Task is to match the experimental data. """
        
        #Choose here the states and measurements compared
        u = self.bicycle.u[1]
        yaw_angle, roll_angle, steer_angle = self.bicycle.q[3], self.bicycle.q[4], self.bicycle.q[7]
        roll_rate, yaw_rate, steer_rate = self.bicycle.u[3], self.bicycle.u[4], self.bicycle.u[7]
        torso_roll_rate = self.rider.torsojoint.u[0]
        
        x_simulated = np.array([yaw_angle, roll_angle, steer_angle, roll_rate, yaw_rate, steer_rate, torso_roll_rate])
        
        x_meas_vec = np.array([self.metadata.x_meas_vec[k] for k in self.metadata.x_meas_vec.keys()])
        
        J = x_meas_vec - x_simulated

        return J