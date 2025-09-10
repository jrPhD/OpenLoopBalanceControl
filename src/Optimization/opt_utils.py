from __future__ import annotations

import argparse
import enum
import json
import os
import re
from copy import copy
from time import perf_counter
import cloudpickle as cp
import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from symbrim.core.base_classes import BrimBase
from symbrim.utilities.plotting import Plotter
from sympy.physics.mechanics import Point

import PIL
from matplotlib.animation import FuncAnimation, HTMLWriter, PillowWriter, FFMpegWriter
from scipy.interpolate import CubicSpline
from symmeplot.matplotlib import PlotBody, PlotVector
from src.Optimization.opt_container import DataStorage
from src.SimTorsoJoints import PinTorsoJoint, SphericalTorsoJoint, FixedTorsoJoint
from src.sim_seats import InterSeatJoint
from symbrim import SideLeanSeat, FixedSeat
from copy import copy


class Timer:
    def __init__(self):
        self.current_description = None
        self.readout = []

    def __call__(self, current_description: str) -> "Timer":
        self.current_description = current_description
        return self

    def __enter__(self):
        print(f"{self.current_description}...")
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start
        self.readout.append((self.current_description, self.time))
        print(f'{self.current_description} ran in {self.time:.3f} seconds')

    def to_file(self, file) -> None:
        with open(file, "w") as f:
            for description, time in self.readout:
                f.write(f"{description}: {time} seconds\n")


class EnumAction(argparse.Action):
    """Argparse action for handling Enums.

    Source: https://stackoverflow.com/a/60750535/20185124
    """
    def __init__(self, **kwargs):
        # Pop off the type value
        enum_type = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum_type is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum_type, enum.Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Generate choices from the Enum
        kwargs.setdefault("choices", tuple(e.name for e in enum_type))
        super(EnumAction, self).__init__(**kwargs)
        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        value = self._enum[values]
        setattr(namespace, self.dest, value)

class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types.

    Source: https://stackoverflow.com/a/49677241/20185124
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        try:
            return json.JSONEncoder.default(self, obj)
        except TypeError:
            return str(obj)

def get_all_symbols_from_model(brim_obj: BrimBase) -> set[sm.Symbol]:
    """Get all the symbols from a model."""
    syms = set(brim_obj.symbols.values())
    if hasattr(brim_obj, "submodels"):
        for submodel in brim_obj.submodels:
            syms.update(get_all_symbols_from_model(submodel))
    if hasattr(brim_obj, "connections"):
        for connection in brim_obj.connections:
            syms.update(get_all_symbols_from_model(connection))
    if hasattr(brim_obj, "load_groups"):
        for load_group in brim_obj.load_groups:
            syms.update(get_all_symbols_from_model(load_group))
    return syms


def get_ipopt_statistics(result_dir):
    with open(os.path.join(result_dir, "ipopt.txt"), "r", encoding="utf-8") as f:
        ipopt_output = f.read()
    objective = float(re.search(
        re.compile(r"Objective.*?:\s+(.*?)\s+(.*)"), ipopt_output).group(2))
    nlp_iterations = int(re.search(
        re.compile(r"Number of Iterations\.\.\.\.: (\d+)"), ipopt_output).group(1))
    ipopt_time = float(re.search(
        re.compile(r"Total seconds in IPOPT[ ]+= (\d+\.\d+)"), ipopt_output).group(1))
    ipopt_exit = re.search(re.compile(f"EXIT: (.*)"), ipopt_output).group(1)
    return {
        "Objective": objective,
        "#NLP iterations": nlp_iterations,
        "Time in Ipopt": ipopt_time,
        "Ipopt exit status": ipopt_exit,
    }

def create_objective_function(data: DataStorage, objective: sm.Expr, free=None):
    nx = data.x.shape[0]
    nr = data.input_vars.shape[0]
    #na = data.angles.shape[0]
    print('nx:', nx, ', nr:', nr)#, 'na:', na)
    N, interval = data.metadata.num_nodes, data.metadata.interval_value

    split_N = nx * N
    split_U = split_N + (nr * N)
    #split_A = split_U + (na * N)

    objective = me.msubs(objective, data.constants)
    objective_grad = sm.ImmutableMatrix([objective]).jacobian(
        data.x.col_join(data.input_vars))
        #data.x.col_join(data.input_vars).col_join(data.angles))
    objective_grad = [
        np.zeros(N) if grad == 0 else grad for grad in objective_grad]
    print('objective:', objective, type(objective))

    eval_objective = sm.lambdify((data.x, data.input_vars), objective, cse=True)
    eval_objective_grad = sm.lambdify((data.x, data.input_vars), objective_grad, cse=True)
    def obj(free):
        return interval * eval_objective(
            free[:split_N].reshape((nx, N)), free[split_N:].reshape((nr, N))).sum()

    def obj_grad(free):
        return interval * np.hstack(
            eval_objective_grad(free[:split_N].reshape((nx, N)),
                                free[split_N:].reshape((nr, N))))
    """
    eval_objective = sm.lambdify((data.x, data.input_vars, data.angles), objective, cse=True)
    eval_objective_grad = sm.lambdify((data.x, data.input_vars, data.angles), objective_grad, cse=True)

    def obj(free):
        print(f"free[:split_N]: {free[:split_N].shape}")  # Debug size of X
        print(f"free[split_N:split_U]: {free[split_N:split_U].shape}")  # Debug size of U
        print(f"free[split_U:split_A]: {free[split_U:split_A].shape}")  # Debug size of A
        return (interval * eval_objective(
            free[:split_N].reshape((nx, N)),
            free[split_N:split_U].reshape((nr, N)),
            free[split_U:split_A].reshape((na, N)))).sum()

    def obj_grad(free):
        return interval * np.hstack(
            eval_objective_grad(free[:split_N].reshape((nx, N)),
                                free[split_N:split_U].reshape((nr, N)),
                                free[split_U:split_A].reshape((na, N))))
    """
    print('obj & obj_grad::', obj, obj_grad)
    return obj, obj_grad

def plot_constraint_violations(self, vector):
    """Improved version of opty's ``Problem.plot_constraint_violations``."""

    con_violations = self.con(vector)
    con_nodes = range(self.collocator.num_states,
                      self.collocator.num_collocation_nodes + 1)
    N = len(con_nodes)
    fig, axes = plt.subplots(2, 1, figsize=(25, 25))

    for i, symbol in enumerate(self.collocator.state_symbols):
        axes[0].plot(con_nodes, con_violations[i * N:i * N + N],
                     label=sm.latex(symbol, mode='inline'))
    axes[0].legend()

    axes[0].set_title('Constraint Violations')
    axes[0].set_xlabel('Node Number')

    left = range(len(self.collocator.instance_constraints))
    axes[-1].bar(left, con_violations[-len(self.collocator.instance_constraints):],
                 tick_label=[sm.latex(s, mode='inline')
                             for s in self.collocator.instance_constraints])
    axes[-1].set_ylabel('Instance')
    axes[-1].set_xticklabels(axes[-1].get_xticklabels(), rotation=-10)

    return axes

def check_config(data: DataStorage):
    """ This function checks if the settings selected in the ArgumentParser in opt_run.py
        can be combined with each other, and will form a logical model."""
    if data.metadata.model == data.metadata.model.TRIPLE_PENDULUM:
        if data.metadata.model_head is False:
            raise ValueError("Cannot have a triple pendulum upper body model without a head. So add the head.")
        elif data.metadata.model_torso is False:
            raise ValueError("Cannot have a triple pendulum upper body model without a torso. So add the torso.")
        elif data.metadata.seat_type == data.metadata.seat_type.FIXED:
            raise ValueError("Cannot have a triple pendulum upper body model with a fixed seat. So un-fix it")
        else:
            print("Model configuration seems to be OK.")
    elif data.metadata.model == data.metadata.model.DOUBLE_PENDULUM:
        if data.metadata.model_torso is False:
            raise ValueError("Cannot have a double pendulum model without a torso. So add the torso.")
        elif data.metadata.steer_with == data.metadata.steer_with.UPPER_BODY_TORQUE:
            raise ValueError("Cannot control a double pendulum model with all upper body torques. "
                             "So change to 'SEAT_AND_TORSO_TORQUE', or to 'TRIPLE_PENDULUM'.")
        else:
            print("Model configuration seems to be OK.")
    elif data.metadata.model == data.metadata.model.SINGLE_PENDULUM:
        if data.metadata.steer_with == data.metadata.steer_with.SEAT_AND_TORSO_TORQUE:
            raise ValueError("Cannot control a single pendulum model with two different joint torques. "
                             "So change to 'SEAT_TORQUE', or to 'DOUBLE_PENDULUM'.")
        elif data.metadata.steer_with == data.metadata.steer_with.UPPER_BODY_TORQUE:
            raise ValueError("Cannot control a single pendulum model with three different joint torques. "
                             "So change to 'SEAT_TORQUE', or to 'TRIPLE_PENDULUM'.")
        elif data.metadata.steer_with == data.metadata.steer_with.TORSO_TORQUE:
            if data.metadata.seat_type == data.metadata.seat_type.FIXED:
                print("Model configuration seems to be OK.")
            else:
                raise ValueError("Cannot have a single pendulum model at the torso joint without a fixed seat. "
                                 "So fix the seat, or change to 'DOUBLE_PENDULUM'.")
        else:
            print("Model configuration seems to be OK.")
    else:
        print("Model configuration seems to be OK.")

def create_plots(data: DataStorage) -> tuple[plt.Figure, plt.Axes]:
    t_arr = data.time_array
    x_arr = data.solution_state
    r_arr = data.solution_input

    print('data.x stuffies', data.x, data.x.shape)
    print('lenghts, shapes & types of t, x and r array:')
    print(len(t_arr), np.shape(t_arr), type(t_arr))
    print(len(x_arr), np.shape(x_arr), type(x_arr))
    print(len(r_arr), np.shape(r_arr), type(r_arr))
    print('input variables r_arr:', data.r)
    print('solution state vector:', data.system.q.col_join(data.system.u))
    cm = 1/2.54 ## from inch to centimeter
    # To calculate the RMS error and make correct path plots, the path trajectories divided over points spaced with equal distance to eachother
    if data.metadata.task == data.metadata.task.LANE_SWITCH:
        """ This task is only used for testing purposes, and not for relevant data gathering. 
            Therefore, the path datapoints are calculated from set q1 steps, and not following the actual path and is thus less accurate"""
        q1_path = np.linspace(0, 3 * data.metadata.straight_length, num=data.metadata.num_nodes)
        q2_path = sm.lambdify(
            (data.bicycle.q[0],), sm.solve(data.lane_switch_task, data.bicycle.q[1])[0], cse=True
        )(q1_path)
    elif data.metadata.task == data.metadata.task.DOUBLE_LANE_SWITCH:
        path_length = data.metadata.straight_length + \
                      (data.metadata.straight_length) + \
                      data.metadata.straight_length + \
                      (data.metadata.straight_length) + \
                      data.metadata.straight_length
        path_len = np.linspace(0, path_length, data.metadata.num_nodes)
        q1_path = []
        q2_path = []
        s = data.metadata.straight_length
        d_lat = data.metadata.lateral_displacement
        d_long1 = 3 * s
        d_long2 = 5 * s

        for q1 in path_len:
            if q1 < s:
                # First straight segment
                q2 = 0
            elif q1 <= d_long1 - s:
                # First lane switch
                progress = (q1 - s) / (d_long1 - 2 * s)
                q2 = d_lat * (1 - np.cos(np.pi * progress)) / 2
            elif q1 <= d_long1:
                # Middle straight segment
                q2 = d_lat
            elif q1 <= d_long2 - s:
                # Second lane switch (back)
                progress = (q1 - d_long1) / (d_long1 - 2 * s)  # Progress normalized for the second switch
                q2 = d_lat * (1 + np.cos(np.pi * progress)) / 2
            else:
                # Final straight segment
                q2 = 0
            q1_path.append(q1)
            q2_path.append(q2)
        q1_path = np.array(q1_path)
        q2_path = np.array(q2_path)
    elif data.metadata.task == data.metadata.task.STRAIGHT_TURN:
        path_length = 2 * data.metadata.straight_length + (np.abs(data.metadata.turn_radius) * np.pi / 2)
        path_len = np.linspace(0, path_length, data.metadata.num_nodes)
        q1_path = []
        q2_path = []
        for path in path_len:
            if path <= data.metadata.straight_length:
                q1 = path
                q2 = 0
            elif path <= data.metadata.straight_length + (np.abs(data.metadata.turn_radius) * np.pi / 2):
                #
                angle = (path - data.metadata.straight_length) / np.abs(data.metadata.turn_radius)
                q1 = data.metadata.straight_length + data.metadata.turn_radius * np.sin(angle)
                q2 = np.sign(data.metadata.turn_radius) * data.metadata.turn_radius * (1 - np.cos(angle))
            else:
                # Second straight segment
                remaining_length = path - (data.metadata.straight_length + (np.abs(data.metadata.turn_radius) * np.pi / 2))
                q1 = data.metadata.straight_length + np.abs(data.metadata.turn_radius)
                q2 = np.sign(data.metadata.turn_radius) * data.metadata.turn_radius + remaining_length
            q1_path.append(q1)
            q2_path.append(q2)

        q1_path = np.array(q1_path)
        q2_path = np.array(q2_path)
    elif data.metadata.task == data.metadata.task.PERTURBED_CYCLING:
        q1_path = np.linspace(0, (5 * data.metadata.straight_length), num=data.metadata.num_nodes)
        q2_path = sm.lambdify(
            (data.bicycle.q[0],), sm.solve(data.straight_line_task, data.bicycle.q[1])[0], cse=True
        )(q1_path)

    q1_error = q1_path - x_arr[0]
    q2_error = q2_path - x_arr[1]
    combined_error = np.sqrt(q1_error ** 2 + q2_error ** 2)
    print('combined error thing:', len(combined_error), type(combined_error))
    # Compute the RMS error
    rms_error = np.sqrt(np.mean(combined_error ** 2))

    # Create mapping of the bicycle coordinates (qs) and speeds (us) from the bicycle data to be used in plotting
    idx_mapping_bicycle = [(0, "x"), (1, "y"), (2, "bicycle yaw"), (3, "bicycle roll"), (5, "rear wheel"),
                           (6, "bicycle steer"), (7, "front wheel")]
    qs = {name: data.bicycle.q[i] for i, name in idx_mapping_bicycle}
    us = {name: data.bicycle.u[i] for i, name in idx_mapping_bicycle}

    # Create mapping of the rider coordinates from the rider data to be used in plotting
    if type(data.bicycle_rider.seat) == SideLeanSeat:
        idx_mapping_seat = [(0, "seat lean")]
    elif type(data.bicycle_rider.seat) == InterSeatJoint:
        idx_mapping_seat = [(0, "seat lean [$^\circ$]")]
        idx_mapping_interseat = [(0, "seat shift [cm]")]
    if data.metadata.model_torso:
        if type(data.rider.torsojoint) == PinTorsoJoint:
            idx_mapping_torsojoint = [(0, "torso lean")]
        elif type(data.rider.torsojoint) == SphericalTorsoJoint:
            idx_mapping_torsojoint = [(0, "torso flexion"), (1, "torso lean"), (2, "torso rotation")]
        elif type(data.rider.torsojoint) == FixedTorsoJoint:
            None
    if data.metadata.model_legs == True:
        idx_mapping_left_hip = [(0, "left hip flexion"), (1, "left hip adduction"), (2, "left hip rotation")]
        idx_mapping_left_leg = [(0, "left leg knee angle"), (1, "left leg ankle angle")]
        idx_mapping_right_hip = [(0, "right hip flexion"), (1, "right hip adduction"), (2, "right hip rotation")]
        idx_mapping_right_leg = [(0, "left leg knee angle"), (1, "left leg ankle angle")]
    if data.metadata.model_head:
        if data.metadata.model == data.metadata.model.TRIPLE_PENDULUM:
            idx_mapping_neck = [(0, "neck lean")]
        else:
            None

    if type(data.bicycle_rider.seat) == SideLeanSeat:
        qs.update({name: data.bicycle_rider.seat.q[i] for i, name in idx_mapping_seat})
        print('qs for the seat:', qs)
    elif type(data.bicycle_rider.seat) == InterSeatJoint:
        qs.update({name: data.rider.shiftingsideleanseat.q[i] for i, name in idx_mapping_seat})
        qs.update({name: data.bicycle_rider.seat.q[i] for i, name in idx_mapping_interseat})
        print('qs for the seat:', qs)
    elif type(data.bicycle_rider.seat) == FixedSeat:
        None
    if data.metadata.model_torso:
        if (data.metadata.model == data.metadata.model.SINGLE_PENDULUM and 
                data.metadata.steer_with == data.metadata.steer_with.SEAT_TORQUE):
            None
        else:
            qs.update({name: data.rider.torsojoint.q[i] for i, name in idx_mapping_torsojoint})
    if data.metadata.model_legs:
        qs.update({name: data.rider.left_hip.q[i] for i, name in idx_mapping_left_hip})
        qs.update({name: data.rider.right_hip.q[i] for i, name in idx_mapping_right_hip})
        qs.update({name: data.rider.left_leg.q[i] for i, name in idx_mapping_left_leg})
        qs.update({name: data.rider.right_leg.q[i] for i, name in idx_mapping_right_leg})
    if data.metadata.model_head:
        if data.metadata.model == data.metadata.model.TRIPLE_PENDULUM:
            qs.update({name: data.rider.neck.q[i] for i, name in idx_mapping_neck})
        else:
            None
    print('qs ->', qs)
    # Do the same thing for the rider speeds
    if type(data.bicycle_rider.seat) == SideLeanSeat:
        us.update({name: data.bicycle_rider.seat.u[i] for i, name in idx_mapping_seat})
        print('us for the seat:', us)
    elif type(data.bicycle_rider.seat) == InterSeatJoint:
        us.update({name: data.rider.shiftingsideleanseat.u[i] for i, name in idx_mapping_seat})
        us.update({name: data.bicycle_rider.seat.u[i] for i, name in idx_mapping_interseat})
        print('us for the seat:', us)
    elif type(data.bicycle_rider.seat) == FixedSeat:
        None
    if data.metadata.model_torso:
        if (data.metadata.model == data.metadata.model.SINGLE_PENDULUM and
                data.metadata.steer_with == data.metadata.steer_with.SEAT_TORQUE):
            None
        else:
            print("torsojoint should have its u's")
            us.update({name: data.rider.torsojoint.u[i] for i, name in idx_mapping_torsojoint})
            print(us)
    if data.metadata.model_legs:
        us.update({name: data.rider.left_hip.u[i] for i, name in idx_mapping_left_hip})
        us.update({name: data.rider.right_hip.u[i] for i, name in idx_mapping_right_hip})
        us.update({name: data.rider.left_leg.u[i] for i, name in idx_mapping_left_leg})
        us.update({name: data.rider.right_leg.u[i] for i, name in idx_mapping_right_leg})
    if data.metadata.model_head:
        if data.metadata.model == data.metadata.model.TRIPLE_PENDULUM:
            us.update({name: data.rider.neck.u[i] for i, name in idx_mapping_neck})
        else:
            None

    # Combine all qs and us in an array so that the plot functions can call them
    get_q = lambda q_name: x_arr[data.system.q[:].index(qs[q_name]), :]  # noqa: E731
    get_u = lambda u_name: x_arr[len(data.system.q) + data.system.u[:].index(us[u_name]), :]  # noqa: E731

    # Set figure style for all figures:
    #plt.rcParams['text.usetex'] = True  # Disable LaTeX dependency if False
    plt.style.use(['science', 'ieee', 'no-latex'])

    # Figure 1: Subplots for trajectory and angles
    if data.metadata.task == data.metadata.task.STRAIGHT_TURN:
        fig1, ax1 = plt.subplots(1, 2, figsize=(12*cm, 6*cm))
    else:
        fig1, ax1 = plt.subplots(2, 1, figsize=(12*cm, 6*cm))
    ax1[0].plot(q1_path, q2_path, label="Target path")
    ax1[0].plot(get_q("x"), get_q("y"), label="trajectory")
    ax1[0].set_xlabel("Longitudinal displacement [m]")
    ax1[0].set_ylabel("Lateral displacement [m]")
    if data.metadata.task == data.metadata.task.LANE_SWITCH:
        ax1[0].set_ylim([-0.5, data.metadata.lateral_displacement + 0.5])
    elif data.metadata.task == data.metadata.task.DOUBLE_LANE_SWITCH:
        ax1[0].set_ylim([-0.5, data.metadata.lateral_displacement + 0.5])
    elif data.metadata.task == data.metadata.task.PERTURBED_CYCLING:
        #ax1[0].set_ylim([-0.5, 0.5])
        None
    elif data.metadata.task == data.metadata.task.STRAIGHT_TURN:
        ax1[0].set_ylim([-0.5, data.metadata.straight_length + data.metadata.turn_radius + 0.5])
    ax1[0].grid(color='gray', linestyle='--', linewidth=0.5)
    ax1[0].legend()

    name_mapping_r = {
            "T_p": ("pedal torque", "black", "dashdot"),
            "T_sls": ("sidelean torque", "green", "solid"),
            "T_tor": ("torsojoint torque", "red", "solid"),
            "T_neck": ("neck torque", "blue", "solid"),
            "T_tor_flexion": ("torsojoint flexion torque", "red", "dashed"),
            "T_tor_adduction": ("torsojoint adduction torque", "red", "solid"),
            "T_tor_rotation": ("torsojoint rotation torque", "red", "dotted"),
            "k_steer": ("front frame spring stiffness", "black", "dashdot")}

    for i, ri in enumerate(data.r):
        name, color, linestyle = name_mapping_r.get(ri.name, ("torque", "grey", "solid"))
        ax1[1].plot(t_arr, r_arr[i, :], label=name, color=color, linestyle=linestyle)
    ax1[1].set_xlabel("Time [s]")
    ax1[1].set_ylabel("Torque [Nm]")
    ax1[1].grid(color='gray', linestyle='--', linewidth=0.5)
    ax1[1].legend()

    color_map = {
        "bicycle roll": ("black", "dashdot"),
        "bicycle yaw": ("black", "dotted"),
        "bicycle steer": ("black", "dashed"),
        "x": ("darkorange", "solid"),
        "y": ("darkviolet", "solid"),
        "rear wheel": ("lime", "solid"),
        "front wheel": ("cyan", "solid"),
        "seat lean": ("green", "solid"),
        "seat lean [$^\circ$]": ("green", "solid"),
        "seat shift [cm]": ("green", "dashed"),
        "torso lean": ("red", "solid"),
        "torso flexion": ("red", "dashed"),
        "torso rotation": ("red", "dotted"),
        "neck lean": ("blue", "solid")}

    # Figure 2: plot for model angles
    fig2, ax2 = plt.subplots(figsize=(9*cm, 6*cm))
    ax2.set_title("Model Angles - Counter-steering")

    if type(data.bicycle_rider.seat) == InterSeatJoint:
        if data.metadata.model == data.metadata.model.SINGLE_PENDULUM:
            if data.metadata.steer_with == data.metadata.steer_with.SEAT_TORQUE:
                for name in ("bicycle roll", "seat lean [$^\circ$]", "seat shift [cm]"):  # , "seat shift"):
                    color, linestyle = color_map.get(name, ("grey", "solid"))
                    if name == "seat shift [cm]":
                        ax2.plot(t_arr, get_q(name) * 100, label=name, color=color, linestyle=linestyle)
                    else:
                        ax2.plot(t_arr, get_q(name) * 180 / np.pi, label=name, color=color, linestyle=linestyle)
        elif data.metadata.model == data.metadata.model.DOUBLE_PENDULUM:
            if type(data.rider.torsojoint) == PinTorsoJoint:
                for name in ("bicycle roll", "seat lean [$^\circ$]", "seat shift [cm]", "torso lean"):
                    color, linestyle = color_map.get(name, ("grey", "solid"))
                    if name == "seat shift [cm]":
                        ax2.plot(t_arr, get_q(name) * 100, label=name, color=color, linestyle=linestyle)
                    else:
                        ax2.plot(t_arr, get_q(name) * 180 / np.pi, label=name, color=color, linestyle=linestyle)
            if type(data.rider.torsojoint) == SphericalTorsoJoint:
                for name in ("bicycle roll", "seat lean [$^\circ$]", "seat shift [cm]", "torso lean", "torso flexion", "torso rotation"):
                    color, linestyle = color_map.get(name, ("grey", "solid"))
                    if name == "seat shift [cm]":
                        ax2.plot(t_arr, get_q(name) * 100, label=name, color=color, linestyle=linestyle)
                    else:
                        ax2.plot(t_arr, get_q(name) * 180 / np.pi, label=name, color=color, linestyle=linestyle)
        elif data.metadata.model == data.metadata.model.TRIPLE_PENDULUM:
            if type(data.rider.torsojoint) == PinTorsoJoint:
                for name in ("bicycle roll", "seat lean [$^\circ$]", "seat shift [cm]", "torso lean", "neck lean"):
                    color, linestyle = color_map.get(name, ("grey", "solid"))
                    if name == "seat shift [cm]":
                        ax2.plot(t_arr, get_q(name) * 100, label=name, color=color, linestyle=linestyle)
                    else:
                        ax2.plot(t_arr, get_q(name) * 180 / np.pi, label=name, color=color, linestyle=linestyle)
            if type(data.rider.torsojoint) == SphericalTorsoJoint:
                for name in ("bicycle roll", "seat lean [$^\circ$]", "seat shift [cm]", "torso lean", "torso flexion", "torso rotation", "neck lean"):
                    color, linestyle = color_map.get(name, ("grey", "solid"))
                    if name == "seat shift [cm]":
                        ax2.plot(t_arr, get_q(name) * 100, label=name, color=color, linestyle=linestyle)
                    else:
                        ax2.plot(t_arr, get_q(name) * 180 / np.pi, label=name, color=color, linestyle=linestyle)
    else:
        if data.metadata.model == data.metadata.model.SINGLE_PENDULUM:
            if data.metadata.steer_with == data.metadata.steer_with.SEAT_TORQUE:
                for name in ("bicycle roll", "seat lean"):  # , "seat shift"):
                    color, linestyle = color_map.get(name, ("grey", "solid"))
                    ax2.plot(t_arr, get_q(name) * 180 / np.pi, label=name, color=color, linestyle=linestyle)
            if data.metadata.steer_with == data.metadata.steer_with.TORSO_TORQUE:
                if type(data.rider.torsojoint) == PinTorsoJoint:
                    for name in ("bicycle roll", "torso lean"):
                        color, linestyle = color_map.get(name, ("grey", "solid"))
                        ax2.plot(t_arr, get_q(name) * 180 / np.pi, label=name, color=color, linestyle=linestyle)
                if type(data.rider.torsojoint) == SphericalTorsoJoint:
                    for name in ("bicycle roll", "torso lean", "torso flexion", "torso rotation"):
                        color, linestyle = color_map.get(name, ("grey", "solid"))
                        ax2.plot(t_arr, get_q(name) * 180 / np.pi, label=name, color=color, linestyle=linestyle)
        elif data.metadata.model == data.metadata.model.DOUBLE_PENDULUM:
            if type(data.rider.torsojoint) == PinTorsoJoint:
                for name in ("bicycle roll", "seat lean", "torso lean"):
                    color, linestyle = color_map.get(name, ("grey", "solid"))
                    ax2.plot(t_arr, get_q(name) * 180 / np.pi, label=name, color=color, linestyle=linestyle)
            if type(data.rider.torsojoint) == SphericalTorsoJoint:
                for name in ("bicycle roll", "seat lean", "torso lean", "torso flexion", "torso rotation"):
                    color, linestyle = color_map.get(name, ("grey", "solid"))
                    ax2.plot(t_arr, get_q(name) * 180 / np.pi, label=name, color=color, linestyle=linestyle)
        elif data.metadata.model == data.metadata.model.TRIPLE_PENDULUM:
            if type(data.rider.torsojoint) == PinTorsoJoint:
                for name in ("bicycle roll", "seat lean", "torso lean", "neck lean"):
                    color, linestyle = color_map.get(name, ("grey", "solid"))
                    ax2.plot(t_arr, get_q(name) * 180 / np.pi, label=name, color=color, linestyle=linestyle)
            if type(data.rider.torsojoint) == SphericalTorsoJoint:
                for name in ("bicycle roll", "seat lean", "torso lean", "torso flexion", "torso rotation", "neck lean"):
                    color, linestyle = color_map.get(name, ("grey", "solid"))
                    ax2.plot(t_arr, get_q(name) * 180 / np.pi, label=name, color=color, linestyle=linestyle)

    ax2.set_ylabel("Angle [deg]")
    ax2.set_xlabel("Time [s]")
    ax2.grid(color='gray', linestyle='--', linewidth=0.5)
    ax2.legend()

    # bicycle state plots:
    fig3, ax3 = plt.subplots(1, 2, figsize=(8, 5))
    # Angle plot
    for name in ("bicycle yaw", "bicycle roll", "bicycle steer"):
        ax3[0].plot(t_arr, get_q(name) * 180 / np.pi, label=name)
    ax3[0].set_ylabel("Angle [deg])")
    #ax3[0].set_ylim([-1, 1])
    #ax3[0].set_xlim([0, 1.5])
    ax3[0].legend()

    # Angular velocity plot
    for name in ("bicycle yaw", "bicycle roll", "bicycle steer"):
        ax3[1].plot(t_arr, get_u(name) * 180 / np.pi, label=name)
    ax3[1].set_xlabel("Time [s]")
    ax3[1].set_ylabel("Angular velocity [deg/s]")
    fig3.align_labels()
    #fig3.tight_layout()

    joint_power_mapping = {
        "seat": (("T_sls", "seat lean"), ("green", "solid")),
        "torso": (("T_tor", "torso lean"), ("red", "solid")),
        "neck": (("T_neck", "neck lean"), ("blue", "solid")),
        "torso flexion": (("T_tor_flexion", "torso flexion"), ("red", "dashed")),
        "torso adduction": (("T_tor_adduction", "torso lean"), ("red", "solid")),
        "torso rotation": (("T_tor_rotation", "torso rotation"), ("red", "dotted"))}

    if data.metadata.steer_with == data.metadata.steer_with.SEAT_TORQUE:
        active_joints = ["seat"]
        if type(data.bicycle_rider.seat) == SideLeanSeat:
            P = r_arr[0] * x_arr[12]
        elif type(data.bicycle_rider.seat) == InterSeatJoint:
            P = r_arr[0] * x_arr[13]
        W = np.abs((P) * (data.metadata.duration / data.metadata.num_nodes))
        cumulative_energy = np.cumsum(np.abs(W))
    elif data.metadata.steer_with == data.metadata.steer_with.TORSO_TORQUE:
        active_joints = ["torso"]
        if type(data.rider.torsojoint) == PinTorsoJoint:
            print('calculating used energy by the torso torque')
            P = r_arr[0] * x_arr[12]
            W = np.abs((P) * (data.metadata.duration / data.metadata.num_nodes))
            cumulative_energy = np.cumsum(np.abs(W))
        if type(data.rider.torsojoint) == SphericalTorsoJoint:
            print('calculating used energy by the spherical torso torque')
            P1 = r_arr[0] * x_arr[15]
            P2 = r_arr[1] * x_arr[14]
            P3 = r_arr[2] * x_arr[16]
            W1 = np.abs((P1) * (data.metadata.duration / data.metadata.num_nodes))
            W2 = np.abs((P2) * (data.metadata.duration / data.metadata.num_nodes))
            W3 = np.abs((P3) * (data.metadata.duration / data.metadata.num_nodes))
            W = W1 + W2 + W3
            cumulative_energy = np.cumsum(np.abs(W))
    elif data.metadata.steer_with == data.metadata.steer_with.SEAT_AND_TORSO_TORQUE:
        active_joints = ["seat", "torso"]
        print('calculating used energy by the seat and torso torques')
        if type(data.bicycle_rider.seat) == SideLeanSeat:
            P1 = r_arr[0] * x_arr[13]
            P2 = r_arr[1] * x_arr[14]
        elif type(data.bicycle_rider.seat) == InterSeatJoint:
            P1 = r_arr[0] * x_arr[14]
            P2 = r_arr[1] * x_arr[15]
        W1 = np.abs((P1) * (data.metadata.duration / data.metadata.num_nodes))
        W2 = np.abs((P2) * (data.metadata.duration / data.metadata.num_nodes))
        W = W1 + W2
        cumulative_energy = np.cumsum(np.abs(W))
    elif data.metadata.steer_with == data.metadata.steer_with.UPPER_BODY_TORQUE:
        active_joints = ["seat", "torso", "neck"]
        if type(data.bicycle_rider.seat) == SideLeanSeat:
            P1 = r_arr[1] * x_arr[14]
            P2 = r_arr[2] * x_arr[15]
            P3 = r_arr[0] * x_arr[16]
        elif type(data.bicycle_rider.seat) == InterSeatJoint:
            P1 = r_arr[1] * x_arr[15]
            P2 = r_arr[2] * x_arr[16]
            P3 = r_arr[0] * x_arr[17]
        W1 = np.abs((P1) * (data.metadata.duration / data.metadata.num_nodes))
        W2 = np.abs((P2) * (data.metadata.duration / data.metadata.num_nodes))
        W3 = np.abs((P3) * (data.metadata.duration / data.metadata.num_nodes))
        W = np.abs(W1) + np.abs(W2) + np.abs(W3)
        cumulative_energy = np.cumsum(W)

    # This plots the model torques, the angular velocities, and the used energy from this torque and angular speed.
    fig4, ax4 = plt.subplots(3, 1, figsize=(8, 6))
    for i, ri in enumerate(data.r):
        name, color, linestyle = name_mapping_r.get(ri.name, ("torque", "grey", "solid"))
        ax4[0].plot(t_arr, r_arr[i, :], label=name, color=color, linestyle=linestyle)
    ax4[0].set_ylabel("Torque [Nm]")
    ax4[0].grid(color='gray', linestyle='--', linewidth=0.5)
    ax4[0].legend()

    print('get q:', get_q, 'qs:', qs)
    print('get u:', get_u, 'us:', us)

    if type(data.bicycle_rider.seat) == InterSeatJoint:
        if data.metadata.model == data.metadata.model.SINGLE_PENDULUM:
            if data.metadata.steer_with == data.metadata.steer_with.SEAT_TORQUE:
                for name in ("seat lean [$^\circ$]", "seat shift [cm]"):
                    color, linestyle = color_map.get(name, ("grey", "solid"))
                    if name == "seat shift [cm]":
                        ax4[1].plot(t_arr, get_u(name) * 100, label=name, color=color, linestyle=linestyle)
                    else:
                        ax4[1].plot(t_arr, get_u(name) * 180 / np.pi, label=name, color=color, linestyle=linestyle)
        elif data.metadata.model == data.metadata.model.DOUBLE_PENDULUM:
            if type(data.rider.torsojoint) == PinTorsoJoint:
                for name in ("seat lean [$^\circ$]", "seat shift [cm]", "torso lean"):
                    color, linestyle = color_map.get(name, ("grey", "solid"))
                    if name == "seat shift [cm]":
                        ax4[1].plot(t_arr, get_u(name) * 100, label=name, color=color, linestyle=linestyle)
                    else:
                        ax4[1].plot(t_arr, get_u(name) * 180 / np.pi, label=name, color=color, linestyle=linestyle)
            if type(data.rider.torsojoint) == SphericalTorsoJoint:
                for name in ("seat lean [$^\circ$]", "seat shift [cm]", "torso lean", "torso flexion", "torso rotation"):
                    color, linestyle = color_map.get(name, ("grey", "solid"))
                    if name == "seat shift [cm]":
                        ax4[1].plot(t_arr, get_u(name) * 100, label=name, color=color, linestyle=linestyle)
                    else:
                        ax4[1].plot(t_arr, get_u(name) * 180 / np.pi, label=name, color=color, linestyle=linestyle)
        elif data.metadata.model == data.metadata.model.TRIPLE_PENDULUM:
            if type(data.rider.torsojoint) == PinTorsoJoint:
                for name in ("seat lean [$^\circ$]", "seat shift [cm]", "torso lean", "neck lean"):
                    color, linestyle = color_map.get(name, ("grey", "solid"))
                    if name == "seat shift [cm]":
                        ax4[1].plot(t_arr, get_u(name) * 100, label=name, color=color, linestyle=linestyle)
                    else:
                        ax4[1].plot(t_arr, get_u(name) * 180 / np.pi, label=name, color=color, linestyle=linestyle)
            if type(data.rider.torsojoint) == SphericalTorsoJoint:
                for name in ("seat lean [$^\circ$]", "seat shift [cm]", "torso lean", "torso flexion", "torso rotation",
                             "neck lean"):
                    color, linestyle = color_map.get(name, ("grey", "solid"))
                    if name == "seat shift [cm]":
                        ax4[1].plot(t_arr, get_u(name) * 100, label=name, color=color, linestyle=linestyle)
                    else:
                        ax4[1].plot(t_arr, get_u(name) * 180 / np.pi, label=name, color=color, linestyle=linestyle)
    else:
        if data.metadata.model == data.metadata.model.SINGLE_PENDULUM:
            if data.metadata.steer_with == data.metadata.steer_with.SEAT_TORQUE:
                for name in ["seat lean"]:
                    color, linestyle = color_map.get(name, ("grey", "solid"))
                    ax4[1].plot(t_arr, get_u(name) * 180 / np.pi, label=name, color=color, linestyle=linestyle)
            elif data.metadata.steer_with == data.metadata.steer_with.TORSO_TORQUE:
                if type(data.rider.torsojoint) == PinTorsoJoint:
                    for name in ["torso lean"]:
                        color, linestyle = color_map.get(name, ("grey", "solid"))
                        ax4[1].plot(t_arr, get_u(name) * 180 / np.pi, label=name, color=color, linestyle=linestyle)
                if type(data.rider.torsojoint) == SphericalTorsoJoint:
                    for name in ("torso lean", "torso flexion", "torso rotation"):
                        color, linestyle = color_map.get(name, ("grey", "solid"))
                        ax4[1].plot(t_arr, get_u(name) * 180 / np.pi, label=name, color=color, linestyle=linestyle)
        elif data.metadata.model == data.metadata.model.DOUBLE_PENDULUM:
            if type(data.rider.torsojoint) == PinTorsoJoint:
                for name in ("seat lean", "torso lean"):
                    color, linestyle = color_map.get(name, ("grey", "solid"))
                    ax4[1].plot(t_arr, get_u(name) * 180 / np.pi, label=name, color=color, linestyle=linestyle)
            if type(data.rider.torsojoint) == SphericalTorsoJoint:
                for name in ("seat lean", "torso lean", "torso flexion", "torso rotation"):
                    color, linestyle = color_map.get(name, ("grey", "solid"))
                    ax4[1].plot(t_arr, get_u(name) * 180 / np.pi, label=name, color=color, linestyle=linestyle)
        elif data.metadata.model == data.metadata.model.TRIPLE_PENDULUM:
            if type(data.rider.torsojoint) == PinTorsoJoint:
                for name in ("seat lean", "torso lean", "neck lean"):
                    print('name:', name)
                    color, linestyle = color_map.get(name, ("grey", "solid"))
                    ax4[1].plot(t_arr, get_u(name) * 180 / np.pi, label=name, color=color, linestyle=linestyle)
            if type(data.rider.torsojoint) == SphericalTorsoJoint:
                for name in ("seat lean", "torso lean", "torso flexion", "torso rotation", "neck lean"):
                    color, linestyle = color_map.get(name, ("grey", "solid"))
                    ax4[1].plot(t_arr, get_u(name) * 180 / np.pi, label=name, color=color, linestyle=linestyle)
    ax4[1].set_ylabel("Angular velocity [degree/s]")
    ax4[1].grid(color='gray', linestyle='--', linewidth=0.5)
    ax4[1].legend()

    ax4[2].plot(t_arr, cumulative_energy, color='deepskyblue')
    ax4[2].fill_between(t_arr, cumulative_energy, color='deepskyblue', alpha=0.33)
    ax4[2].set_ylabel("Cumulative energy used [J]")
    ax4[2].set_xlabel("Time [s]")
    print('Total energy used =', cumulative_energy[-1], '[J]')
    print('The RMS tracking error =', rms_error, '[m]')

    # If there is wind in play, this plots the wind force and the model torques in response to this wind
    if data.metadata.task == data.metadata.task.PERTURBED_CYCLING:
        figw, axw = plt.subplots(1, 1, figsize=(10*cm, 6*cm))
        axw.plot(t_arr, data.wind_array)
        axw.set_ylabel("Wind force [N]")
        axw.set_xlabel("Time [s]")
        #for i, ri in enumerate(data.r):
        #    name, color, linestyle = name_mapping_r.get(ri.name, ("torque", "grey", "solid"))
        #    axw[1].plot(t_arr, r_arr[i, :], label=name, color=color, linestyle=linestyle)
        #axw[1].set_ylabel("Torque [Nm]")
        axw.grid(color='gray', linestyle='--', linewidth=0.5)
        #axw[1].legend()

    # This plots the bicycle longitudinal velocity (of the rear wheel) and the velocity of the rear wheel in longitudnial direction
    fig5, ax5 = plt.subplots(1, 1, figsize=(8, 6))
    radius = data.constants[data.bicycle.rear_wheel.radius]
    bicycle_speed_data = []
    for name in ("x", "rear wheel"):
        color, linestyle = color_map.get(name, ("grey", "solid"))
        if name == "x":
            ax5.plot(t_arr, get_u(name), label=name, color=color, linestyle=linestyle)
        else:
            ax5.plot(t_arr, get_u(name) * (- radius), label=name, color=color, linestyle=linestyle)
            speed_data = get_u(name) * (- radius)
    ax5.set_ylabel("Linear Velocity [m/s]")
    ax5.set_xlabel("Time [s]")
    ax5.autoscale(tight=True)
    ax5.grid(color='gray', linestyle='--', linewidth=0.5)
    ax5.legend(title="Order")
    average_speed = np.sum(speed_data) / data.metadata.num_nodes
    av_speed_kmh = average_speed * 3.6
    print('average speed =', average_speed, '[m/s], (=', av_speed_kmh,'[kmh])')


    return cumulative_energy[-1], rms_error, av_speed_kmh, cumulative_energy, speed_data, combined_error, fig1, ax1, fig2, ax2, fig3, ax3, fig4, ax4, fig5, ax5

def create_animation(data: DataStorage, output: str
                     ) -> tuple[plt.Figure, plt.Axes, FuncAnimation]:
    x_eval = CubicSpline(data.time_array, data.solution_state.T)
    r_eval = CubicSpline(data.time_array, data.solution_input.T)
    p, p_vals = zip(*data.constants.items())

    #max_disturbance = r_eval(data.time_array)[:, tuple(data.inp).index(wind)].max()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(20, 20))
    plotter = Plotter.from_model(data.bicycle_rider, ax=ax)
    #plotter.add_vector(data.problem.collocator.known_input_trajectories.keys() * data.bicycle.rear_frame.wheel_hub.axis / 30,
    #                   data.bicycle.rear_frame.saddle.point, name="wind", color="r")

    plotter.lambdify_system((data.x[:], data.input_vars[:], p))
    plotter.evaluate_system(x_eval(0), r_eval(0), p_vals)
    plotter.plot()
    _plot_ground(data, plotter)
    # Set explicit limits based on your data range for better layout control
    ax.set_xlim([-10, 10])  # Adjust these limits as needed
    ax.set_ylim([-10, 10])
    ax.set_zlim([-5, 5])

    ax.invert_zaxis()
    ax.invert_yaxis()
    ax.view_init(19, 14)
    ax.set_aspect("auto")  # Set aspect to auto for 3D stability
    ax.axis("off")

    fps = 30
    ani = plotter.animate(
        lambda ti: (x_eval(ti), r_eval(ti), p_vals),
        frames=np.arange(0, data.time_array[-1], 1 / fps),
        blit=False)
    html_writer = HTMLWriter()
    ani.save(output if output.endswith(".html") else output + ".html", writer=html_writer)
    """
    gif_output = output if output.endswith(".gif") else output + ".gif"
    gif_writer = PillowWriter(fps=fps)
    ani.save(gif_output, writer=gif_writer)

    mp4_output = output if output.endswith(".mp4") else output + ".mp4"
    mp4_writer = FFMpegWriter(fps=fps)
    ani.save(mp4_output, writer=mp4_writer)
    """
    return fig, ax, ani

def create_animation2(data: DataStorage, output: str
                     ) -> tuple[plt.Figure, plt.Axes, FuncAnimation]:
    x_eval = CubicSpline(data.time_array, data.solution_state.T)
    r_eval = CubicSpline(data.time_array, data.solution_input.T)
    p, p_vals = zip(*data.constants.items())

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(20, 20))
    plotter = Plotter.from_model(data.bicycle_rider, ax=ax)
    plotter.lambdify_system((data.x[:], data.input_vars[:], p))
    plotter.evaluate_system(x_eval(0), r_eval(0), p_vals)
    plotter.plot()
    _plot_ground(data, plotter)
    # Set explicit limits based on your data range for better layout control
    ax.set_xlim([-3, 3])  # Adjust these limits as needed
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])

    ax.invert_zaxis()
    ax.invert_yaxis()
    ax.view_init(7, 0)
    ax.set_aspect("auto")  # Set aspect to auto for 3D stability
    ax.axis("off")

    fps = 30
    ani = plotter.animate(
        lambda ti: (x_eval(ti), r_eval(ti), p_vals),
        frames=np.arange(0, data.time_array[-1], 1 / fps),
        blit=False)
    html_writer = HTMLWriter()
    ani.save(output if output.endswith(".html") else output + ".html", writer=html_writer)

    return fig, ax, ani

def bike_following_animation(data: DataStorage, output: str, angly=None, elevv=None) -> tuple[plt.Figure, plt.Axes, FuncAnimation]:

    plt.rcParams['animation.convert_path'] = r"C:\ImageMagick-7.1.1-47-Q16-HDRI-x64-dll\magick.exe"
    plt.rcParams['lines.linewidth'] = 4
    C = 2
    rider_angles = [3, 7, 8, 10] # this is to multiply the model angles to get a better view of the lean angles
    state_modified = data.solution_state.copy()
    state_modified[rider_angles, :] *= C
    # Create some functions to interpolate the results.
    x_eval = CubicSpline(data.time_array, state_modified.T) # possible multiplier for model angles.
    r_eval = CubicSpline(data.time_array, [[cf(t, x) for cf in data.simulator.inputs.values()]
                                           for t, x in zip(data.time_array, data.solution_state.T)])
    p, p_vals = zip(*data.constants.items())

    rear_cp = data.bicycle.rear_tire.contact_point
    front_cp = data.bicycle.front_tire.contact_point
    N = data.system.frame

    # Plot the initial configuration of the model
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6, 6))
    plotter = Plotter.from_model(data.bicycle_rider, ax=ax)#, linewidth=4)

    queue = [plotter]
    while queue:
        parent = queue.pop()
        if isinstance(parent, PlotBody):
            parent.plot_frame.visible = False
            parent.plot_masscenter.visible = False
        elif isinstance(parent, PlotVector):
            parent.visible = False
        else:
            queue.extend(parent.children)

    plotter.lambdify_system((data.system.q[:] + data.system.u[:], data.simulator.inputs.keys(), p))
    plotter.evaluate_system(x_eval(0), r_eval(0), p_vals)

    plotter.plot()
    ax.axis("off")
    _plot_ground(data, plotter)

    ax.invert_zaxis()
    ax.invert_yaxis()

    ax.set_aspect("equal")
    ax.axis("off")

    fps = 50
    track_x, track_y, track_z, FWCP_x, FWCP_y, FWCP_z = [], [], [], [], [], []

    def get_args(ti):
        x_val = x_eval(ti)
        r_val = r_eval(ti)
        x_pos, y_pos, yaw_ang, roll_ang = x_val[0], x_val[1], x_val[2], x_val[3]

        track_x.append(x_pos)
        track_y.append(y_pos)
        track_z.append(0)

        vec_fwcp = front_cp.pos_from(rear_cp)

        # Project into rear frame
        x_expr = vec_fwcp.dot(N.x)
        y_expr = vec_fwcp.dot(N.y)
        z_expr = vec_fwcp.dot(N.z)

        # Substitute numerical values
        subs_dict = dict(zip(data.system.q[:] + data.system.u[:], x_val))
        subs_dict.update(zip(data.simulator.inputs.keys(), r_val))
        subs_dict.update(zip(p, p_vals))

        dx = float(x_expr.subs(subs_dict))
        dy = float(y_expr.subs(subs_dict))
        dz = float(z_expr.subs(subs_dict))

        FWCP_x.append(x_pos + dx)
        FWCP_y.append(y_pos + dy)
        FWCP_z.append(0)

        # Plot the track
        ax.plot( FWCP_x, FWCP_y, FWCP_z, color='red', linewidth=1, linestyle='dashed')
        ax.plot(track_x, track_y, track_z, color='blue', linewidth=1, linestyle='dashed')
        ax.set_xlim(x_pos - 0.5, x_pos + 0.5)
        ax.set_ylim(y_pos - 1.5, y_pos + 1.5)
        ax.set_aspect("equal")
        azim = np.degrees(yaw_ang) # * C)
        roll = np.degrees(roll_ang)
        ax.view_init(elev=elevv, azim=(azim - 180 + angly), roll=0) #roll=-roll
        return x_val, r_eval(ti), p_vals

    ani = plotter.animate(
        get_args,
        frames=np.arange(0, data.time_array[175], 1 / (fps)),
        blit=False)
    # display(HTML(ani.to_jshtml(fps=fps)))
    #html_writer = HTMLWriter()
    #ani.save(output if output.endswith(".html") else output + ".html", writer=html_writer)
    # to make a GIF:
    ani.save(output if output.endswith(".gif") else output + ".gif", writer="imagemagick", fps=20, dpi=150)

    plt.close()

def bike_following_animation2(data: DataStorage, output: str, angly=None, elevv=None) -> tuple[plt.Figure, plt.Axes, FuncAnimation]:

    plt.rcParams['animation.convert_path'] = r"C:\ImageMagick-7.1.1-47-Q16-HDRI-x64-dll\magick.exe"
    plt.rcParams['lines.linewidth'] = 4
    C = 1.25
    rider_angles = [2, 3, 5, 7, 8, 9] # this is to multiply the model angles to get a better view of the lean angles
    
    # rider_angles = [1,1,1,1,1,1]
    
    state_modified = data.solution_state.copy()
    state_modified[rider_angles, :] *= C
    # Create some functions to interpolate the results.
    x_eval = CubicSpline(data.time_array, state_modified.T) # possible multiplier for model angles.
    r_eval = CubicSpline(data.time_array, [[cf(t, x) for cf in data.simulator.inputs.values()]
                                           for t, x in zip(data.time_array, data.solution_state.T)])
    p, p_vals = zip(*data.constants.items())

    # Plot the initial configuration of the model
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6, 6))
    plotter = Plotter.from_model(data.bicycle_rider, ax=ax)#, linewidth=4)

    queue = [plotter]
    while queue:
        parent = queue.pop()
        if isinstance(parent, PlotBody):
            parent.plot_frame.visible = False
            parent.plot_masscenter.visible = False
        elif isinstance(parent, PlotVector):
            parent.visible = False
        else:
            queue.extend(parent.children)

    plotter.lambdify_system((data.system.q[:] + data.system.u[:], data.simulator.inputs.keys(), p))
    plotter.evaluate_system(x_eval(0), r_eval(0), p_vals)

    plotter.plot()
    ax.axis("off")

    _plot_ground(data, plotter)

    ax.invert_zaxis()
    ax.invert_yaxis()

    ax.set_aspect("equal")
    ax.axis("off")

    fps = 50
    track_x, track_y, track_z, Q_loc = [], [], [], []

    def get_args(ti):
        x_val = x_eval(ti)
        r_val = r_eval(ti)
        x_pos, y_pos, yaw_ang, roll_ang = x_val[0], x_val[1], x_val[2], x_val[3]

        track_x.append(x_pos)
        track_y.append(y_pos)
        track_z.append(0)
        azim = np.degrees(yaw_ang)  # * C)
        roll = np.degrees(roll_ang)
        # Plot the track
        ax.plot(track_x, track_y, track_z, color='red', linewidth=1, linestyle='dashed')
        #ax.set_xlim(x_pos + 1.12*np.cos(yaw_ang) - 0.5, x_pos + 1.12*np.cos(yaw_ang) + 0.5)
        #ax.set_ylim(y_pos + 1.12*np.sin(yaw_ang) - 0.5, y_pos + 1.12*np.sin(yaw_ang) + 0.5)
        ax.set_xlim(x_pos - 0.5, x_pos + 0.5)
        ax.set_ylim(y_pos - 1.25, y_pos + 1.25)
        ax.set_aspect("equal")

        ax.view_init(elev=(elevv), azim=(angly), roll=0) #roll=-roll
        return x_val, r_eval(ti), p_vals

    ani = plotter.animate(
        get_args,
        frames=np.arange(0, data.time_array[175], 1 / (fps)),
        blit=False)
    # display(HTML(ani.to_jshtml(fps=fps)))
    #html_writer = HTMLWriter()
    #ani.save(output if output.endswith(".html") else output + ".html", writer=html_writer)
    # to make a GIF:
    ani.save(output if output.endswith(".gif") else output + ".gif", writer="imagemagick", fps=20, dpi=150)

    plt.close()

def bike_following_timelapse(data: DataStorage, n_frames: int = 5
                      ) -> tuple[plt.Figure, plt.Axes]:

    # Create some functions to interpolate the results.
    x_eval = CubicSpline(data.time_array, (data.solution_state.T))
    r_eval = CubicSpline(data.time_array, [[cf(t, x) for cf in data.simulator.inputs.values()]
                                           for t, x in zip(data.time_array, data.solution_state.T)])
    p, p_vals = zip(*data.constants.items())

    # Plot the initial configuration of the model
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6, 6))
    plotter = Plotter.from_model(data.bicycle, ax=ax, color="gray", linewidth=1.5)
    plotter = Plotter.from_model(data.bicycle_rider, ax=ax, color="black", linewidth=3)
    plotter.lambdify_system((data.system.q[:] + data.system.u[:], data.simulator.inputs.keys(), p))
    queue = [plotter]
    while queue:
        parent = queue.pop()
        if isinstance(parent, PlotBody):
            parent.plot_frame.visible = False
            parent.plot_masscenter.visible = False
        elif isinstance(parent, PlotVector):
            parent.visible = False
        else:
            queue.extend(parent.children)
    plotter.evaluate_system(x_eval(0), r_eval(0), p_vals)
    for i in range(n_frames):
        for artist in plotter.artists:
            artist.set_alpha(i * 1 / (n_frames + 1) + 1 / n_frames)
            ax.add_artist(copy(artist))
        time = i / (n_frames - 1) * data.time_array[10]
        plotter.evaluate_system(x_eval(time), r_eval(time), p_vals)
        plotter.update()
    for artist in plotter.artists:
        artist.set_alpha(1)
        ax.add_artist(copy(artist))
    _plot_ground(data, plotter)
    ax.invert_zaxis()
    ax.invert_yaxis()
    ax.view_init(elev=0, azim=180, roll=0)
    ax.set_aspect("equal")
    ax.axis("off")

    return fig, ax



def create_time_lapse(data: DataStorage, n_frames: int = 6
                      ) -> tuple[plt.Figure, plt.Axes]:
    x_eval = CubicSpline(data.time_array, data.solution_state.T)
    r_eval = CubicSpline(data.time_array, data.solution_input.T)
    p, p_vals = zip(*data.constants.items())
    cm = 1/2.54
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(80*cm, 80*cm))
    plotter = Plotter.from_model(data.bicycle_rider, ax=ax)
    plotter.lambdify_system((data.x[:], data.input_vars[:], p))
    queue = [plotter]
    while queue:
        parent = queue.pop()
        if isinstance(parent, PlotBody):
            parent.plot_frame.visible = False
            parent.plot_masscenter.visible = False
        elif isinstance(parent, PlotVector):
            parent.visible = False
        else:
            queue.extend(parent.children)
    plotter.evaluate_system(x_eval(0), r_eval(0), p_vals)
    for i in range(n_frames):
        for artist in plotter.artists:
            artist.set_alpha(i * 1 / (n_frames + 1) + 1 / n_frames)
            ax.add_artist(copy(artist))
        time = i / (n_frames - 1) * data.time_array[-1]
        plotter.evaluate_system(x_eval(time), r_eval(time), p_vals)
        plotter.update()
    for artist in plotter.artists:
        artist.set_alpha(1)
        ax.add_artist(copy(artist))
    _plot_ground(data, plotter)
    ax.invert_zaxis()
    ax.invert_yaxis()
    ax.view_init(elev=7, azim=-2.5, roll=0)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax

def create_time_lapse_front(data: DataStorage, n_frames: int = 7
                      ) -> tuple[plt.Figure, plt.Axes]:
    x_eval = CubicSpline(data.time_array, data.solution_state.T)
    r_eval = CubicSpline(data.time_array, data.solution_input.T)
    p, p_vals = zip(*data.constants.items())

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(30, 30))
    plotter = Plotter.from_model(data.bicycle_rider, ax=ax)
    plotter.lambdify_system((data.x[:], data.input_vars[:], p))
    queue = [plotter]
    while queue:
        parent = queue.pop()
        if isinstance(parent, PlotBody):
            parent.plot_frame.visible = False
            parent.plot_masscenter.visible = False
        elif isinstance(parent, PlotVector):
            parent.visible = False
        else:
            queue.extend(parent.children)
    plotter.evaluate_system(x_eval(0), r_eval(0), p_vals)
    for i in range(n_frames):
        for artist in plotter.artists:
            artist.set_alpha(i * 1 / (n_frames + 1) + 1 / n_frames)
            ax.add_artist(copy(artist))
        time = i / (n_frames - 1) * data.time_array[-1]
        plotter.evaluate_system(x_eval(time), r_eval(time), p_vals)
        plotter.update()
    for artist in plotter.artists:
        artist.set_alpha(1)
        ax.add_artist(copy(artist))
    _plot_ground(data, plotter)
    ax.invert_zaxis()
    ax.invert_yaxis()
    ax.view_init(elev=5, azim=0)
    ax.set_aspect("equal")
    ax.axis("off")

    return fig, ax



def _plot_ground(data: DataStorage, plotter: Plotter):
    q1_arr = data.solution_state[data.system.q[:].index(data.bicycle.q[0]), :]
    q2_arr = data.solution_state[data.system.q[:].index(data.bicycle.q[1]), :]
    p, p_vals = zip(*data.constants.items())

    front_contact_coord = data.bicycle.front_tire.contact_point.pos_from(
        plotter.zero_point).to_matrix(plotter.inertial_frame)[:2]
    eval_fc = sm.lambdify((data.system.q[:] + data.system.u[:], p), front_contact_coord,
                          cse=True)
    fc_arr = np.array(eval_fc(data.solution_state, p_vals))
    x_min = min((fc_arr[0, :].min(), q1_arr.min()))
    x_max = max((fc_arr[0, :].max(), q1_arr.max()))
    y_min = min((fc_arr[1, :].min(), q2_arr.min()))
    y_max = max((fc_arr[1, :].max(), q2_arr.max()))
    X, Y = np.meshgrid(np.arange(x_min - 5, x_max + 1, 0.5),
                       np.arange(y_min - 1, y_max + 1, 0.5))
    plotter.axes.plot_wireframe(X, Y, np.zeros_like(X), color="k", alpha=0.3, rstride=1,
                                cstride=1)
    plotter.axes.set_xlim(X.min(), X.max())
    plotter.axes.set_ylim(Y.min(), Y.max())


def plot_model_figures(data: DataStorage, frames: list[int]) -> tuple[plt.Figure, plt.Axes]:
    cm = 1 / 2.54  # Convert inches to cm
    figures = []
    C = 1 # model angle multiplier
    # Create interpolation functions
    x_eval = CubicSpline(data.time_array, C * data.solution_state.T)
    r_eval = CubicSpline(data.time_array, data.solution_input.T)
    p, p_vals = zip(*data.constants.items())

    for frame in frames:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 9))
        time = data.time_array[frame]

        # Create plotter for the current timestep
        plotter = Plotter.from_model(data.bicycle_rider, ax=ax, color="black", linewidth=3)
        plotter.lambdify_system((data.system.q[:] + data.system.u[:], data.simulator.inputs.keys(), p))

        #bicycle_plotter = Plotter.from_model(data.bicycle, ax=ax, color="gray", linewidth=1.0)
        #bicycle_plotter.lambdify_system((data.system.bicycle.q[:] + data.system.u[:], data.simulator.inputs.keys(), p))
        #rider_plotter = Plotter.from_model(data.rider, ax=ax, color="black", linewidth=3.0)
        #rider_plotter.lambdify_system((data.system.rider.q[:] + data.system.u[:], data.simulator.inputs.keys(), p))
        # Hide frame vectors and mass center points
        queue = [plotter]
        while queue:
            parent = queue.pop()
            if isinstance(parent, PlotBody):
                parent.plot_frame.visible = False
                parent.plot_masscenter.visible = False
            elif isinstance(parent, PlotVector):
                parent.visible = False
            else:
                queue.extend(parent.children)
        x_val = x_eval(time)
        r_val = r_eval(time)

        #for plotter in [bicycle_plotter, rider_plotter]:
        #    queue = [plotter]
        #    while queue:
        #        parent = queue.pop()
        #        if isinstance(parent, PlotBody):
        #            parent.plot_frame.visible = False
        #            parent.plot_masscenter.visible = False
        #        elif isinstance(parent, PlotVector):
        #            parent.visible = False
        #        else:
        #            queue.extend(parent.children)

        plotter.evaluate_system(x_val, r_val, p_vals)
        plotter.plot()
        for child in plotter.children:
            if isinstance(child, PlotBody):
                if child.plot_geometry is not None:
                    child.plot_geometry.set_color("red")
                    child.plot_geometry.set_linewidth(4)
        plotter.update()
        # Evaluate system at the given time frame

        #plotter.evaluate_system(x_val, r_val, p_vals)
        #plotter.plot()
        #plotter.update()

        # Ground mesh centered around current X and Y
        pos_x = x_val[0] #/ C
        pos_y = x_val[1] #/ C
        X, Y = np.meshgrid(np.arange(pos_x - 1, pos_x + 1, 1),
                           np.arange(pos_y - 1, pos_y + 1.5, 1))
        ax.plot_wireframe(X, Y, np.zeros_like(X), color="k", alpha=0.15, rstride=1, cstride=1)

        heading_deg = np.degrees(x_val[2]) # / C)
        azim_angle = (180 - heading_deg) % 360

        # Set camera to follow the model
        ax.set_xlim(pos_x - 0.5, pos_x + 0.5)
        ax.set_ylim(pos_y - 1, pos_y + 1)
        #ax.set_aspect("equal")

        ax.invert_zaxis()
        ax.invert_yaxis()
        ax.view_init(elev=3, azim=(azim_angle + 180), roll=0)
        ax.axis("off")

        figures.append((fig, ax))


    return figures
