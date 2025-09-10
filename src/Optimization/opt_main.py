from __future__ import annotations

import json
import os

import cloudpickle as cp
import matplotlib.pyplot as plt
import numpy as np

from opt_container import DataStorage, Metadata, SteerWith, SeatType, TorsoType, Task, Model, InitGuess
from src.Optimization.opt_model import set_simulator, set_model
from src.Optimization.opt_problem import set_problem, set_constraints, set_initial_guess

from src.Optimization.opt_utils import NumpyEncoder, Timer, create_time_lapse, create_animation, create_plots

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SRC_DIR, "data")
OUTPUT_DIR = os.path.join(SRC_DIR, "AnimationSims")
i = 0
while os.path.exists(os.path.join(OUTPUT_DIR, f"result{i}")):
    i += 1
DEFAULT_RESULT_DIR = os.path.join(OUTPUT_DIR, f"result{i}")

LATERAL_DISPLACEMENT = 6
STRAIGHT_LENGTH = 6
TURN_RADIUS = 6
DURATION = 5
NUM_NODES = int(DURATION * 50)
print('NUM_NODES:', NUM_NODES)
tracking_weight = 1000
torque_weight = 1
verticality_weight = 1

trajectory_cost = tracking_weight ** 2       # Aimed path cost
control_cost = torque_weight ** 2            # Estimated input cost
verticality_cost = verticality_weight ** 2   # Cost to stay upright

#weight_tr and weight_ct are only to be used when trajectory, torque and verticality are minimized simultaneously
WEIGHT_TR = trajectory_cost / (verticality_cost + control_cost + trajectory_cost)
WEIGHT_CT = control_cost / (verticality_cost + control_cost + trajectory_cost)
WEIGHT = trajectory_cost / (control_cost + trajectory_cost)

METADATA = Metadata(
    bicycle_only=False,
    model_upper_body=True,
    model_legs=False,
    model_torso=True,
    model_head=True,
    sprung_steering=False,
    model=Model.SINGLE_PENDULUM,
    task=Task.STRAIGHT_TURN,
    steer_with=SteerWith.UPPER_BODY_TORQUE,
    parameter_data_dir=DATA_DIR,
    seat_type=SeatType.SIDELEAN,
    torso_type=TorsoType.FIXED,
    init_guess=InitGuess.RANDOM,
    bicycle_parametrization="Browser",
    rider_parametrization="Jason",
    duration=DURATION,
    lateral_displacement=LATERAL_DISPLACEMENT,
    straight_length=STRAIGHT_LENGTH,
    turn_radius=TURN_RADIUS,
    num_nodes=NUM_NODES,
    weight=WEIGHT,
    weight_tr=WEIGHT_TR,
    weight_ct=WEIGHT_CT
)

if __name__ == "__main__":
    timer = Timer()
    if not os.path.exists(DEFAULT_RESULT_DIR):
        os.mkdir(DEFAULT_RESULT_DIR)
    with open(os.path.join(DEFAULT_RESULT_DIR, "README.md"), "w") as f:
        f.write(f"# Result {i}\n## Metadata\n{METADATA}\n")
    data = DataStorage(METADATA)
    REUSE_LAST_MODEL = True
    if REUSE_LAST_MODEL and os.path.exists("last_model.pkl"):
        with timer("Reloading last model"):
            with open("last_model.pkl", "rb") as f:
                data = cp.load(f)
    else:
        with timer("Computing the equations of motion"):
            set_model(data)
        with timer("Initializing the simulator"):
            set_simulator(data)
        with open("last_model.pkl", "wb") as f:
            cp.dump(data, f)
    with timer("Defining the constraints and objective"):
        set_constraints(data)
    with timer("Making an initial guess"):
        set_initial_guess(data)
    with timer("Initializing the Problem object"):
        set_problem(data)
    data.problem.add_option("output_file",
                            os.path.join(DEFAULT_RESULT_DIR, "ipopt.txt"))
    with timer("Solving the problem"):
        data.solution, info = data.problem.solve(data.initial_guess)
    timer.to_file(os.path.join(DEFAULT_RESULT_DIR, "timings.txt"))
    print("Estimated torque:",
          np.sqrt(data.metadata.interval_value * (data.solution_input ** 2).sum() /
                  data.metadata.duration))
    print("Plotting the results...")
    data.problem.plot_objective_value()
    data.problem.plot_trajectories(data.solution)
    data.problem.plot_constraint_violations(data.solution)
    with open(os.path.join(DEFAULT_RESULT_DIR, "solution_info.txt"), "w",
              encoding="utf-8") as f:
        json.dump(info, f, cls=NumpyEncoder)
    with open(os.path.join(DEFAULT_RESULT_DIR, "data.pkl"), "wb") as f:
        cp.dump(data, f)
    #data.problem.plot
    create_plots(data)
    create_time_lapse(data, n_frames=7)
    create_animation(data, os.path.join(DEFAULT_RESULT_DIR, "animation.gif"))

    for i in plt.get_fignums():
        plt.figure(i).savefig(os.path.join(DEFAULT_RESULT_DIR, f"figure{i}.png"))

    plt.show()