from __future__ import annotations

import argparse
import json
import os

import cloudpickle as cp
import matplotlib.pyplot as plt
import numpy as np
import sympy as sm
from opt_container import DataStorage, Metadata, SeatType, TorsoType, SteerWith, Task, Model, InitGuess
from src.Optimization.opt_main import (
    LATERAL_DISPLACEMENT, STRAIGHT_LENGTH, NUM_NODES, DURATION,
    WEIGHT, WEIGHT_CT, WEIGHT_TR, DATA_DIR, DEFAULT_RESULT_DIR, OUTPUT_DIR, TURN_RADIUS
)
from src.Optimization.opt_model import set_model, set_simulator
from src.Optimization.opt_problem import set_constraints, set_initial_guess, set_problem
from src.Optimization.opt_utils import (
    EnumAction, NumpyEncoder, Timer, create_animation, create_animation2, create_plots, create_time_lapse,
    create_time_lapse_front, check_config, bike_following_animation, bike_following_animation2, bike_following_timelapse, plot_model_figures)

parser = argparse.ArgumentParser(description="Run a trajectory tracking problem.")
parser.add_argument("--bicycle-only", action="store_true",
                    help="Use a bicycle-only model.")
parser.add_argument("--model-upper-body", default=True, action="store_true",
                    help="Use a model with an upper body.")
parser.add_argument("--model-torso", default=True, action="store_true",
                    help="Use a model with a torso")
parser.add_argument("--model-head", default=True, action="store_true",
                    help="Use a model with a head")
parser.add_argument("--model-legs", default=False, action="store_true",
                    help="Use a model with legs")
parser.add_argument("--sprung_steering", default=False, action="store_true",
                    help="Use a model with a spring between the front and rear frames.")
parser.add_argument("--model", type=Model,
                    default=Model.DOUBLE_PENDULUM, action=EnumAction)
parser.add_argument("--task", type=Task,
                    default=Task.DOUBLE_LANE_SWITCH, action=EnumAction)
parser.add_argument("--seat-type", type=SeatType,
                    default=SeatType.SHIFTINGSIDELEAN, action=EnumAction)
parser.add_argument("--steer-with", type=SteerWith,
                    default=SteerWith.SEAT_AND_TORSO_TORQUE, action=EnumAction)
parser.add_argument("--init_guess", type=InitGuess,
                    default=InitGuess.ZEROS, action=EnumAction)
parser.add_argument("--torso-type", type=TorsoType,
                    default=TorsoType.PIN, action=EnumAction) # The option FIXED doesn't really do anything atm, if SINGLE_PENDULUM and SEAT_TORQUE are activated, then the torsojoint will automatically be fixed.
parser.add_argument("--bicycle-parametrization", type=str, default="Browser",
                    help="The parametrization of the bicycle model.")
parser.add_argument("--rider-parametrization", type=str, default="Jason",
                    help="The parametrization of the rider model.")
parser.add_argument("--duration", type=float, default=DURATION,
                    help="The time duration of the trajectory.")
parser.add_argument("--lateral-displacement", type=float,
                    default=LATERAL_DISPLACEMENT, help="The lateral displacement of the trajectory.")
parser.add_argument("--straight-length", type=float, default=STRAIGHT_LENGTH,
                    help="The length of the straight sections at the start and end of the trajectory.")
parser.add_argument("--turn-radius", type=float, default=TURN_RADIUS,
                    help="The radius of the turn of the 90 deg turn task")
parser.add_argument("--num-nodes", type=int, default=NUM_NODES,
                    help="The number of nodes in the optimization problem.")
parser.add_argument("--weight", type=float, default=WEIGHT,
                    help="The weight of the path objective [0, 1].")
parser.add_argument("--weight_tr", type=float, default=WEIGHT_TR,
                    help="The weight of the tracking objective in case there are 3 cost variables [0, 1].")
parser.add_argument("--weight_ct", type=float, default=WEIGHT_CT,
                    help="The weight of the control objective in case there are 3 cost variables [0, 1].")
parser.add_argument("--parameter-data-dir", type=str, default=DATA_DIR,
                    help="The directory containing the parameter data.")
parser.add_argument('-o', '--output', type=str, default=DEFAULT_RESULT_DIR,
                    help="The directory to save the results in.")
parser.add_argument('--reuse-model', default=False, action="store_true",
                    help="Whether the last model should be reused if it is the same.")

timer = Timer()
result_dir = parser.parse_args().output
last_model_path = os.path.join(OUTPUT_DIR, "last_model.pkl")
METADATA = Metadata(**{
    k: v for k, v in vars(parser.parse_args()).items()
    if k in Metadata.__dataclass_fields__})



print("Running optimization with the following metadata:", METADATA, sep="\n")
print("Saving results to:", result_dir)
check_config(DataStorage(METADATA))

if not os.path.exists(result_dir):
    os.mkdir(result_dir)
with open(os.path.join(result_dir, "README.md"), "w") as f:
    f.write(f"# Result\n## Metadata\n{METADATA}\n")

if parser.parse_args().reuse_model and os.path.exists(last_model_path):
    with timer("Reloading last model"):
        with open(last_model_path, "rb") as f:
            last_data = cp.load(f)
        if last_data.metadata == METADATA:
            data = last_data
        else:
            print("Last model does not match the current metadata, recomputing...")
            data = DataStorage(METADATA)
else:
    data = DataStorage(METADATA)

if data.system is None:
    with timer("Computing the equations of motion"):
        set_model(data)
if data.simulator is None:
    with timer("Initializing the simulator"):
        set_simulator(data)
with timer("Defining the constraints and objective"):
    set_constraints(data)

with timer("Making an initial guess"):
    set_initial_guess(data)
    print('obj_expr,, constraints:', data.objective_expr, data.constraints)


with timer("Initializing the Problem object"):
    set_problem(data)
data.problem.add_option("output_file", os.path.join(result_dir, "ipopt.txt"))


with timer("Solving the problem"):
    print('data.problem: ', data.problem)

    data.solution, info = data.problem.solve(data.initial_guess)
timer.to_file(os.path.join(result_dir, "timings.txt"))

print("Plotting the results...")
data.problem.plot_objective_value()
data.problem.plot_trajectories(data.solution)
data.problem.plot_constraint_violations(data.solution)

plots_and_numbers = create_plots(data)


print(data.time_array.shape, data.solution_state.shape, data.solution_input.shape,)
optimization_data = np.vstack((data.time_array, data.solution_state, data.solution_input,
                               plots_and_numbers[3], plots_and_numbers[4], plots_and_numbers[5]))
AA, BB, CC, TARR= sm.Matrix([sm.Symbol("energy")]), sm.Matrix([sm.Symbol("speed_data")]), sm.Matrix([sm.Symbol("tracking error")]), sm.Matrix([sm.Symbol("time")])
optimization_dict = sm.Matrix([TARR]).col_join(data.system.q.col_join(data.system.u)).col_join(data.r).col_join(AA).col_join(BB).col_join(CC)

print('data shaperiño ->', optimization_data.shape)
print('data dict ->', optimization_dict)
with open(os.path.join(result_dir, "solution_data.pkl"), "wb") as f:
    cp.dump(optimization_data, f)

with open(os.path.join(result_dir, "solution_info.txt"), "w", encoding="utf-8") as f:
    json.dump(info, f, cls=NumpyEncoder)
data_dict = {}  # Create a regular Python dictionary

for i, row in enumerate(optimization_dict.tolist()):
    key_name = str(row[0])
    data_dict[key_name] = optimization_data[i, :].tolist()

with open(os.path.join(result_dir, "solution_info_dictionary.txt"), "w") as f:
    json.dump(data_dict, f, indent=4)



with open(os.path.join(result_dir, "data.pkl"), "wb") as f:
    cp.dump(data, f)
with open(last_model_path, "wb") as f:
    cp.dump(data, f)


with open(os.path.join(result_dir, "performance_metrix.txt"), "w") as f:
    f.write(f"Model configuration is as follows: \n"
            f"{METADATA}"
            f"\n"
            f"# Performance Metrics #\n"
            f"Total energy used: {plots_and_numbers[0]} [J] \n"
            f"RMS tracking error: {plots_and_numbers[1]} [m] \n"
            f"Average bicycle speed: {plots_and_numbers[2]} [kmh] \n"
            f"Data for the plotter -> Length: {len(optimization_dict)} \n"
            f"List: {optimization_dict} \n"
            f"Tracking weight: {np.rint(np.sqrt(1/(1 / data.metadata.weight - 1)))}"
            )

print("Creating the animations (?)")
create_time_lapse(data, n_frames=7)
create_time_lapse_front(data, n_frames=7)
#bike_following_animation2(data, os.path.join(result_dir, "bike following animation2_angle"), angly=0, elevv=5)
print('ello!')
bike_following_animation(data, os.path.join(result_dir, "bike following animation"), angly=0, elevv=4)

#bike_following_timelapse(data, n_frames=3)
#plot_model_figures(data, frames=[0, 22, 40, 80])
#create_animation(data, os.path.join(result_dir, "animation"))
#create_animation2(data, os.path.join(result_dir, "animation_front"))

for i in plt.get_fignums():
    plt.figure(i).savefig(os.path.join(result_dir, f"figure{i}.png"), dpi=200)