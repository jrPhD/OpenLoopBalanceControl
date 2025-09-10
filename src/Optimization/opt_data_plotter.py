import os
import numpy as np
import json
import re
import matplotlib.pyplot as plt
import cloudpickle as cp
import itertools
import pickle
import scienceplots as scp
from matplotlib.lines import Line2D
from enum import Enum, auto, unique
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from symbrim.core.base_classes import BrimBase
from symbrim.utilities.plotting import Plotter
from matplotlib.animation import FuncAnimation, HTMLWriter, PillowWriter, FFMpegWriter
from scipy.interpolate import CubicSpline
from symmeplot.matplotlib import PlotBody, PlotVector
import numpy as np
import pandas as pd
import sympy as sm
import scipy as sp
import sympy.physics.mechanics as me
from sympy import symbols, sin, cos, Symbol, Matrix
from symbrim.utilities.plotting import Plotter
from matplotlib.animation import FuncAnimation, HTMLWriter, PillowWriter, FFMpegWriter
from scipy.interpolate import CubicSpline
from symmeplot.matplotlib import PlotBody, PlotVector
from opt_container import DataStorage, Metadata
from symbrim.utilities.plotting import Plotter
from matplotlib.animation import FuncAnimation, HTMLWriter, PillowWriter, FFMpegWriter

import bicycleparameters as bp
from bicycleparameters import tables, plot
from bicycleparameters.models import Meijaard2007Model
from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
from uncertainties import nominal_value

import numpy as np
import matplotlib.pyplot as plt


@unique
class Unit(Enum):
    TORQUE = auto()
    ANGLE = auto()
    ANGULARRATE = auto()
    VELOCITY = auto()
    ENERGY = auto()
    DISTANCE = auto()

@unique
class Task(Enum):
    DOUBLE = auto()
    PERTURBED = auto()
    TURN = auto()



#OUTPUT_DIR = "output_PerturbedCycling"
OUTPUT_DIR = "output_DoubleLaneSwitch"
PLOTS_DIR = "post_process_plots"

plt.style.use(['science', 'high-contrast', 'no-latex'])
cm = 1/2.54 # get centimeters in inches
startt = 1
endd = 4
selected_results = [0, 6]
custom_names = ["Seat Lean", "Torso Lean"]#"Single Pendulum", "Double Pendulum", "Triple Pendulum", "Single Pendulum", "Double Pendulum", "Triple Pendulum"]
row_names = ["Seat Lean", "Torso Lean"]#, "PinJoint Seat", "CombiJoint Seat", "CombiJoint Seat", "CombiJoint Seat"]#, "Seat Angular Velocity"]

# This thing determines which multiplication factor to use when plotting.
Unit = Unit.ENERGY
Task = Task.TURN

# In case different data rows from the solution data have to be plotted per optimization (data matrices from different models have different lengths):
per_result_rows_example = {
    "result1_singlepen_sidelean_14.2kmh_dur7.8_dt0.02": [19],
    "result2_doublepen_sidelean_14.2kmh_dur7.8_dt0.02": [19],
    "result3_triplepen_sidelean_14.2kmh_dur7.8_dt0.02": [19],
}

# in case i want to use different multiplication factors for plotting different data types in the same plot:
row_scaling_factors = {
    19: 1,               # Seat Torque (no scaling)
    20: 1,               # Energy (no scaling)
    #13: (180/np.pi),      # Seat Angular Velocity (rad/s -> deg/s)
    #21: -0.34433,        # Some velocity-related scaling
    #22: (180/np.pi),     # Angular rate
}

def get_result_dirs(start=None, end=None, selected_indices=None):
    """ Returns sorted list of result directories in the range resultX (X=start to end) """
    all_dirs = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("result") and d[6:].split("_")[0].isdigit()]
    sorted_dirs = sorted(all_dirs, key=lambda x: int(x[6:].split("_")[0]))

    if selected_indices:
        return [d for d in sorted_dirs if int(d[6:].split("_")[0]) in selected_indices]
    if start is not None and end is not None:
        return sorted_dirs[start - 1: end]

    return sorted_dirs

def load_solution_data(result_dir):
    """Loads solution_data.pkl from a given result directory."""
    file_path = os.path.join(OUTPUT_DIR, result_dir, "solution_data.pkl")

    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found!")
        return None

    with open(file_path, "rb") as file:
        return cp.load(file)  # Load the cloudpickle file

def plot_selected_data(start=startt, end=endd, row_indices=None, per_result_rows=None, save_plot=True, custom_names=custom_names, row_names=row_names):
    """ Plots selected rows from solution matrices in multiple result directories """
    # The next 6 lines used to be outside this function, not sure if it still works like this.
    result_dirs = get_result_dirs(start=startt, end=endd)
    #print(result_dirs)
    for result_dir in result_dirs:
        data_matrix = load_solution_data(result_dir)  # ✅ Pass a single directory name
        if data_matrix is not None:
            print(f"Loaded data from {result_dir}: shape {data_matrix.shape}")

    result_dirs = get_result_dirs(start, end)
    if not result_dirs:
        print("No result directories found in the specified range.")
        return

    if custom_names and len(custom_names) != len(result_dirs):
        print("Warning: Number of custom names does not match number of result directories. Using default names.")
        custom_names = None

    plt.figure(figsize=(9*cm, 6*cm), dpi=100)
    custom_colors = ["green", "red", "blue"]#, "green", "red", "blue", "green"]#, "orange"]#, "orange", "purple", "cyan", "brown"]#, "brown", "magenta", "black"]
    # The following two lines give each result directory that is plotted their own color, pick if you want to manually select the colours or not
    color_cycle = plt.cm.gist_rainbow  # This thing picks a colour map
    #colors = [color_cycle(i / max(1, len(result_dirs) - 1)) for i in range(len(result_dirs))]
    colors = [custom_colors[i % len(custom_colors)] for i in range(len(result_dirs))]
    # And this line gives each type of data its own linestyle, change depending on the amount of data things plotted

    linestyles = itertools.cycle(["solid", "solid"])#, "solid", "dashed", "dashed", "dashed"])#, "solid", "dashed", "solid", "dashed"])#, "dashdot"])#, "dotted"])#, "dotted", "dashdot"])
    row_names_cycle = itertools.cycle(row_names)

    left_legend_handles = []
    right_legend_handles = []
    added_variable_names = set()
    added_base_names = set()

    row_name_linestyle = {"Seat Lean": "solid", "Torso Lean": "solid"}

    for i, result_dir in enumerate(result_dirs):
        data_matrix = load_solution_data(result_dir)
        if data_matrix is None:
            continue

        #label_name = custom_names[i] if custom_names else result_dir
        base_name = custom_names[i] if custom_names else result_dir
        color = colors[i]

        # Determine which rows to plot for this result_dir
        if per_result_rows and result_dir in per_result_rows:
            selected_rows = per_result_rows[result_dir]  # Use custom row indices
        else:
            selected_rows = row_indices  # Use the general row indices

        for row in selected_rows:
            if row >= data_matrix.shape[0]:  # Ensure row exists
                print(f"Warning: {custom_names} {row} not found in {result_dir}. Skipping.")
                continue

            #variable_name = next(row_names_cycle)
            #linestyle = next(linestyles)
            variable_name = row_names[i]
            linestyle = row_name_linestyle[variable_name]  # Use linestyle based on row_name (10 km/h or 14 km/h)

            # The following "commented" option must be used if different data types are plotted together,
            # and use the row_scaling_factors function to map the according mutiplications to the right data type.
            """
            factor = row_scaling_factors.get(row, 1)
            plt.plot(data_matrix[1, :], data_matrix[row, :] * factor,
                     label=f"{base_name} - {variable_name}", color=color, linestyle=linestyle)
            """
            if Unit == Unit.TORQUE:
                plt.plot(data_matrix[0, :], data_matrix[row, :], label=f"{base_name} - {variable_name}",
                        color=color, linestyle=linestyle)
            elif Unit == Unit.ANGLE:
                plt.plot(data_matrix[0, :], data_matrix[row, :] * (180/np.pi), label=f"{base_name} - {variable_name}",
                         color=color, linestyle=linestyle)
            elif Unit == Unit.ANGULARRATE:
                plt.plot(data_matrix[0, :], data_matrix[row, :] * (180/np.pi), label=f"{base_name} - {variable_name}",
                         color=color, linestyle=linestyle)
            elif Unit == Unit.VELOCITY:
                plt.plot(data_matrix[0, :], data_matrix[row, :] * (-0.34096), label=f"{base_name} - {variable_name}",
                         color=color, linestyle=linestyle)
            elif Unit == Unit.ENERGY:
                plt.plot(data_matrix[0, :], data_matrix[row, :], label=f"{base_name} - {variable_name}",
                         color=color, linestyle=linestyle)
            elif Unit == Unit.DISTANCE:
                plt.plot(data_matrix[0, :], data_matrix[row, :], label=f"{base_name} - {variable_name}",
                         color=color, linestyle=linestyle)
            #"""

            # Add to the left legend if this variable_name hasn't been added yet
            if variable_name not in added_variable_names:
                left_legend_handles.append(Line2D([0], [0], color='black', linestyle=linestyle, label=variable_name))
                added_variable_names.add(variable_name)

            # Add to the right legend if this base_name hasn't been added yet
            if base_name not in added_base_names:
                right_legend_handles.append(Line2D([0], [0], color=color, lw=4, linestyle=linestyle, label=base_name))
                added_base_names.add(base_name)


    # Set the left and right column legends
    #first_legend = plt.legend(handles=left_legend_handles, loc="upper left", bbox_to_anchor=(0.45, 1),
    #                              fontsize=10)
    #second_legend = plt.legend(handles=right_legend_handles, loc="upper left", bbox_to_anchor=(0.45, 0.35),
    #                            fontsize=10)

    #plt.gca().add_artist(first_legend)
    #plt.gca().add_artist(second_legend)

    #plt.xlabel("Distance [m]")
    plt.xlabel("Time [s]")
    if Unit == Unit.TORQUE:
        plt.ylabel("Torque [Nm]")
    elif Unit == Unit.ANGLE:
        plt.ylabel("Angle [deg]")
    elif Unit == Unit.ANGULARRATE:
        plt.ylabel("Angular Velocity [deg/s]")
    elif Unit == Unit.VELOCITY:
        plt.ylabel("Velocity [m/s]")
    elif Unit == Unit.ENERGY:
        plt.ylabel("Energy [J]")
    elif Unit == Unit.DISTANCE:
        plt.ylabel("Lateral displacement [m]") #Tracking Error [m]

    #plt.title(f"Energy usage during a double lane switch at 14 kmh")# from {start} to {end}")
    #plt.legend()
    plt.grid(True)

    # Save tha plot
    if save_plot:
        #plot_filename = os.path.join(PLOTS_DIR, f"plot of results_{start-1}_to_{end-1} over distance.png")
        plot_filename = os.path.join(PLOTS_DIR, f"{Unit} results_{start}_to_{end} from {OUTPUT_DIR}.png")
        plt.savefig(plot_filename)
        print(f"Plot saved: {plot_filename}")

def plot_single_array(save_plot=None):
    """ The code below is a plot of the two data arrays, and the axin stuff is about a zoomed in part of the plot."""
    k_data = np.array([-15, -10, -5, -2.5, -1, 0, 1, 2.5, 5, 10, 15, 20, 25, 30])
    W_data = np.array([12.7, 5.30, 1.29, 0.39, 0.21, 0.34, 0.66, 1.31, 1.83, 5.87, 11.0, 23.7, 40.3, 52.8])
    cm = 1 / 2.54  # get centimeters in inches

    fig, ax = plt.subplots(figsize=(18*cm, 12*cm))
    ax.scatter(k_data, W_data, s=100, color="dodgerblue")
    #plt.scatter(k_data, W_data, s=100, color="blue", label="Data points")  # 's' controls dot size

    # 5 lines below are about the trendline in the main plot:
    #coefficients = np.polyfit(k_data, W_data, deg=2)  # Change 'deg' for different polynomial fits
    #trendline = np.poly1d(coefficients)
    #x_fit = np.linspace(min(k_data), max(k_data), 100)  # Smooth trendline
    #y_fit = trendline(x_fit)
    #ax.plot(x_fit, y_fit, color="red", linestyle="--", label="Trendline")

    # plot two dotted lines to clarify where the insert box is from -> doesn't work smoothly yet
    plt.plot([-6, -12.65], [-0.5, 23], "k--", linewidth=0.5)  # First dotted line (black)
    plt.plot([6, 6.9], [-0.5, 23], "k--", linewidth=0.5)

    plt.xlabel("Spring Stiffness [k]")
    plt.ylabel("Energy [J]")
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    #plt.title("Energy required for different spring rates on the front frame")
    #plt.legend()
    axins = inset_axes(ax, width=5.5*cm, height=4*cm, bbox_to_anchor=(0.25, 0.65, 0.25, 0.25),
                       bbox_transform=ax.transAxes) ## adjust size and location
    # Set zoomed-in limits
    axins.set_xlim(-6, 6)  # Adjust zoom region for X
    axins.set_ylim(-0.5, 2.5)   # Adjust zoom region for Y

    # Scatter and trendline inside inset
    axins.scatter(k_data, W_data, s=50, color="dodgerblue")
    #axins.plot(x_fit, y_fit, color="red", linestyle="--") #trendline main plot thing
    mask = (k_data >= -6) & (k_data <= 6)
    k_data_inset = k_data[mask]
    W_data_inset = W_data[mask]
    # lines below are to create a trendline just within the inset plot
    coefficients_inset = np.polyfit(k_data_inset, W_data_inset, deg=2)
    trendline_inset = np.poly1d(coefficients_inset)
    x_fit_inset = np.linspace(min(k_data_inset), max(k_data_inset), 100)
    y_fit_inset = trendline_inset(x_fit_inset)
    axins.plot(x_fit_inset, y_fit_inset, color="red", linestyle="--", label="Inset Trendline")
    # lines below create and set gridlines for the inset plot
    axins.grid(True, linestyle="--", alpha=0.6)
    axins.set_xticks(np.arange(-5, 7.5, 2.5))  # X-ticks every 2.5 units
    axins.set_yticks(np.arange(0, 2.5, 0.5))   # Y-ticks every 0.5 units

    # box around the zoom-in section
    ax.indicate_inset_zoom(axins, edgecolor="red")

    if save_plot:
        #plot_filename = os.path.join(PLOTS_DIR, f"plot of results_{start-1}_to_{end-1} over distance.png")
        plot_filename = os.path.join(PLOTS_DIR, f"Plot single data array... .png")
        plt.savefig(plot_filename)
        print(f"Plot saved: {plot_filename}")

    plt.show()

def plot_multiple_arrays(save_plot=None):
    """Code below is intended to plot different arrays of data"""
    data_sets = {
        #"Spring = -10Nm/rad": {
        #    "E": np.array([23.11, 49.33, 38.05, 36.82, 32.80]),
        #    "m": np.array([0.219, 0.670, 0.718, 0.703, 0.243]),
        #    "v": np.array([12, 15, 18, 21.5, 25])},

        "No Spring": {
            "E": np.array([36.81, 21.57, 19.31, 16.49, 19.98]),
            "m": np.array([0.516, 0.469, 0.465, 0.426, 0.425]),
            "v": np.array([12, 15, 18, 21.5, 25])
        },
        "Spring k = 10 Nm/rad": {
            "E": np.array([31.87, 30.77, 22.66, 11.44, 13.19]),
            "m": np.array([0.407, 0.452, 0.404, 0.390, 0.368]),
            "v": np.array([12, 15, 18, 21.5, 25])
        },
        "Spring k = 20 Nm/rad": {
            "E": np.array([35.60, 31.69, 28.96, 17.33, 18.74]),
            "m": np.array([0.485, 0.442, 0.397, 0.376, 0.315]),
            "v": np.array([12, 15, 18, 21.5, 25])
        }
    }
    """data_sets = {
        "1": {
            "e": np.array([11.75, 8.46, 7.40, 6.16, 7.37, 13.85]),
            "m": np.array([8, 10, 12, 14, 16, 19.5]),
            "l":np.array([14, 11, 9.2, 7.8, 6.9, 5.7])},
        "2": {
            "e": np.array([21.39, 12.97, 11.29, 6.97, 10.93, 19.31]),
            "m": np.array([8, 10, 12, 14, 16, 19.5]),
            "l": np.array([14, 11, 9.2, 7.8, 6.9, 5.7])},
        "3": {
            "e": np.array([19.69, 12.61, 10.48, 7.53, 10.34, 17.92]),
            "m": np.array([8, 10, 12, 14, 16, 19.5]),
            "l": np.array([14, 11, 9.2, 7.8, 6.9, 5.7])},
        "4": {
            "e": np.array([5.82, 4.23, 2.96, 2.20, 2.86, 4.96]),
            "m": np.array([8, 10, 12, 14, 16, 19.5]),
            "l": np.array([14, 11, 9.2, 7.8, 6.9, 5.7])},
        "5": {
            "e": np.array([10.25, 6.77, 5.00, 3.39, 4.38, 12.70]),
            "m": np.array([8, 10, 12, 14, 16, 19.5]),
            "l": np.array([14, 11, 9.2, 7.8, 6.9, 5.7])},
        "6": {
            "e": np.array([9.51, 7.56, 5.35, 3.73, 3.69, 7.35]),
            "m": np.array([8, 10, 12, 14, 16, 19.5]),
            "l": np.array([14, 11, 9.2, 7.8, 6.9, 5.7])}
    } """
    colors = ["dodgerblue", "orange", "purple", "black", "lime", "brown"]  # Colors for each dataset
    marker_styles = {12: "o", 15: "s", 18: "^", 21.5: "D", 25: "P"}  # Different markers for each velocity

    fig, ax = plt.subplots(figsize=(16*cm, 12*cm))

    velocity_legend_handles = {}
    """
    for i, key in enumerate(data_sets):
        e = data_sets[key]["e"]
        m = data_sets[key]["m"]
        l = data_sets[key]["l"]
        #ax.plot(m, e, color=colors[i], marker='o')
        ax.plot(m, (e / l), color=colors[i], marker='o')
    # Loop through datasets and plot them
    for i, (label, data) in enumerate(data_sets.items()):
        ax.scatter(data["m"], ..., s=100, color=colors[i], label=label)  # Dots
        ax.plot(data["m"], ..., color=colors[i], linestyle="-", linewidth=2)  # Line
    """
    for i, (label, data) in enumerate(data_sets.items()):
        color = colors[i]
        ax.plot(data["m"], data["E"], color=color, linestyle="-", linewidth=1.5)
        for j in range(len(data["m"])):
            marker = marker_styles[data["v"][j]]
            sc = ax.scatter(data["m"][j], data["E"][j], s=100, color=color, marker=marker,
                            label=label if j == 0 else "")
            # Store marker legend handle (only once per velocity)
            if data["v"][j] not in velocity_legend_handles:
                velocity_legend_handles[data["v"][j]] = plt.Line2D([0], [0], color='black', marker=marker, linestyle='',
                                                                   markersize=10, label=f" {data['v'][j]}")



    # Labels and grid
    plt.xlabel("Velocity [km/h]") #plt.xlabel("RMS error [m]")
    plt.ylabel("Energy [J]")
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    # plt.title("")

    #plt.legend(loc="upper left", bbox_to_anchor=(0.25, 1), fontsize=10)  # Adjust location as needed

    # things below is to plot two legends
    dataset_legend = ax.legend(loc="upper left", bbox_to_anchor=(0.25, 1), fontsize=10)
    # Velocity legend (Marker style)
    velocity_legend = ax.legend(handles=velocity_legend_handles.values(), loc="upper right", bbox_to_anchor=(0.24, 1),
                                fontsize=10, title="Velocity [km/h]")

    # Add both legends
    ax.add_artist(dataset_legend)
    ax.add_artist(velocity_legend)



    if save_plot:
        # plot_filename = os.path.join(PLOTS_DIR, f"plot of results_{start-1}_to_{end-1} over distance.png")
        plot_filename = os.path.join(PLOTS_DIR, f"Plot energy vs rms.png")
        plt.savefig(plot_filename)
        print(f"Plot saved: {plot_filename}")

    plt.show()

def plot_single_sim(save_plot=True, x_label=None, y_label=None, x_limits=None, y_limits=None):

    result_dir = 'result6_NOCS_triplepen_sidelean_dur5_w1000_dt0.02_no_violations'
    rows_to_plot = [3, 7, 8, 9] # Selecting the required data rows
    labels = ["Bicycle roll", "Seat Lean", "Torso Lean", "Neck Lean"]  #  labels
    scales = [1,1,1,1]#[(180 / np.pi), (180 / np.pi), (180 / np.pi), (180/np.pi)] #  scaling factors
    line_styles = ['-.', '-', '-', '-']#, '-'] #  ['-', '-', '-']     #    line styles
    line_colors = ['k', 'r', 'g', 'b']#, 'g', 'b']                      # line colours

    solution_data = load_solution_data(result_dir)
    time = solution_data[0]    # 0=time, 1=q1 -> longitudinal distance

    num_rows = len(rows_to_plot)
    if labels is None or len(labels) != num_rows:
        labels = [f'Row {row}' for row in rows_to_plot]  # Default labels
    if scales is None or len(scales) != num_rows:
        scales = [1] * num_rows  # Default scale = 1
    if line_styles is None or len(line_styles) != num_rows:
        line_styles = ['-'] * num_rows  # Default solid line
    if line_colors is None or len(line_colors) != num_rows:
        line_colors = [None] * num_rows  # Use default Matplotlib colors

    plt.style.use(['science', 'ieee', 'no-latex'])
    plt.figure(figsize=(12*cm, 10*cm), dpi=150, layout="constrained")

    # Iterate over the selected rows and plot them
    for i, row_index in enumerate(rows_to_plot):
        if row_index >= len(solution_data):
            print(f"Error: Row index {row_index} out of bounds!")
            continue  # Skip invalid indices

        output = solution_data[row_index] * scales[i]  # Apply scaling

        plt.plot(time, output, label=labels[i], linestyle=line_styles[i], color=line_colors[i])

    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    #plt.title("Simulation Results")

    if x_limits:
        plt.xlim(x_limits)
    if y_limits:
        plt.ylim(y_limits)

    plt.legend()#loc="lower right", bbox_to_anchor=(1.035, -0.03))
    plt.grid(True)

    if save_plot:
        short_result_dir = result_dir[:12]  # Select first 12 characters
        plot_filename = os.path.join(PLOTS_DIR, f"plot_{short_result_dir}_modelangles.png")
        plt.savefig(plot_filename)
        print(f"Plot saved: {plot_filename}")
    plt.show()

def get_path(data: DataStorage, save_plot=None) -> None:
    OUTPUT_DIRR, result_dirr = "output_90degTurn", "result6_NOCS_triplepen_sidelean_dur5_w1000_dt0.02_no_violations"
    file_path = os.path.join(OUTPUT_DIRR, result_dirr, "solution_data.pkl")
    with open(file_path, "rb") as file:
        dataa = cp.load(file)  # Load the cloudpickle file
    x_data = dataa[1]
    y_data = dataa[2]
    s = 6
    d_lat = 2
    r = 6
    num_nodes = 250   # 50 * DURATION of the optimization that is plotted
    if Task == Task.DOUBLE:

        path_length = s + \
                      s + \
                      s + \
                      s + \
                      s
        path_len = np.linspace(0, path_length, num_nodes)
        q1_path = []
        q2_path = []

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
    elif Task == Task.TURN:
        path_length = 2 * s + (np.abs(6) * np.pi / 2)
        path_len = np.linspace(0, path_length, num_nodes)
        q1_path = []
        q2_path = []
        for path in path_len:
            if path <= s:
                q1 = path
                q2 = 0
            elif path <= s + (np.abs(r) * np.pi / 2):
                #
                angle = (path - s) / np.abs(r)
                q1 = s + r * np.sin(angle)
                q2 = np.sign(r) * r * (1 - np.cos(angle))
            else:
                # Second straight segment
                remaining_length = path - (
                            s + (np.abs(r) * np.pi / 2))
                q1 = s + np.abs(r)
                q2 = np.sign(r) * r + remaining_length
            q1_path.append(q1)
            q2_path.append(q2)

        q1_path = np.array(q1_path)
        q2_path = np.array(q2_path)
    elif Task == Task.PERTURBED:
        q1_path = np.linspace(0, (5 * s), num=num_nodes)
        q2_path = np.linspace(0, 0, num=num_nodes)

    if Task == Task.TURN:
        fig1, ax1 = plt.subplots(figsize=(6*cm, 5*cm))
    else:
        fig1, ax1 = plt.subplots(figsize=(6*cm, 3*cm))
    ax1.plot(q1_path, q2_path, color='k', label="Target Path")
    ax1.plot(x_data, y_data, color='r', linestyle='--', label="Bicycle Trajectory")
    ax1.set_xlabel("Longitudinal displacement [m]")
    ax1.set_ylabel("Lateral displacement [m]")
    if Task == Task.DOUBLE:
        ax1.set_ylim([-0.5, d_lat + 0.5])
    elif Task == Task.PERTURBED:
        ax1.set_ylim([-0.5, 1.5])
    elif Task == Task.TURN:
        ax1.set_ylim([-0.15, 0.25])
        ax1.set_xlim([0, 9])
        ax1.grid(color='gray', linestyle='--', linewidth=0.5)
    #ax1.legend()
    #ax1.legend(loc="upper right", bbox_to_anchor=(1.05, 1), fontsize=8.5)
    if save_plot:
        #plot_filename = os.path.join(PLOTS_DIR, f"plot of results_{start-1}_to_{end-1} over distance.png")
        plot_filename = os.path.join(PLOTS_DIR, f"Plot path of {Task}_with {result_dirr}.png")
        plt.savefig(plot_filename)
        print(f"Plot saved: {plot_filename}")
    plt.show()

def get_energy_data(output_dir, save_plot=None):
    def extract_number(filename):
        match = re.match(r"result(\d+)_", filename)
        return int(match.group(1)) if match else float('inf')  # Sort unknown files last
    result_dirs = sorted(os.listdir(output_dir), key=extract_number)
    energy_values = np.zeros((7, 6))  # _x_ matrix to store energy values
    torque_values = np.zeros((7, 6))  # _x_ matrix to store torque values
    selected_indices = [3, 9, 15]#, 19]#, 24]  # Indices of selected results to plot
    selected_results = [result_dirs[i] for i in selected_indices]

    #labels = ["No Spring", "Spring K = 10 Nm/rad", "Spring K=20 Nm/rad" ]
    labels = ["Single Pendulum", "Double Pendulum", "Triple Pendulum"]#, "Single Pendulum", "Double Pendulum", "Triple Pendulum"]  # , "Neck Lean"]  # Custom labels
    line_styles = ['solid', 'solid', 'solid']#, 'dashed', 'dashed', 'dashed']#, 'solid']# 'dashed', 'dashed', 'dashed']  # , '-']  # Custom line styles
    line_colors = ['g', 'r', 'b']#, 'g', 'r', 'b']
    #line_colors = ["red", "orange", "purple"]
    plt.figure(figsize=(12*cm, 8*cm))
    color_legend_handles = []  # Left legend (Pendulum Type - Based on color)
    style_legend_handles = []  # Right legend (Line Style - Based on linestyle)
    added_colors = set()  # Track added colors to avoid duplicate entries
    added_styles = set()  # Track added linestyles to avoid duplicate entries

    for idx, result_dir in enumerate(result_dirs, start=0):
        file_path = os.path.join(output_dir, result_dir, "solution_data.pkl")
        if not os.path.isfile(file_path):
            print("No .pkl")
            continue  # Skip if solution_data.pkl is missing

        with open(file_path, "rb") as file:
            dataa = cp.load(file)  # Load the cloudpickle file
        print(f"Processing {idx}: {result_dir} - Data length: {len(dataa)}")
        # These if/elif statements determine which data to select based on which model is used (thus also which length the state array is)
        if len(dataa) == 23:
            P = dataa[19] * dataa[13]
            W_array = np.abs(P) / 50
            W = np.cumsum(W_array)
            T_array = np.abs(dataa[19]) / 50
            T = np.cumsum(T_array)
            print('Energy:', W[-1], '  Torque:', T[-1])
        elif len(dataa) == 26:
            P1, P2 = dataa[21] * dataa[14], dataa[22] * dataa[15]
            W_array = (np.abs(P1) + np.abs(P2)) / 50
            W = np.cumsum(W_array)
            T_array = (np.abs(dataa[21]) + np.abs(dataa[22])) / 50
            T = np.cumsum(T_array)
            print('Energy:', W[-1], '  Torque:', T[-1])
        elif len(dataa) == 29:
            P1, P2, P3 = dataa[23] * dataa[17], dataa[24] * dataa[15], dataa[25] * dataa[16]
            W_array = (np.abs(P1) + np.abs(P2) + np.abs(P3)) / 50
            W = np.cumsum(W_array)
            T_array = (np.abs(dataa[23]) + np.abs(dataa[24]) + np.abs(dataa[25])) / 50
            T = np.cumsum(T_array)
            print('Energy:', W[-1], '  Torque:', T[-1])
        elif len(dataa) == 25:
            P = dataa[21] * dataa[14]
            W_array = np.abs(P) / 50
            W = np.cumsum(W_array)
            T_array = (np.abs(dataa[21])) / 50
            T = np.cumsum(T_array)
            print('Energy:', W[-1], '  Torque:', T[-1])
        elif len(dataa) == 28:
            P1, P2 = dataa[23] * dataa[15], dataa[24] * dataa[16]
            W_array = (np.abs(P1) + np.abs(P2)) / 50
            W = np.cumsum(W_array)
            T_array = (np.abs(dataa[23]) + np.abs(dataa[24])) / 50
            T = np.cumsum(T_array)
            print('Energy:', W[-1], '  Torque:', T[-1])
        elif len(dataa) == 31:
            P1, P2, P3 = dataa[25] * dataa[18], dataa[26] * dataa[16], dataa[27] * dataa[17]
            W_array = (np.abs(P1) + np.abs(P2) + np.abs(P3)) / 50
            W = np.cumsum(W_array)
            T_array = (np.abs(dataa[25]) + np.abs(dataa[26]) + np.abs(dataa[27])) / 50
            T = np.cumsum(T_array)
            print('Energy:', W[-1], '  Torque:', T[-1])
        else:
            print("doesn't match data length")
            continue  # Skip unknown cases

        row, col = divmod(idx, 6)
        energy_values[row, col] = W[-1]  # Store final energy value
        torque_values[row, col] = T[-1]  # Store final torque value
        time = dataa[0]  # Assuming time is in dataa[0]
        x = dataa[1]  # Assuming distance is in dataa[1]
        if result_dir in selected_results:
            plot_idx = selected_results.index(result_dir)  # Find position in selected results

            label = labels[plot_idx]
            linestyle = line_styles[plot_idx]
            color = line_colors[plot_idx]

            # Plot the energy data
            plt.plot(x, W, linestyle=linestyle, color=color, linewidth=1)

            # Add to left legend (color-based) if not already added
            if label not in added_colors:
                color_legend_handles.append(Line2D([0], [0], color=color, lw=1, label=label))
                added_colors.add(label)

            # Add to right legend (linestyle-based) if not already added
            if linestyle not in added_styles:
                style_label = "Leaning Seat" if linestyle == 'solid' else "Combination Seat"
                style_legend_handles.append(
                    Line2D([0], [0], color='black', linestyle=linestyle, lw=1, label=style_label))
                added_styles.add(linestyle)

            # Set the left and right column legends
        first_legend = plt.legend(handles=color_legend_handles, loc="upper left", bbox_to_anchor=(0.0, 1.035), fontsize=10)
        #second_legend = plt.legend(handles=style_legend_handles, loc="upper left", bbox_to_anchor=(0.4, 0.975), fontsize=10)
        # Add legends to the plot
        plt.gca().add_artist(first_legend)
        #plt.gca().add_artist(second_legend)


    plt.xlabel('Distance [m]')  # or 'Distance (m)' if using distance
    #plt.ylabel('Angle [Deg]')
    plt.ylabel('Energy [J]')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()


    print("Energy Matrix:")
    for i, row in enumerate(energy_values, start=1):
        print(f"{i}: " + " ".join(f"{val:.2f}" if not np.isnan(val) else "--" for val in row))
    print()
    print("Torque Matrix:")
    for i, row in enumerate(torque_values, start=1):
        print(f"{i}: " + " ".join(f"{val:.2f}" if not np.isnan(val) else "--" for val in row))
    if save_plot:
        #plot_filename = os.path.join(PLOTS_DIR, f"plot of results_{start-1}_to_{end-1} over distance.png")
        plot_filename = os.path.join(PLOTS_DIR, f"E plot tres penduli of results laneswitch at 14kmh.png")
        plt.savefig(plot_filename)
        print(f"Plot saved: {plot_filename}")

    plt.show()

def plot_mocap_data(y_path, z_path, target_markers, ref_marker):
    """
    Plots Y and Z trajectories of specified markers relative to a reference marker.

    Parameters:
    - y_path: str, path to the Y coordinate CSV file
    - z_path: str, path to the Z coordinate CSV file
    - target_markers: list of int, marker indices to plot
    - ref_marker: int, marker index to use as reference
    - figsize: tuple, size of the matplotlib figure
    """
    # Load data
    y_data = pd.read_csv(y_path, header=None, delimiter=';').values
    z_data = pd.read_csv(z_path, header=None, delimiter=';').values


    frame_range = [80, 2000]#[80, 400]
    # Handle NaNs: get valid range if not manually specified
    if frame_range is None:
        valid_frames = ~np.isnan(y_data[:, target_markers + [ref_marker]]).any(axis=1)
        start_idx = np.argmax(valid_frames)  # first True
        end_idx = len(valid_frames) if not False in valid_frames[start_idx:] else start_idx + np.argmax(
            ~valid_frames[start_idx:])
    else:
        start_idx, end_idx = frame_range

    # Slice data
    y_slice = y_data[start_idx:end_idx]
    z_slice = z_data[start_idx:end_idx]

    # Compute relative positions
    relative_y = y_slice[:, target_markers] - y_slice[:, ref_marker][:, np.newaxis]
    relative_z = z_slice[:, target_markers] - z_slice[:, ref_marker][:, np.newaxis]

    if np.any(np.isnan(relative_y)) or np.any(np.isnan(relative_z)):
        print("Warning: NaN values found in the data!")
    # Plot
    cm = 1/2.54 # inch to cm

    colors = ['darkviolet', 'orange', 'limegreen']
    names = ['Left Hip', 'Buttocks', 'Right Hip']
    colors2 = ['black', 'darkgrey', 'dimgrey']#['blue', 'red', 'green']

    plt.figure(figsize = (9.2 * cm, 7 * cm))
    for i, marker_idx in enumerate(target_markers):
        plt.plot(-relative_y[:, i], relative_z[:, i],
                 color=colors[i % len(colors)], label=names[i], marker='o', linestyle='-', markersize=3)

    plt.xlabel('Distance [m]')
    plt.ylabel('Distance [m]')
    #plt.title('Marker Trajectories (Y vs Z)')
    plt.ylim(0.0, 0.4)
    plt.xlim(-0.3, 0.3)
    plt.grid(True)
    custom_lines = [Line2D([0], [0], color=colors[i % len(colors)], lw=6) for i in range(len(target_markers))]
    plt.legend(custom_lines, names)
    plt.tight_layout()

    timestamps = [15, 80, 145] # 20, 80, 110
    plt.figure(figsize=(8.8 * cm, 7 * cm))
    for j, timestamp in enumerate(timestamps):
        for i, marker_idx in enumerate(target_markers):
            y_values = relative_y[timestamp, i]
            z_values = relative_z[timestamp, i]

            # Plot individual points
            plt.scatter(-y_values, z_values, color=colors2[j], label=f"Timestamp {timestamp}" if i == 0 else "",
                        zorder=5)

            if i < len(target_markers) - 1:
                # Plot lines connecting points from 1st -> 2nd, and 2nd -> 3rd markers
                y_values_next = relative_y[timestamp, i + 1]
                z_values_next = relative_z[timestamp, i + 1]
                plt.plot([-y_values, -y_values_next], [z_values, z_values_next], color=colors2[j], linestyle='-',
                         alpha=0.9)

    plt.xlabel('Distance [m]')
    #plt.ylabel('Distance [m]')
    #plt.title('Selected Timestamps - Connected Points')
    plt.ylim(0.0, 0.4)
    plt.xlim(-0.3, 0.3)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def bicycle_parameters_plot():
    bicycle = bp.Bicycle('Browser', pathToData='C:\THESIS\src\Optimization\data', forceRawCalc=True)

    # bicycle.plot_bicycle_geometry(show=True)
    speeds = np.linspace(0, 10, 100)
    # bicycle.plot_eigenvalues_vs_speed(speeds, show=True, show_legend=True)
    print(Meijaard2007ParameterSet.par_strings)

    bicycle.add_rider('Jason', reCalc=True)

    print(bicycle.parameters)
    print(Meijaard2007ParameterSet.par_strings.keys())
    cm = 1 / 2.54

    bicycle.plot_bicycle_geometry(show=True)
    bicycle.plot_eigenvalues_vs_speed(speeds, show=True, show_legend=False)
    plt.axvspan(5.15, 7.7, color='grey', alpha=0.3)  # alpha controls transparency
    plt.ylim(-15, 7.5)
    plt.xlabel("Velocity [m/s]")
    plt.grid()
    benchmark_params = {
        key: float(nominal_value(val))
        for key, val in bicycle.parameters['Benchmark'].items()
        if key in Meijaard2007ParameterSet.par_strings  # only include expected params
    }
    benchmark_params['v'] = 0.0

    par_set = Meijaard2007ParameterSet(parameters=benchmark_params, includes_rider=True)
    model = Meijaard2007Model(par_set)

    # model = Meijaard2007Model
    v = np.linspace(0, 10, 100)

    # Plot using the provided parameters and axes
    model.plot_eigenvalue_parts(v=v,
                                colors=['C0', 'C0', 'C2', 'C3'],  # Provide 4 colors for the 4 modes
                                show_stable_regions=True,  # Optional, enable stable regions shading
                                hide_zeros=False,  # Optional, can hide zeros if needed
                                show_legend=True
                                )
    plt.ylim(-10, 7.5)
    plt.xlabel("Velocity [m/s]")
    plt.show()




""" The plot functions are called in the lines below. """
#plot_selected_data(start=startt, end=endd, row_indices=None, per_result_rows=per_result_rows_example) # row_indices=[3,2]

#plot_single_sim(save_plot=True, x_label="Time [s]", y_label="Angle [Deg]", x_limits=(0, 2), y_limits=None) #

#plot_single_array(save_plot=True)

#plot_multiple_arrays(save_plot=True)

#get_path(data=DataStorage, save_plot=True)

get_energy_data(output_dir=OUTPUT_DIR, save_plot=True)

#plot_mocap_data(y_path='data/3101Y.csv', z_path='data/3101Z.csv', target_markers=[7, 8, 3], ref_marker=25)

#bicycle_parameters_plot()