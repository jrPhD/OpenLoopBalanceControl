from pydy.system import System as PyDySystem
import bicycleparameters as bp
import matplotlib.pyplot as plt
import numpy as np
import sympy as sm
import sympy.physics.mechanics as me

BPDATADIR = "/home/moorepants/Data/bicycle-parameters"


def plot_trajetories(sys, trajectories):
    fig, axes = plt.subplots(len(sys.states), 1, sharex=True,
                            layout='constrained')
    fig.set_size_inches(8, 16)
    for ax, traj, s in zip(axes, trajectories.T, sys.states):
        ax.plot(sys.times, traj)
        ax.set_ylabel(sm.latex(s, mode='inline'))
    axes[-1].set_xlabel('Time [s]')
    return axes


def control(x, t):
    T4 = 0.0
    T6 = -10.0
    u4 = x[8]
    T7 = 5.0*u4
    return np.array([T4, T6, T7])


def simulate_system(bicycle, system):

    bike_params = bp.Bicycle("Browser", pathToData=BPDATADIR,
                             forceRawCalc=True)
    # TODO: This is failing:
    #bike_params.add_rider("Jason", reCalc=True)
    constants_def = bicycle.get_param_values(bike_params)
    # TODO : g should be retrieved from the bicycle model
    g = sm.symbols('g')
    constants_def[g] = 9.81

    q1, q2, q3, q4, q6, q7, q8, q5 = system.q
    u4, u6, u7, u1, u2, u3, u5, u8 = system.u
    # get_all_symbols() seems to only return geometry
    (d2,
     d3,
     front_frame_d6,
     front_frame_d7,
     front_frame_d8,
     front_frame_l3,
     front_frame_l4,
     rf,
     d1,
     rear_frame_d4,
     rear_frame_d5,
     rear_frame_l1,
     rear_frame_l2,
     rear_frame_l_bbx,
     rear_frame_l_bbz,
     rr) = list(sm.ordered(list(bicycle.get_all_symbols())))

    pydy_sys = PyDySystem(system.eom_method)

    for k, v in constants_def.items():
        if k in pydy_sys.constants_symbols:
            pydy_sys.constants[k] = v

    initial_speed = 2.6  # m/s
    initial_roll_rate = 0.5  # rad/s

    # TODO : Why is q8 in the holonomic constraint?
    pydy_sys.initial_conditions = {
        q1: 0.0,
        q2: 0.0,
        q3: 0.0,
        q4: 0.0,
        q5: np.deg2rad(20.0),  # guess
        q7: 0.0,
        u1: initial_speed,
        u2: 0.0,
        u3: 0.0,
        u4: initial_roll_rate,
        u5: 0.0,
        u6: -initial_speed/pydy_sys.constants[rr],  # guess
        u7: 0.0,
        u8: -initial_speed/pydy_sys.constants[rf],  # guess
    }

    pydy_sys.set_dependent_initial_conditions(
        dep_vars=(q5, u2, u3, u5, u6, u8))

    T4, T6, T7 = me.dynamicsymbols('T4, T6, T7')
    pydy_sys.specifieds = {(T4, T6, T7): control}

    fps = 30  # frames per second
    duration = 10.0  # seconds
    pydy_sys.times = np.linspace(0.0, duration, num=int(duration*fps))

    return pydy_sys
