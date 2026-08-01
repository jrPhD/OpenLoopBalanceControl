import configparser
import os

from bicycleparameters.models import Meijaard2007WithFeedbackModel
from bicycleparameters.parameter_dicts import meijaard2007_browser_jason
from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
from pydy.system import System as PyDySystem
from scipy.linalg import solve_continuous_are
from symbrim.utilities.plotting import Plotter
import bicycleparameters as bp
import matplotlib.pyplot as plt
import numpy as np
import sympy as sm
import sympy.physics.mechanics as me

if os.path.exists('conf.ini'):
    config = configparser.ConfigParser()
    config.read('conf.ini')
    BPDATADIR = config['config']['BPDATADIR']
else:
    msg = """
Create a conf.ini file with contents:
[config]
BPDATADIR = /path/to/bicycleparameters/data
"""
    raise RuntimeError(msg)


def animate_motion(bicycle, pydy_sys, x, r):
    """Returns a matplotlib animation.

    Parameters
    ==========
    bicycle : symbrim.WhippleBicycle
        Intialized bicycle model.
    pydy_sys : pydy.system.System
        Initialized PyDy system.
    x : array_like, shape(n, 16)
        State trajectories.
    r : array_like, shape(n, 6)
        Input trajectories.

    """

    p = list(pydy_sys.constants.keys())
    p_vals = np.array(list(pydy_sys.constants.values()))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 8))

    plotter = Plotter.from_model(bicycle, ax=ax)
    plotter.lambdify_system((pydy_sys.states,
                             pydy_sys.specifieds_symbols,
                             p))
    plotter.evaluate_system(x[0, :], r[0, :], p_vals)
    plotter.plot()
    q1, q2 = x[:, 0], x[:, 1]
    X, Y = np.meshgrid(np.arange(np.min(q1) - 1.0, np.max(q1) + 1.0, 0.5),
                       np.arange(np.min(q2) - 1.0, np.max(q2) + 1.0, 0.5))
    ax.plot_wireframe(X, Y, np.zeros_like(X), color="k", alpha=0.3, rstride=1,
                      cstride=1)
    ax.invert_zaxis()
    ax.invert_yaxis()
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.view_init(19, 14)
    ax.set_aspect("equal")
    ax.axis("off")

    ani = plotter.animate(lambda i: (x[i], r[i], p_vals),
                          frames=range(x.shape[0]),
                          blit=False)
    return ani


def eval_input_trajectories(time, trajectories):
    """Returns the torque trajectories computed by the LQR controller.

    Parameters
    ==========
    trajectories : array_like, shape(n, 16)
        State trajectories.
        [q1, q2, q3, q4, q6, q7, q8, q5, u4, u6, u7, u1, u2, u3, u5, u8]

    Returns
    =======
    u : ndarray, shape(n, 3)
        Torque trajectories [Fx, Fy, Fz, T4, T6, T7]

    """
    u = np.zeros((trajectories.shape[0], 6))
    for i, xi in enumerate(trajectories):
        u[i, :] = compute_inputs(xi, time[i])
    return u


def plot_trajectories(sys, trajectories):
    """Returns axes to four figures: coordinates, speeds, inputs, and rear
    contact trajectory.

    Parameters
    ==========
    sys : pydy.system.System
        Initialized PyDy System.
    trajectories : array_like, shape(n, 16)

    """

    q_units = ['m', 'm', 'rad', 'rad', 'rad', 'rad', 'rad', 'rad']
    u_units = ['rad/s', 'rad/s', 'rad/s', 'm/s', 'm/s', 'rad/s', 'rad/s',
               'rad/s']

    n = len(sys.states)
    fig, axesq = plt.subplots(n//2, 1, sharex=True, layout='constrained')
    fig.set_size_inches(16, n//2)
    for ax, traj, s, unit in zip(axesq, trajectories.T[:n//2],
                                 sys.states[:n//2], q_units):
        if unit == 'rad':
            ax.plot(sys.times, np.rad2deg(traj))
        else:
            ax.plot(sys.times, traj)
        ax.set_ylabel(sm.latex(s, mode='inline'))
    axesq[-1].set_xlabel('Time [s]')

    fig, axesu = plt.subplots(n//2, 1, sharex=True, layout='constrained')
    fig.set_size_inches(16, n//2)
    for ax, traj, s, unit in zip(axesu, trajectories.T[n//2:],
                                 sys.states[n//2:], u_units):
        if unit == 'rad/s':
            ax.plot(sys.times, np.rad2deg(traj))
        else:
            ax.plot(sys.times, traj)
        ax.set_ylabel(sm.latex(s, mode='inline'))
    axesu[-1].set_xlabel('Time [s]')

    u = eval_input_trajectories(sys.times, trajectories)

    fig, axes = plt.subplots(6, 1, sharex=True, layout='constrained')
    fig.set_size_inches(16, 6)
    for ax, traj, s in zip(axes, u.T, list(sys.specifieds.keys())[0]):
        ax.plot(sys.times, traj)
        ax.set_ylabel(sm.latex(s, mode='inline'))
    axes[-1].set_xlabel('Time [s]')

    fig, ax_ground = plt.subplots(1, 1, layout='constrained')
    fig.set_size_inches(8, 8)
    ax_ground.plot(trajectories[:, 0], trajectories[:, 1])
    ax_ground.set_aspect('equal')

    return axesq, axesu, axes, ax_ground


def compute_controller_gains(par, speed, roll=True, steer=True):
    """Returns the LQR gain matrix K for feedback of roll angle, steer angle,
    roll angular rate, and steer angular rate to control roll torque and steer
    torque.

    Parameters
    ==========
    par : dictionary
        Parameter dictionary mapping strings to floats for the Meijaard 2007
        parameter set.
    speed : float
        Speed at which the model is linearized about.
    steer : boolean
        If true gains for the steer torque command are returned.
    roll : boolean
        If true gains for the roll torque commaned are returned.

    Returns
    =======
    K : ndarray, shape(n, 4)
        n = 2 if roll and steer are both true, otherwise n = 1. States are in
        order q4, q7 u4, u7.

    """
    # TODO : Handle having a rider inertia or not (if that matters).
    par_set = Meijaard2007ParameterSet(par, True)
    model = Meijaard2007WithFeedbackModel(par_set)
    A, B = model.form_state_space_matrices(v=speed)
    T4_sq_max, T7_sq_max = 40.0, 8.0
    if steer and roll:
        start, end = 0, 2
        R = np.diag([1.0/T4_sq_max**2, 1.0/T7_sq_max**2])
    elif steer and not roll:
        start, end = 1, 2
        R = 1.0/T7_sq_max**2*np.eye(1)
    elif not steer and roll:
        start, end = 0, 1
        R = 1.0/T4_sq_max**2*np.eye(1)
    else:
        raise ValueError('One of steer or roll must be true')
    #Q = np.eye(4)
    Q = np.deg2rad(np.diag([10.0, 30.0, 10.0, 20.0]))**2
    S = solve_continuous_are(A, B[:, start:end], Q, R)
    K = np.linalg.solve(R, B[:, start:end].T @ S)
    return K


def compute_torques_lqr(x, t):
    """Returns the roll, rear wheel, and steer torques based on speed dependent
    LQR solution for roll and steer with constant torque for the rear wheel.

    Parameters
    ==========
    x : ndarray, shape(16,)
        State vector:
        [q1, q2, q3, q4, q6, q7, q8, q5, u4, u6, u7, u1, u2, u3, u5, u8]
    t : float
        Time

    Returns
    =======
    ndarray, shape(2,)
        Torques: [T4, T7]

    """
    q3, u1, u2 = x[2], x[11], x[12]
    b1 = np.array([np.cos(q3), np.sin(q3)])
    v = np.array([-u1, u2])
    speed = v.dot(b1)
    x_min = x[[3, 5, 8, 10]]  # q4, q7, u4, u7
    K = compute_controller_gains(meijaard2007_browser_jason, speed)
    T4, T7 = -K @ x_min.T
    return np.array([T4, T7])


def compute_inputs(x, t):
    """
    Parameters
    ==========
    x : ndarray, shape(16,)
        State vector:
        [q1, q2, q3, q4, q6, q7, q8, q5, u4, u6, u7, u1, u2, u3, u5, u8]
    t : float
        Time

    Returns
    =======
    ndarray, shape(6,)
        Torques: [Fx, Fy, Fz, T4, T6, T7]

    """
    T4, T7 = compute_torques_lqr(x, t)
    T6 = -3.0
    Fx = 0.0
    if t > 1.0 and t < 2.0:
        Fy = 50.0
    else:
        Fy = 0.0
    Fz = 0.0
    return np.array([Fx, Fy, Fz, T4, T6, T7])


def create_pydy_system(bicycle, system):
    """Returns a PyDy System set up for simulation.

    Parameters
    ==========
    bicycle : symbrim.WhippleBicycle
        Intialized bicycle model.
    system : sympy.physics.mechanics.system.System
        Initialized SymPy system.

    Returns
    =======
    pydy_sys : pydy.system.System
        Initialized PyDy system.

    """

    bike_params = bp.Bicycle("Browser", pathToData=BPDATADIR,
                             forceRawCalc=True)
    bike_params.add_rider("Jason", reCalc=True)
    constants_def = bicycle.get_param_values(bike_params)
    # TODO : g should be retrieved from the bicycle model
    g = sm.symbols('g')
    constants_def[g] = 9.81

    # NOTE : Defines the constants symbols explicitly to allow constants not
    # present in the equations of motion to be stored on the system.
    pydy_sys = PyDySystem(system.eom_method,
                          constants=constants_def,
                          constants_symbols=list(constants_def.keys()))

    q1, q2, q3, q4, q6, q7, q8, q5 = system.q
    u4, u6, u7, u1, u2, u3, u5, u8 = system.u

    initial_speed = 3.0  # m/s
    initial_roll_rate = 0.5  # rad/s

    pydy_sys.initial_conditions = {
        q1: 0.0,
        q2: 0.0,
        q3: 0.0,
        q4: 0.0,
        q5: np.deg2rad(20.0),  # guess
        q6: 0.0,
        q7: 0.0,
        q8: 0.0,
        u1: initial_speed,
        u2: 0.0,
        u3: 0.0,
        u4: initial_roll_rate,
        u5: 0.0,
        u6: -initial_speed/0.3,  # guess
        u7: 0.0,
        u8: -initial_speed/0.3,  # guess
    }

    pydy_sys.set_dependent_initial_conditions(
        dep_vars=(q5, u2, u3, u5, u6, u8))

    T4, T6, T7 = me.dynamicsymbols('T4, T6, T7')
    Fx, Fy, Fz, T4, T6, T7 = sm.ordered(list(pydy_sys.specifieds_symbols))
    pydy_sys.specifieds = {
        (Fx, Fy, Fz, T4, T6, T7): compute_inputs,
    }

    fps = 60  # frames per second
    duration = 6.0  # seconds
    pydy_sys.times = np.linspace(0.0, duration, num=int(duration*fps))

    return pydy_sys


if __name__ == "__main__":

    from bicycle_models import generate_bicycle_rider_model
    bicycle, sys = generate_bicycle_rider_model()
    pydy_sys = create_pydy_system(bicycle, sys)
    pydy_sys.initial_conditions
    pydy_sys.constants
    K = compute_controller_gains(meijaard2007_browser_jason, 1.0)
    traj = pydy_sys.integrate()
    input_traj = eval_input_trajectories(pydy_sys.times, traj)
    plot_trajectories(pydy_sys, traj)
    ani = animate_motion(bicycle, pydy_sys, traj, input_traj)
    plt.show()
