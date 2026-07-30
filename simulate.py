from pydy.system import System as PyDySystem
import bicycleparameters as bp
import matplotlib.pyplot as plt
import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from bicycleparameters.parameter_dicts import meijaard2007_browser_jason
from bicycleparameters.models import Meijaard2007WithFeedbackModel
from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
from scipy.linalg import solve_continuous_are

# TODO : This needs to be loaded from a user configuration.
BPDATADIR = "/home/moorepants/Data/bicycle-parameters"


def evaluate_torques_lqr(trajectories):
    """Returns the torque trajectories computed by the LQR controller.

    Parameters
    ==========
    trajectories : array_like, shape(n, 16)
        State trajectories.
        [q1, q2, q3, q4, q6, q7, q8, q5, u4, u6, u7, u1, u2, u3, u5, u8]

    Returns
    =======
    u : ndarray, shape(n, 3)
        Torque trajectories [T4, T6, T7]

    """
    u = np.zeros((trajectories.shape[0], 3))
    for i, xi in enumerate(trajectories):
        u[i, :] = compute_torques_lqr(xi, 0.0)
    return u


def plot_trajectories(sys, trajectories):
    """Returns axes to three figures: coordinates, speeds, and torques."""

    n = len(sys.states)
    fig, axesq = plt.subplots(n//2, 1, sharex=True, layout='constrained')
    fig.set_size_inches(16, n//2)
    for ax, traj, s in zip(axesq, trajectories.T[:n//2], sys.states[:n//2]):
        ax.plot(sys.times, traj)
        ax.set_ylabel(sm.latex(s, mode='inline'))
    axesq[-1].set_xlabel('Time [s]')

    fig, axesu = plt.subplots(n//2, 1, sharex=True, layout='constrained')
    fig.set_size_inches(16, n//2)
    for ax, traj, s in zip(axesu, trajectories.T[n//2:], sys.states[n//2:]):
        ax.plot(sys.times, traj)
        ax.set_ylabel(sm.latex(s, mode='inline'))
    axesu[-1].set_xlabel('Time [s]')

    u = evaluate_torques_lqr(trajectories)

    fig, axes = plt.subplots(3, 1, sharex=True, layout='constrained')
    fig.set_size_inches(16, 3)
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
    T : ndarray, shape(3,)
        Torques: [T4, T6, T7]

    """
    q3, u1, u2 = x[2], x[11], x[12]
    b1 = np.array([np.cos(q3), np.sin(q3)])
    v = np.array([-u1, u2])
    speed = v.dot(b1)
    x_min = x[[3, 5, 8, 10]]  # q4, q7, u4, u7
    K = compute_controller_gains(meijaard2007_browser_jason, speed)
    T4, T7 = -K @ x_min.T
    T6 = -3.0
    return np.array([T4, T6, T7])


def create_pydy_system(bicycle, system):
    """Returns a PyDy System set up for simulation."""

    bike_params = bp.Bicycle("Browser", pathToData=BPDATADIR,
                             forceRawCalc=True)
    # TODO: This is failing:
    #bike_params.add_rider("Jason", reCalc=True)
    constants_def = bicycle.get_param_values(bike_params)
    # TODO : g should be retrieved from the bicycle model
    g = sm.symbols('g')
    constants_def[g] = 9.81

    pydy_sys = PyDySystem(system.eom_method)

    for k, v in constants_def.items():
        if k in pydy_sys.constants_symbols:
            pydy_sys.constants[k] = v

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
    pydy_sys.specifieds = {(T4, T6, T7): compute_torques_lqr}

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
    torque_traj = evaluate_torques_lqr(traj)
    plot_trajectories(pydy_sys, traj)
    plt.show()
