# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 15:13:09 2025

@author: jrronne
"""

from __future__ import annotations

import json
import os

import cloudpickle as cp
import matplotlib.pyplot as plt
import numpy as np
from src.Optimization.opt_container import DataStorage, Metadata, SteerWith, SeatType, TorsoType, Task, Model, InitGuess
from src.Optimization.opt_model import set_simulator, set_model
from src.Optimization.opt_problem import set_problem, set_constraints, set_initial_guess
from src.Optimization.opt_utils import NumpyEncoder, Timer, create_time_lapse, create_animation, create_plots, plot_model_figures, create_animation2, bike_following_animation, bike_following_animation2, bike_following_timelapse
from opty import Problem
from opty.utils import parse_free
import sympy.physics.mechanics as me
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import ast
import pandas as pd
import sympy as sm
from sympy import Symbol


# from CYINTIA import cyclist, set_axes_equal #My cyclist inertia toolbox

def sol_dict(prob, free):
    """Returns a dictionary that has SymPy dynamic symbols mapped to
    the solution.

    Parameters
    ==========
    prob : Problem
        Instance of an opty Problem.
    free : array_like, shape(n*N + q*N,)
        The free optimization variables.

    Returns
    =======
    d : dictionary
        Maps dynamicsymbols to array of shape(N,)

    """
    x, r, _ = prob.parse_free(free)
    d = {}
    for symbol, array in zip(prob.collocator.state_symbols, x):
        d[symbol] = array
    for symbol, array in zip(prob.collocator.unknown_input_trajectories, r):
        d[symbol] = array
    return d


def RMSE(x, x_meas):
    """
    Computes RMSE to assess optimization performance

    Parameters
    ----------
    x : numpy array
        ground truth.
    x_meas : numpy array
        signal.

    Returns
    -------
    RMSE

    """

    res = np.mean((x_meas - x)**2)
    res = np.sqrt(res)

    return(res)




class OLDC_Solver():

    #Open Loop Direct Collocation Solver

    def __init__(self, exp_file_path, n_part):

        print(exp_file_path)

        self.df_exp = pd.read_csv(exp_file_path)
        self.list_hands_off_trials = self.df_exp[self.df_exp['condition'] == 'straight/hands_off']['trial'].unique()
        self.n_part = n_part
        self.date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')

        print("Trials numbers for straight/hands_off condition", self.list_hands_off_trials)


    def select_trial(self, list_trials = []):

        if list_trials == []:
            self.list_trials = self.list_hands_off_trials

        else:
             self.list_trials = list_trials

    def initialize_problem(self, n_trial):


        df_trial = self.df_exp[self.df_exp['trial'] == n_trial]

        self.n_trial = n_trial

        #Filtering of exp data

        psi = df_trial['psi'].to_numpy()
        psi = psi - np.mean(psi)
        psi = - np.deg2rad(psi)

        phi = df_trial['phi'].to_numpy()
        phi = phi - np.mean(phi)
        phi = - np.deg2rad(phi)

        delta = df_trial['delta'].to_numpy()
        delta = delta - np.mean(delta)
        delta = - np.deg2rad(delta)

        psi_dot = df_trial['psi_dot'].to_numpy()
        psi_dot = - np.deg2rad(psi_dot)

        phi_dot = df_trial['phi_dot'].to_numpy()
        phi_dot = - np.deg2rad(phi_dot)

        delta_dot = df_trial['delta_dot'].to_numpy()
        delta_dot = - np.deg2rad(delta_dot)


        Gyr_x_H0, Gyr_y_H0, Gyr_z_H0 = df_trial[['Gyr_x_H','Gyr_y_H','Gyr_z_H']].to_numpy().T
        # Gyr_z_H = np.deg2rad(Gyr_z_H) - phi_dot
        Gyr_x_H =   np.deg2rad(Gyr_z_H0.copy())
        Gyr_y_H = - np.deg2rad(Gyr_x_H0.copy())
        Gyr_z_H = - np.deg2rad(Gyr_y_H0.copy())

        #Gyr_z_H is used as lean rate that's why it's corrected by phi_dot




        #Accelerations measured in the local head frame (over the nose)
        # X pointing left
        # Y pointing up
        # Z pointing forward
        Acc_x_H0, Acc_y_H0, Acc_z_H0 = df_trial[['Acc_x_H', 'Acc_y_H', 'Acc_z_H']].to_numpy().T

        Acc_x_H =   Acc_z_H0.copy()
        Acc_y_H = - Acc_x_H0.copy()
        Acc_z_H = - Acc_y_H0.copy()

        u = df_trial['u'].to_numpy()
        time = df_trial['time'].to_numpy()

        # fig, axs = plt.subplots(2,1)

        # axs[0].plot(time, Acc_x_H)
        # axs[0].plot(time, Acc_y_H)
        # axs[0].plot(time, Acc_z_H)

        # axs[1].plot(time, Gyr_x_H)
        # axs[1].plot(time, Gyr_y_H)
        # axs[1].plot(time, Gyr_z_H)






        NUM_NODES = len(time)

        self.NUM_NODES = NUM_NODES

        DURATION = time[-1] - time[0]

        self.DURATION = DURATION

        interval = 0.01  # seconds


        x_meas_dict  = {'yaw_angle_q3' :     psi,
                       'roll_angle_q4' :     phi,
                       'steer_angle_q7' :    delta,

                       'roll_rate_u4' :      phi_dot,
                       'yaw_rate_u3' :       psi_dot,
                       'steer_rate_u7' :     delta_dot,

                       'Acc_x_H' : Acc_x_H,
                       'Acc_y_H' : Acc_y_H,
                       'Acc_z_H' : Acc_z_H,

                       'Gyr_x_H' : Gyr_x_H,
                       'Gyr_y_H' : Gyr_y_H,
                       'Gyr_z_H' : Gyr_z_H,

                       'speed' : u} #Include here the signals used to feed the opti

        self.x_meas_dict = x_meas_dict


        SRC_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(SRC_DIR, "data")



        METADATA = Metadata(
            bicycle_only=False,
            model_upper_body=True,
            model_legs=False,
            model_torso=True,
            model_head=True,
            sprung_steering=False,
            model=Model.DOUBLE_PENDULUM,
            task=Task.MATCH_EXP_DATA,
            pedaling_torque = True,
            extra_roll_torque = False,
            steer_with=SteerWith.SEAT_AND_TORSO_TORQUE,
            parameter_data_dir=DATA_DIR,
            seat_type=SeatType.SIDELEAN,
            torso_type=TorsoType.PIN,
            init_guess=InitGuess.RANDOM,
            bicycle_parametrization="Balanceassistv1",
            rider_parametrization="Jason",
            part_personalized_constants = {},
            duration=DURATION,
            lateral_displacement=0,
            straight_length=0,
            turn_radius=0,
            num_nodes=NUM_NODES,
            weight=0,
            weight_tr=0,
            weight_ct=0,
            x_meas_vec = self.x_meas_dict
        )

        self.data = DataStorage(METADATA)

        t = me.dynamicsymbols._t
        set_model(self.data)

        eval_ang_vel_mat_head_cst = ['seat_roll','seat_pitch','seat_yaw','torsojoint_theta']
        eval_ang_vel_head_cst_num = [self.data.constants.get(Symbol(cst)) for cst in eval_ang_vel_mat_head_cst]


        # eval_ang_vel_mat_head = sm.lambdify((self.data.x, eval_ang_vel_mat_head_cst), ang_vel_mat_head[:])
        # eval_acc_mat_head = sm.lambdify((self.data.x, eval_ang_vel_mat_head_cst), acc_mat_head[:])
        # print('----------------')
        # print(self.data.eoms[-1])
        # print(self.data.x)
        # print('----------------')


        # print(eval_ang_vel_mat_head(np.random.random(18), eval_ang_vel_head_cst_num))
        # print(eval_acc_mat_head(np.random.random(18), eval_ang_vel_head_cst_num))



        # x_meas_vec = np.array([x_meas_dict[k] for k in x_meas_dict.keys()]).flatten()


        #Weigths set to 1 until now
        K_angles = 1
        K_angle_rates = 1
        K_head_acc = 0
        K_head_gyr_X = 10
        K_speed = 1000
        K_torque = 1

        def obj(prob, free):
            """Minimize the error in all of the states."""
            d = sol_dict(prob, free)
            print(d[prob.collocator.state_symbols[0]])

            # C_yaw = (x_meas_dict['yaw_angle_q3'] - free[2*NUM_NODES:3*NUM_NODES])**2
            # C_roll = (x_meas_dict['roll_angle_q4'] - free[3*NUM_NODES:4*NUM_NODES])**2
            # C_steer = (x_meas_dict['steer_angle_q7'] - free[5*NUM_NODES:6*NUM_NODES])**2

            # C_roll_rate = (x_meas_dict['roll_rate_u4'] - free[9*NUM_NODES:10*NUM_NODES])**2
            # C_yaw_rate = (x_meas_dict['yaw_rate_u3'] - free[15*NUM_NODES:16*NUM_NODES])**2
            # C_steer_rate = (x_meas_dict['steer_rate_u7'] - free[11*NUM_NODES:12*NUM_NODES])**2

            # C_speed = (x_meas_dict['speed'] - free[13*NUM_NODES:14*NUM_NODES])**2


            # C_Acc_x_H = (x_meas_dict['Acc_x_H'] - free[18*NUM_NODES:19*NUM_NODES])**2
            # C_Acc_y_H = (x_meas_dict['Acc_y_H'] - free[19*NUM_NODES:20*NUM_NODES])**2
            # C_Acc_z_H = (x_meas_dict['Acc_z_H'] - free[20*NUM_NODES:21*NUM_NODES])**2

            # C_Gyr_x_H = (x_meas_dict['Gyr_x_H'] - free[21*NUM_NODES:22*NUM_NODES])**2
            # C_Gyr_y_H = (x_meas_dict['Gyr_y_H'] - free[22*NUM_NODES:23*NUM_NODES])**2
            # C_Gyr_z_H = (x_meas_dict['Gyr_z_H'] - free[23*NUM_NODES:24*NUM_NODES])**2

            C_yaw = (x_meas_dict['yaw_angle_q3'] - free[2*NUM_NODES:3*NUM_NODES])**2
            C_roll = (x_meas_dict['roll_angle_q4'] - free[3*NUM_NODES:4*NUM_NODES])**2
            C_steer = (x_meas_dict['steer_angle_q7'] - free[5*NUM_NODES:6*NUM_NODES])**2

            C_roll_rate = (x_meas_dict['roll_rate_u4'] - free[10*NUM_NODES:11*NUM_NODES])**2
            C_yaw_rate = (x_meas_dict['yaw_rate_u3'] - free[17*NUM_NODES:18*NUM_NODES])**2
            C_steer_rate = (x_meas_dict['steer_rate_u7'] - free[12*NUM_NODES:13*NUM_NODES])**2

            C_speed = (x_meas_dict['speed'] - free[15*NUM_NODES:16*NUM_NODES])**2


            C_Acc_x_H = (x_meas_dict['Acc_x_H'] - free[20*NUM_NODES:21*NUM_NODES])**2
            C_Acc_y_H = (x_meas_dict['Acc_y_H'] - free[21*NUM_NODES:22*NUM_NODES])**2
            C_Acc_z_H = (x_meas_dict['Acc_z_H'] - free[22*NUM_NODES:23*NUM_NODES])**2

            C_Gyr_x_H = (x_meas_dict['Gyr_x_H'] - free[23*NUM_NODES:24*NUM_NODES])**2
            C_Gyr_y_H = (x_meas_dict['Gyr_y_H'] - free[24*NUM_NODES:25*NUM_NODES])**2
            C_Gyr_z_H = (x_meas_dict['Gyr_z_H'] - free[25*NUM_NODES:26*NUM_NODES])**2

            #include torque to objective function
            C_torso_torque = free[26*NUM_NODES:27*NUM_NODES]**2
            C_pelvis_torque = free[27*NUM_NODES:28*NUM_NODES]**2
            C_pedaling_torque = free[28*NUM_NODES:29*NUM_NODES]**2

            J = (K_angles*np.sum(C_yaw + C_roll + C_steer)
            + K_angle_rates*np.sum(C_roll_rate + C_yaw_rate + C_steer_rate)
            + K_head_gyr_X*np.sum(C_Gyr_x_H)
            + K_speed*np.sum(C_speed)
            + K_head_acc*np.sum(C_Acc_y_H)
            + K_torque*np.sum(C_torso_torque + C_pelvis_torque + C_pedaling_torque))

            return interval*J

        def obj_grad(prob, free):
            """
            Gradient of the objective function

            Parameters
            ----------
            free : TYPE
                DESCRIPTION.

            Returns
            -------
            grad : TYPE
                DESCRIPTION.

            """
            d = sol_dict(prob, free)
            print(d[prob.collocator.state_symbols[0]])

            grad = np.zeros_like(free)

            #Without pelvis motion
            # grad[2*NUM_NODES:3*NUM_NODES] = 2.0*interval*K_angles*(free[2*NUM_NODES:3*NUM_NODES] - x_meas_dict['yaw_angle_q3'])
            # grad[3*NUM_NODES:4*NUM_NODES] = 2.0*interval*K_angles*(free[3*NUM_NODES:4*NUM_NODES] - x_meas_dict['roll_angle_q4'])
            # grad[5*NUM_NODES:6*NUM_NODES] = 2.0*interval*K_angles*(free[5*NUM_NODES:6*NUM_NODES] - x_meas_dict['steer_angle_q7'])


            # grad[13*NUM_NODES:14*NUM_NODES] = 2.0*interval*K_angle_rates*(free[16*NUM_NODES:14*NUM_NODES] - x_meas_dict['speed'])

            # grad[9*NUM_NODES:10*NUM_NODES] = 2.0*interval*K_angle_rates*(free[9*NUM_NODES:10*NUM_NODES] - x_meas_dict['roll_rate_u4'])
            # grad[15*NUM_NODES:16*NUM_NODES] = 2.0*interval*K_angle_rates*(free[15*NUM_NODES:16*NUM_NODES] - x_meas_dict['yaw_rate_u3'])
            # grad[11*NUM_NODES:12*NUM_NODES] = 2.0*interval*K_angle_rates*(free[11*NUM_NODES:12*NUM_NODES] - x_meas_dict['steer_rate_u7'])
            # grad[12*NUM_NODES:13*NUM_NODES] = 2.0*interval*K_angle_rates*(free[12*NUM_NODES:13*NUM_NODES] - x_meas_dict['torso_lean_rate'])


            # grad[2*NUM_NODES:3*NUM_NODES] = 2.0*interval*K_angles*(free[2*NUM_NODES:3*NUM_NODES] - x_meas_dict['yaw_angle_q3'])
            # grad[3*NUM_NODES:4*NUM_NODES] = 2.0*interval*K_angles*(free[3*NUM_NODES:4*NUM_NODES] - x_meas_dict['roll_angle_q4'])
            # grad[5*NUM_NODES:6*NUM_NODES] = 2.0*interval*K_angles*(free[5*NUM_NODES:6*NUM_NODES] - x_meas_dict['steer_angle_q7'])

            # grad[9*NUM_NODES:10*NUM_NODES] = 2.0*interval*K_angle_rates*(free[9*NUM_NODES:10*NUM_NODES] - x_meas_dict['roll_rate_u4'])
            # grad[15*NUM_NODES:16*NUM_NODES] = 2.0*interval*K_angle_rates*(free[15*NUM_NODES:16*NUM_NODES] - x_meas_dict['yaw_rate_u3'])
            # grad[11*NUM_NODES:12*NUM_NODES] = 2.0*interval*K_angle_rates*(free[11*NUM_NODES:12*NUM_NODES] - x_meas_dict['steer_rate_u7'])

            # grad[13*NUM_NODES:14*NUM_NODES] = 2.0*interval*K_speed*(free[13*NUM_NODES:14*NUM_NODES] - x_meas_dict['speed'])


            # grad[18*NUM_NODES:19*NUM_NODES] = 2.0*interval*0*(free[18*NUM_NODES:19*NUM_NODES] - x_meas_dict['Acc_x_H'])
            # grad[19*NUM_NODES:20*NUM_NODES] = 2.0*interval*K_head_acc*(free[19*NUM_NODES:20*NUM_NODES] - x_meas_dict['Acc_y_H'])
            # grad[20*NUM_NODES:21*NUM_NODES] = 2.0*interval*0*(free[20*NUM_NODES:21*NUM_NODES] - x_meas_dict['Acc_z_H'])

            # grad[21*NUM_NODES:22*NUM_NODES] = 2.0*interval*K_head_gyr_X*(free[21*NUM_NODES:22*NUM_NODES] - x_meas_dict['Gyr_x_H'])
            # grad[22*NUM_NODES:23*NUM_NODES] = 2.0*interval*0*(free[22*NUM_NODES:23*NUM_NODES] - x_meas_dict['Gyr_y_H'])
            # grad[23*NUM_NODES:24*NUM_NODES] = 2.0*interval*0*(free[23*NUM_NODES:24*NUM_NODES] - x_meas_dict['Gyr_z_H'])

            grad[2*NUM_NODES:3*NUM_NODES] = 2.0*interval*K_angles*(free[2*NUM_NODES:3*NUM_NODES] - x_meas_dict['yaw_angle_q3'])
            grad[3*NUM_NODES:4*NUM_NODES] = 2.0*interval*K_angles*(free[3*NUM_NODES:4*NUM_NODES] - x_meas_dict['roll_angle_q4'])
            grad[5*NUM_NODES:6*NUM_NODES] = 2.0*interval*K_angles*(free[5*NUM_NODES:6*NUM_NODES] - x_meas_dict['steer_angle_q7'])

            grad[10*NUM_NODES:11*NUM_NODES] = 2.0*interval*K_angle_rates*(free[10*NUM_NODES:11*NUM_NODES] - x_meas_dict['roll_rate_u4'])
            grad[17*NUM_NODES:18*NUM_NODES] = 2.0*interval*K_angle_rates*(free[17*NUM_NODES:18*NUM_NODES] - x_meas_dict['yaw_rate_u3'])
            grad[12*NUM_NODES:13*NUM_NODES] = 2.0*interval*K_angle_rates*(free[12*NUM_NODES:13*NUM_NODES] - x_meas_dict['steer_rate_u7'])

            grad[15*NUM_NODES:16*NUM_NODES] = 2.0*interval*K_speed*(free[15*NUM_NODES:16*NUM_NODES] - x_meas_dict['speed'])


            grad[20*NUM_NODES:21*NUM_NODES] = 2.0*interval*0*(free[20*NUM_NODES:21*NUM_NODES] - x_meas_dict['Acc_x_H'])
            grad[21*NUM_NODES:22*NUM_NODES] = 2.0*interval*K_head_acc*(free[21*NUM_NODES:22*NUM_NODES] - x_meas_dict['Acc_y_H'])
            grad[22*NUM_NODES:23*NUM_NODES] = 2.0*interval*0*(free[22*NUM_NODES:23*NUM_NODES] - x_meas_dict['Acc_z_H'])

            grad[23*NUM_NODES:24*NUM_NODES] = 2.0*interval*K_head_gyr_X*(free[23*NUM_NODES:24*NUM_NODES] - x_meas_dict['Gyr_x_H'])
            grad[24*NUM_NODES:25*NUM_NODES] = 2.0*interval*0*(free[24*NUM_NODES:25*NUM_NODES] - x_meas_dict['Gyr_y_H'])
            grad[25*NUM_NODES:26*NUM_NODES] = 2.0*interval*0*(free[25*NUM_NODES:26*NUM_NODES] - x_meas_dict['Gyr_z_H'])

            grad[26*NUM_NODES:27*NUM_NODES] = 2.0*interval*K_torque*grad[26*NUM_NODES:27*NUM_NODES]
            grad[27*NUM_NODES:28*NUM_NODES] = 2.0*interval*K_torque*grad[27*NUM_NODES:28*NUM_NODES]
            grad[28*NUM_NODES:29*NUM_NODES] = 2.0*interval*K_torque*grad[28*NUM_NODES:29*NUM_NODES]

            return grad

        #I haven't use bounds so far because it tends to make the opti diverge
        
        bounds = {
             'q1(t)': (-0.2,  35),
             'q2(t)': (-4, 4),
             'q3(t)': (-np.deg2rad(45), np.deg2rad(45)),  # bicycle yaw
             'q4(t)': (-np.deg2rad(25), np.deg2rad(25)),    # bicycle roll
             'q5(t)': (np.deg2rad(10), np.deg2rad(40)),     # bicycle pitch

             'q7(t)': (-np.deg2rad(25), np.deg2rad(25)),    # steering
             'q8(t)': (-200.0, 10.0),

             'torsojoint_q(t)': (np.deg2rad(-20), np.deg2rad(20)),
}


        # bounds = {self.data.rider.torsojoint.q[0]: (np.deg2rad(-180), np.deg2rad(180))}


        # data.input_vars[0]: (-500.0, 500.0),
        # data.input_vars[1]: (-500.0, 500.0),
        # data.input_vars[2]: (-500.0, 500.0),

        self.data.constants['seat_pitch'] = - np.deg2rad(15)


        self.problem = Problem(
            obj,
            obj_grad,
            self.data.eoms,
            self.data.x,
            self.data.metadata.num_nodes,
            interval,
            known_parameter_map = self.data.constants,
            # instance_constraints=data.constraints.instance_constraints,
            bounds= bounds,
            time_symbol=me.dynamicsymbols._t,
            integration_method='midpoint'
            )

        max_item = 50000

        self.problem.add_option('max_iter' , max_item)

        # x0 = np.zeros((24+3, NUM_NODES)).flatten()

        # x0[2*NUM_NODES:3*NUM_NODES] = x_meas_dict['yaw_angle_q3']
        # x0[3*NUM_NODES:4*NUM_NODES] = x_meas_dict['roll_angle_q4']
        # x0[5*NUM_NODES:6*NUM_NODES] = x_meas_dict['steer_angle_q7']

        # x0[9*NUM_NODES:10*NUM_NODES] = x_meas_dict['roll_rate_u4']
        # x0[15*NUM_NODES:16*NUM_NODES] = x_meas_dict['yaw_rate_u3']
        # x0[11*NUM_NODES:12*NUM_NODES] = x_meas_dict['steer_rate_u7']

        # x0[10*NUM_NODES:11*NUM_NODES] = x_meas_dict['speed']
        # x0[13*NUM_NODES:14*NUM_NODES] = - x_meas_dict['speed'] /self.data.constants.get(Symbol('rear_wheel_r'))
        # x0[17*NUM_NODES:18*NUM_NODES] = - x_meas_dict['speed'] /self.data.constants.get(Symbol('front_wheel_r'))


        # x0[18*NUM_NODES:19*NUM_NODES] = x_meas_dict['Acc_x_H']
        # x0[19*NUM_NODES:20*NUM_NODES] = x_meas_dict['Acc_y_H']
        # x0[20*NUM_NODES:21*NUM_NODES] = x_meas_dict['Acc_z_H']

        # x0[21*NUM_NODES:22*NUM_NODES] = x_meas_dict['Gyr_x_H']
        # x0[22*NUM_NODES:23*NUM_NODES] = x_meas_dict['Gyr_y_H']
        # x0[23*NUM_NODES:24*NUM_NODES] = x_meas_dict['Gyr_z_H']

        x0 = np.zeros((26+3, NUM_NODES)).flatten()

        x0[2*NUM_NODES:3*NUM_NODES] = x_meas_dict['yaw_angle_q3']
        x0[3*NUM_NODES:4*NUM_NODES] = x_meas_dict['roll_angle_q4']
        x0[5*NUM_NODES:6*NUM_NODES] = x_meas_dict['steer_angle_q7']

        x0[10*NUM_NODES:11*NUM_NODES] = x_meas_dict['roll_rate_u4']
        x0[17*NUM_NODES:18*NUM_NODES] = x_meas_dict['yaw_rate_u3']
        x0[12*NUM_NODES:13*NUM_NODES] = x_meas_dict['steer_rate_u7']

        x0[15*NUM_NODES:16*NUM_NODES] = x_meas_dict['speed']
        x0[11*NUM_NODES:12*NUM_NODES] = - x_meas_dict['speed'] /self.data.constants.get(Symbol('rear_wheel_r'))
        x0[19*NUM_NODES:20*NUM_NODES] = - x_meas_dict['speed'] /self.data.constants.get(Symbol('front_wheel_r'))


        x0[20*NUM_NODES:21*NUM_NODES] = x_meas_dict['Acc_x_H']
        x0[21*NUM_NODES:22*NUM_NODES] = x_meas_dict['Acc_y_H']
        x0[22*NUM_NODES:23*NUM_NODES] = x_meas_dict['Acc_z_H']

        x0[23*NUM_NODES:24*NUM_NODES] = x_meas_dict['Gyr_x_H']
        x0[24*NUM_NODES:25*NUM_NODES] = x_meas_dict['Gyr_y_H']
        x0[25*NUM_NODES:26*NUM_NODES] = x_meas_dict['Gyr_z_H']




        self.initial_guess = x0  # u


        self.problem_to_save = {'eoms' : self.data.eoms,
                                'x' : self.data.x,
                                'num_nodes' : self.data.metadata.num_nodes,
                                'max_item' : max_item,
                                'x0' : x0}

        #idee initialiser avec les derives des vitesses


    def solve_and_save(self):

        self.solution, self.info = self.problem.solve(self.initial_guess)
        self.data.solution = self.solution
        self.time_simu = np.linspace(0, self.DURATION, num = self.NUM_NODES)



        if not os.path.exists(f"results/{self.date}_part_{self.n_part}_trial_{self.n_trial}"):
            os.makedirs(f"results/{self.date}_part_{self.n_part}_trial_{self.n_trial}")

        try:
            self.problem.plot_constraint_violations(self.solution)
        except:
            plt.savefig(f'results/{self.date}_part_{self.n_part}_trial_{self.n_trial}/constraints_violation.png')
            plt.close()


        dict_saved = {'metadata' : self.data,
                      'x_meas_dict' : self.x_meas_dict,
                      'initial_guess' : self.initial_guess,
                      'solution' : self.solution,
                      'info' : self.info,
                      'n_part': self.n_part,
                      'n_trial': self.n_trial,
                      'time_simu' : self.time_simu,
                      'problem' : self.problem_to_save}

        try:
            if not os.path.exists(f"results/{self.date}_part_{self.n_part}_trial_{self.n_trial}"):
                os.makedirs(f"results/{self.date}_part_{self.n_part}_trial_{self.n_trial}")

            dict_saved_file = open(f'results/{self.date}_part_{self.n_part}_trial_{self.n_trial}/{self.date}_part_{self.n_part}_trial_{self.n_trial}.pkl', 'wb')
            pickle.dump(dict_saved, dict_saved_file)
            dict_saved_file.close()

        except:
            print("Something went wrong when saving dict_saved_file")

        print(self.info['status_msg'])


        # self.NUM_NODES = NUM_NODES

        # self.data.solution_state()
        # self.data.solution_input()

    def load_results(self, results_path):

        with open(f'{results_path}.pkl', 'rb') as file:
            dict_loaded = pickle.load(file)


        self.data = dict_loaded['metadata']
        self.x_meas_dict = dict_loaded['x_meas_dict']
        self.initial_guess = dict_loaded['initial_guess']
        self.solution = dict_loaded['solution']
        self.info = dict_loaded['info']
        self.n_part = dict_loaded['n_part']
        self.n_trial = dict_loaded['n_trial']
        self.time_simu = dict_loaded['time_simu']
        self.NUM_NODES = len(self.time_simu)
        self.date = results_path.split('_part')[0].split('/')[1]


        # index = self.n_trial
        # for sig_name in self.data.x:


    def load_results_for_initial_guess(self, results_path):

        with open(f'{results_path}.pkl', 'rb') as file:
            dict_loaded = pickle.load(file)

        self.dict_loaded = dict_loaded


        loaded_var_names = list(self.dict_loaded['metadata'].x)
        loaded_input_var_names = list(self.dict_loaded['metadata'].input_vars)
        loaded_all_var = np.array(loaded_var_names + loaded_input_var_names)
        loaded_num_nodes = self.dict_loaded['problem']['num_nodes']

        if self.NUM_NODES != loaded_num_nodes:
            raise ValueError(f"Missmatch in num_nodes, current num_nodes is {self.NUM_NODES} while loaded one is {loaded_num_nodes}")


        loaded_solution = self.dict_loaded['solution'].reshape(len(loaded_var_names) + len(loaded_input_var_names), loaded_num_nodes)


        current_var_names = list(self.data.x)
        current_input_var_names = list(self.data.input_vars)

        for k, var in enumerate(current_var_names + current_input_var_names):
            if var in loaded_all_var:
                var_index = np.argwhere(loaded_all_var == var)


                self.initial_guess[k:k+self.NUM_NODES] = loaded_solution[var_index]
                print('Previous solution loaded for:',var)


        # self.initial_guess = dict_loaded['solution']
        # print('Older solution imported as initial guess')




    def plot_res_type_1(self, save, path, figname):

        plt.rcParams['lines.linewidth'] = 1

        fig, axs = plt.subplots(3, 2, figsize=(6*3, 8*3), sharex=True)

        color = ['b','g','r','black']

        # Trajectories
        axs[0,0].plot(self.time_simu, self.solution[1*self.NUM_NODES:(1+1)*self.NUM_NODES])
        # axs[0,0].set_xlim()
        # axs[0,0].set_aspect('equal', adjustable='box')
        # axs[0,0].set_title('Trajectory')
        axs[0,0].set_xlabel('time [s]')
        axs[0,0].set_ylabel('y [m]')

        # Speed
        u_opti = self.solution[13*self.NUM_NODES:(13+1)*self.NUM_NODES]
        u_exp = self.x_meas_dict['speed']
        axs[0,1].plot(self.time_simu, u_opti,color = color[0], label = '$u_{opt}$'+f'- RMSE: {round(RMSE(u_opti, u_exp),3)}')
        axs[0,1].plot(self.time_simu, u_exp,color = color[0], ls = '--',label = 'exp')
        axs[0,1].set_xlim(self.time_simu[0], self.time_simu[-1])
        axs[0,1].set_title('Speed')
        axs[0,1].set_ylabel('u [m/s]')
        axs[0,1].set_xlabel('time [s]')
        axs[0,1].legend(bbox_to_anchor=(1.01, 1.05))


        ## Angles

        #Yaw angle
        psi_opti = np.rad2deg(self.solution[2*self.NUM_NODES:(2+1)*self.NUM_NODES])
        psi_exp = np.rad2deg(self.x_meas_dict['yaw_angle_q3'])
        RMSE_psi = round(RMSE(psi_opti, psi_exp),3)
        axs[1,0].plot(self.time_simu, psi_opti, color = color[0] ,label = '$\psi_{opt}$'+f'- RMSE: {RMSE_psi}')
        axs[1,0].plot(self.time_simu, psi_exp, ls = '--',label = '$\psi_{meas}$', color = color[0])

        #Roll angle
        phi_opti = np.rad2deg(self.solution[3*self.NUM_NODES:(3+1)*self.NUM_NODES])
        phi_exp = np.rad2deg(self.x_meas_dict['roll_angle_q4'])
        RMSE_phi = round(RMSE(phi_opti, phi_exp),3)

        axs[1,0].plot(self.time_simu, phi_opti, color = color[1], label = '$\phi_{opt}$'+ f'- RMSE: {RMSE_phi}')
        axs[1,0].plot(self.time_simu, phi_exp, ls = '--',label = '$\phi_{meas}$', color = color[1])

        #Steer angle
        delta_opti = np.rad2deg(self.solution[5*self.NUM_NODES:(5+1)*self.NUM_NODES])
        delta_exp = np.rad2deg(self.x_meas_dict['steer_angle_q7'])
        RMSE_delta = round(RMSE(delta_opti, delta_exp),3)
        axs[1,0].plot(self.time_simu, delta_opti, color = color[2], label = '$\delta_{opt}$'+f'- RMSE: {RMSE_delta}')
        axs[1,0].plot(self.time_simu, delta_exp, ls = '--',label = '$\delta_{meas}$', color = color[2])

        # Lean angle: theta
        theta_opti = np.rad2deg(self.solution[7*self.NUM_NODES:(7+1)*self.NUM_NODES])
        axs[1,0].plot(self.time_simu, theta_opti, color = color[3], label = '$\\theta_{opt}$')
        # axs[1,0].plot(self.time_simu, np.rad2deg(theta), color = color[3], label = '$\\theta_{meas}$', ls = '--')


        axs[1,0].set_xlim(self.time_simu[0], self.time_simu[-1])
        axs[1,0].set_title('Angles')
        axs[1,0].set_ylabel('Angles [deg]')
        axs[1,0].set_xlabel('time [s]')
        axs[1,0].legend(bbox_to_anchor=(1.01, 1.05))


        ## Angle Rates

        #Yaw angle rate
        psi_dot_opti = np.rad2deg(self.solution[15*self.NUM_NODES:(15+1)*self.NUM_NODES])
        psi_dot_exp = np.rad2deg(self.x_meas_dict['yaw_rate_u3'])
        RMSE_psi_dot = round(RMSE(psi_dot_opti, psi_dot_exp),3)
        axs[2,0].plot(self.time_simu, psi_dot_opti, color = color[0] ,label = '$\dot{\psi_{opt}}$'+f'- RMSE: {RMSE_psi_dot}')
        axs[2,0].plot(self.time_simu, psi_dot_exp, ls = '--',label = '$\dot{\psi_{meas}}$', color = color[0])

        #Roll angle rate
        phi_dot_opti = np.rad2deg(self.solution[9*self.NUM_NODES:(9+1)*self.NUM_NODES])
        phi_dot_exp = np.rad2deg(self.x_meas_dict['roll_rate_u4'])
        RMSE_phi_dot = round(RMSE(phi_dot_opti, phi_dot_exp),3)
        axs[2,0].plot(self.time_simu, phi_dot_opti, color = color[1], label = '$\dot{\phi_{opt}}$'+ f'- RMSE: {RMSE_phi_dot}')
        axs[2,0].plot(self.time_simu, phi_dot_exp, ls = '--',label = '$\dot{\phi_{meas}}$', color = color[1])

        #Steer angle rate
        delta_dot_opti = np.rad2deg(self.solution[11*self.NUM_NODES:(11+1)*self.NUM_NODES])
        delta_dot_exp = np.rad2deg(self.x_meas_dict['steer_rate_u7'])
        RMSE_dot_delta = round(RMSE(delta_opti, delta_exp),3)
        axs[2,0].plot(self.time_simu, delta_dot_opti, color = color[2], label = '$\dot{\delta_{opt}}$'+f'- RMSE: {RMSE_dot_delta}')
        axs[2,0].plot(self.time_simu, delta_dot_exp, ls = '--',label = '$\dot{\delta_{meas}}$', color = color[2])

        # Lean angle: theta
        theta_dot_opti = np.rad2deg(self.solution[12*self.NUM_NODES:(12+1)*self.NUM_NODES])
        # theta_dot_exp = np.rad2deg(self.x_meas_dict['torso_lean_rate'])
        # RMSE_dot_theta = round(RMSE(theta_dot_opti, theta_dot_exp),3)

        axs[2,0].plot(self.time_simu, theta_dot_opti, color = color[3], label = '$\dot{\\theta_{opt}}$')
        # axs[2,0].plot(self.time_simu, theta_dot_exp, ls = '--', label = '$\dot{\\theta_{meas}}$', color = color[3])


        axs[2,0].set_xlim(self.time_simu[0], self.time_simu[-1])
        axs[2,0].set_title('Angles rates')
        axs[2,0].set_ylabel('Angles rates [deg/s]')
        axs[2,0].set_xlabel('time [s]')
        axs[2,0].legend(bbox_to_anchor=(1.01, 1.05))


        #Torques

        #Balance control actions
        axs[1,1].plot(self.time_simu, self.solution[26*self.NUM_NODES:(26+1)*self.NUM_NODES], label = '$T_{lean}$', color = color[0])
        axs[1,1].plot(self.time_simu, self.solution[24*self.NUM_NODES:(24+1)*self.NUM_NODES], label = '$T_{roll}$', color = color[1])

        axs[1,1].set_xlim(self.time_simu[0], self.time_simu[-1])
        axs[1,1].set_title('Balance control torques')
        axs[1,1].set_xlabel('time [s]')
        axs[1,1].set_ylabel('Torque [Nm]')
        axs[1,1].legend(bbox_to_anchor=(1.01, 1.05))


        #Longitudinal motion control action
        axs[2,1].plot(self.time_simu, self.solution[25*self.NUM_NODES:(25+1)*self.NUM_NODES], label = '$T_{pedal}$', color = color[0])
        axs[2,1].set_xlim(self.time_simu[0], self.time_simu[-1])
        axs[2,1].set_title('Longitudinal motion torques')
        axs[2,1].set_xlabel('time [s]')
        axs[2,1].set_ylabel('Torque [Nm]')
        axs[2,1].legend(bbox_to_anchor=(1.01, 1.05))


        plt.tight_layout()

        if save == 1:
            plt.savefig(f'{path}/{figname}'+'_1_2'+'.svg')
            plt.savefig(f'{path}/{figname}'+'_1_2'+'.png')
            plt.close()


        fig2, axs2 = plt.subplots(3, 2, figsize=(6*3, 8*3), sharex=True)

        Acc_x_H_opti = self.solution[18*self.NUM_NODES:(18+1)*self.NUM_NODES]
        Acc_x_H_exp = self.x_meas_dict['Acc_x_H']
        RMSE_Acc_x_H = round(RMSE(Acc_x_H_opti, Acc_x_H_exp), 3)
        axs2[0, 0].plot(self.time_simu, Acc_x_H_opti, color = color[0] ,label = '$Acc_x$'+f'- RMSE: {RMSE_Acc_x_H}')
        axs2[0, 0].plot(self.time_simu, Acc_x_H_exp, ls = '--',label = '$Accx_{meas}$', color = color[0])

        Acc_y_H_opti = self.solution[19*self.NUM_NODES:(19+1)*self.NUM_NODES]
        Acc_y_H_exp = self.x_meas_dict['Acc_y_H']
        RMSE_Acc_y_H = round(RMSE(Acc_y_H_opti, Acc_y_H_exp), 3)
        axs2[1, 0].plot(self.time_simu, Acc_y_H_opti, color = color[0] ,label = '$Acc_y$'+f'- RMSE: {RMSE_Acc_y_H}')
        axs2[1, 0].plot(self.time_simu, Acc_y_H_exp, ls = '--',label = '$Accy_{meas}$', color = color[0])

        Acc_z_H_opti = self.solution[20*self.NUM_NODES:(20+1)*self.NUM_NODES]
        Acc_z_H_exp = self.x_meas_dict['Acc_z_H']
        RMSE_Acc_z_H = round(RMSE(Acc_z_H_opti, Acc_z_H_exp), 3)
        axs2[2, 0].plot(self.time_simu, Acc_z_H_opti, color = color[0] ,label = '$Acc_z$'+f'- RMSE: {RMSE_Acc_z_H}')
        axs2[2, 0].plot(self.time_simu, Acc_z_H_exp, ls = '--',label = '$Accz_{meas}$', color = color[0])


        Gyr_x_H_opti = np.rad2deg(self.solution[21*self.NUM_NODES:(21+1)*self.NUM_NODES])
        Gyr_x_H_exp = np.rad2deg(self.x_meas_dict['Gyr_x_H'])
        RMSE_Gyr_x_H = round(RMSE(Gyr_x_H_opti, Gyr_x_H_exp), 3)
        axs2[0, 1].plot(self.time_simu, Gyr_x_H_opti, color = color[0] ,label = '$Gyr_x$'+f'- RMSE: {RMSE_Gyr_x_H}')
        axs2[0, 1].plot(self.time_simu, Gyr_x_H_exp, ls = '--',label = '$Gyrx_{meas}$', color = color[0])

        Gyr_y_H_opti = np.rad2deg(self.solution[22*self.NUM_NODES:(22+1)*self.NUM_NODES])
        Gyr_y_H_exp = np.rad2deg(self.x_meas_dict['Gyr_y_H'])
        RMSE_Gyr_y_H = round(RMSE(Gyr_y_H_opti, Gyr_y_H_exp), 3)
        axs2[1, 1].plot(self.time_simu, Gyr_y_H_opti, color = color[0] ,label = '$Gyr_y$'+f'- RMSE: {RMSE_Gyr_y_H}')
        axs2[1, 1].plot(self.time_simu, Gyr_y_H_exp, ls = '--',label = '$Gyry_{meas}$', color = color[0])

        Gyr_z_H_opti = np.rad2deg(self.solution[23*self.NUM_NODES:(23+1)*self.NUM_NODES])
        Gyr_z_H_exp = np.rad2deg(self.x_meas_dict['Gyr_z_H'])
        RMSE_Gyr_z_H = round(RMSE(Gyr_z_H_opti, Gyr_z_H_exp), 3)
        axs2[2, 1].plot(self.time_simu, Gyr_z_H_opti, color = color[0] ,label = '$Gyr_z$'+f'- RMSE: {RMSE_Gyr_z_H}')
        axs2[2, 1].plot(self.time_simu, Gyr_z_H_exp, ls = '--',label = '$Gyrz_{meas}$', color = color[0])
        plt.tight_layout()

        axs2[0,0].legend(bbox_to_anchor=(1.01, 1.05))
        axs2[1,0].legend(bbox_to_anchor=(1.01, 1.05))
        axs2[2,0].legend(bbox_to_anchor=(1.01, 1.05))
        axs2[0,1].legend(bbox_to_anchor=(1.01, 1.05))
        axs2[1,1].legend(bbox_to_anchor=(1.01, 1.05))
        axs2[2,1].legend(bbox_to_anchor=(1.01, 1.05))



        if save == 1:
            plt.savefig(f'{path}/{figname}'+'_2_2'+'.svg')
            plt.savefig(f'{path}/{figname}'+'_2_2'+'.png')
            plt.close()


    def plot_results(self):


        self.plot_res_type_1(save = 1, path = f"results/{self.date}_part_{self.n_part}_trial_{self.n_trial}", figname = 'results_type_1')
        # problem.plot_objective_value()

        # create_animation(data, output = 'test.gif') #Works ok

        create_animation2(self.data, output = f'results/{self.date}_part_{self.n_part}_trial_{self.n_trial}/animation') # use this one

        # bike_following_animation(data, output = 'test.gif') # Get error
        # set_simulator(self.data)
        # try:
        #     bike_following_animation2(self.data, output = f'results/{self.date}_part_{self.n_part}_trial_{self.n_trial}/following_animation', angly=0, elevv=10) # Should imporved
        # except:
        #     pass


    def solve_save_plot_all_trials(self):

        for n_trial in self.list_trials:
            print('SOLVING TRIAL', n_trial)
            self.initialize_problem(n_trial)
            self.solve_and_save()
            self.plot_results()


    def TEST(self, res_path):


        self.initialize_problem(54)
        # self.load_results_for_initial_guess(res_path)
        # self.solve_and_save()
        # self.plot_results()




#%% Use of the class

# =============================================================================
#  Solve Inverse Dyn for all trials
# =============================================================================
# n_part = 1
# ids = OLDC_Solver(f'data/Hand_off_on_experiment/part_{n_part}_Hands_Off_On.csv', n_part)
# ids.select_trial([21, 23, 25, 27, 31, 35, 37, 39, 41])
# ids.solve_save_plot_all_trials()

# n_part = 2
# ids = OLDC_Solver(f'data/Hand_off_on_experiment/part_{n_part}_Hands_Off_On.csv', n_part)
# ids.select_trial([])
# ids.solve_save_plot_all_trials()

# n_part = 3
# ids = OLDC_Solver(f'data/Hand_off_on_experiment/part_{n_part}_Hands_Off_On.csv', n_part)
# ids.select_trial([68, 74])
# ids.solve_save_plot_all_trials()


n_part = 4
ids = OLDC_Solver(f'data/Hand_off_on_experiment/part_{n_part}_Hands_Off_On.csv', n_part)
ids.select_trial([54, 60, 62, 67, 69, 75, 77, 83, 85])
# ids.solve_save_plot_all_trials()

res_path = 'results/2025_09_25_10_43_33_part_4_trial_54/2025_09_25_10_43_33_part_4_trial_54'
# ids.TEST(res_path)

# print(ids.data.x)
ids.initialize_problem(54)
# ids.load_results_for_initial_guess(res_path)
ids.solve_and_save()
ids.plot_results()


#cr=0.01221


# ids.load_results(res_path)
# ids.plot_results()


# =============================================================================
# Solve inverse Dyn for only one trial
# =============================================================================
# ids = OLDC_Solver('data/Hand_off_on_experiment/part_1_Hands_Off_On.csv', 1)
# ids.initialize_problem(21)
# ids.solve_and_save()
# ids.plot_results()

# =============================================================================
# Import results and plot them
# =============================================================================

# n_part = 3
# ids = OLDC_Solver(f'data/Hand_off_on_experiment/part_{n_part}_Hands_Off_On.csv', n_part)
# res_path = 'results/2025_07_17_01_18_01_part_1_trial_7/2025_07_17_01_18_01_part_1_trial_7'
# ids.load_results(res_path)
# ids.plot_results()




