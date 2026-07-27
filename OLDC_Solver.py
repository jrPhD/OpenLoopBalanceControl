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

#New module to import a bicycle model
from bicycle_models import generate_model
from utils import animate_solution, butter_bandpass_filter, butter_lowpass_filter



# from CYINTIA import cyclist, set_axes_equal #My cyclist inertia toolbox

def sol_dict(prob, free, str_as_keys=False):
    """Returns a dictionary that has SymPy dynamic symbols mapped to
    the solution.

    Parameters
    ==========
    prob : Problem
        Instance of an opty Problem.
    free : array_like, shape(n*N + q*N,)
        The free optimization variables.
    str_as_keys : boolean
        If True, the dictionary key will be a string version of the function of
        time.

    Returns
    =======
    d : dictionary
        Maps dynamicsymbols to array of shape(N,)

    """
    x, r, _ = prob.parse_free(free)
    d = {}
    for symbol, array in zip(prob.collocator.state_symbols, x):
        if str_as_keys:
            k = symbol.__class__.name
        else:
            k = symbol
        d[k] = array
    for symbol, array in zip(prob.collocator.unknown_input_trajectories, r):
        if str_as_keys:
            k = symbol.__class__.name
        else:
            k = symbol
        d[k] = array
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




class OLDC_model():

    #Open Loop Direct Collocation Solver

    def __init__(self, exp_file_path, run, config):

        self.n_trial = str(run)
        self.df_exp = pd.read_csv(exp_file_path + f'states_{run}.csv')
        # self.list_hands_off_trials = self.df_exp[self.df_exp['condition'] == 'straight/hands_off']['trial'].unique()
        self.n_part = run[0]
        self.date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
        runInfo = pd.read_csv('D:/Users/ronne/Documents/4_side_quest/OpenLoopBalanceControl/data/Moore_mocap/runInfo.csv', sep=';')


        self.treadmill_speed = float(runInfo[runInfo['run'] == int(run)]['speed'])/3.6
        self.config = config

        #Extract config parameters

        self.N_sampling = config['N_sampling']




        # print("Trials numbers for straight/hands_off condition", self.list_hands_off_trials)


    def select_trial(self, list_trials = []):

        if list_trials == []:
            self.list_trials = self.list_hands_off_trials

        else:
             self.list_trials = list_trials

    def initialize_model(self, null_state_solving=False):


        # df_trial = self.df_exp[self.df_exp['trial'] == n_trial]


        # not use this line for general use
        df_trial = self.df_exp[(self.df_exp['time']>15) & (self.df_exp['time']<20)]

        fs = np.mean(1/np.diff(df_trial['time'].to_numpy()))
        fs_adjusted = fs/self.N_sampling

        time = df_trial['time'].to_numpy()[::self.N_sampling]

        u = df_trial['speed'].to_numpy()[::self.N_sampling]


        if null_state_solving == True: #Sloving the problem for null state
            x_meas_dict  = {'time' : time,
                            'yaw_angle_q3' :     np.zeros(len(df_trial)),
                            'roll_angle_q4' :     np.zeros(len(df_trial)),
                            'steer_angle_q7' :    np.zeros(len(df_trial)),
                            # 'pitch_angle_q5' :    theta,
                            'speed' : np.mean(u)*np.ones(len(df_trial)),
                            'roll_rate_u4' :      np.zeros(len(df_trial)),
                            'yaw_rate_u3' :       np.zeros(len(df_trial)),
                            # 'pitch_rate_u5' :     theta_dot,
                            'steer_rate_u7' :     np.zeros(len(df_trial)),
                            'roll_acc' :        np.zeros(len(df_trial)),
                            'steer_acc' :       np.zeros(len(df_trial)),
                           } #Include here the signals used to feed the opti

            # self.config['steer_torque'] = False
            self.config['roll_control'] = False



        else:

            psi = df_trial['yaw angle'].to_numpy()[::self.N_sampling]
            phi = df_trial['roll angle'].to_numpy()[::self.N_sampling]
            delta = df_trial['steer angle'].to_numpy()[::self.N_sampling]
            # theta = df_trial['pitch angle'].to_numpy()[::self.N_sampling]


            psi_dot = np.gradient(psi, time)
            phi_dot = np.gradient(phi, time)
            delta_dot = np.gradient(delta, time)


            phi_ddot = np.gradient(phi_dot, time)
            delta_ddot = np.gradient(delta_dot, time)

            # theta_dot = np.gradient(theta, time)



            if self.config['filtering']['use_filtering'] == True:

                lowcut = self.config['filtering']['low_cut_freq']
                highcut = self.config['filtering']['high_cut_freq']

                # psi = butter_bandpass_filter(psi, lowcut, highcut, fs_adjusted, order=4)
                # phi = butter_bandpass_filter(phi, lowcut, highcut, fs_adjusted, order=4)
                # delta = butter_bandpass_filter(delta, lowcut, highcut, fs_adjusted, order=4)

                psi = butter_lowpass_filter(psi, highcut, fs_adjusted, order=4)
                phi = butter_lowpass_filter(phi, highcut, fs_adjusted, order=4)
                delta = butter_lowpass_filter(delta, highcut, fs_adjusted, order=4)



                # theta= butter_bandpass_filter(theta, lowcut, highcut, fs_adjusted, order=4)
                u = butter_lowpass_filter(u, highcut, fs_adjusted, order=4)

                psi_dot = butter_lowpass_filter(psi_dot, highcut, fs_adjusted, order=4)
                phi_dot = butter_lowpass_filter(phi_dot, highcut, fs_adjusted, order=4)
                delta_dot = butter_lowpass_filter(delta_dot, highcut, fs_adjusted, order=4)
                # theta_dot= butter_bandpass_filter(theta_dot, lowcut, highcut, fs_adjusted, order=4)

            x_meas_dict  = {'time' : time,
                            'yaw_angle_q3' :     psi,
                            'roll_angle_q4' :     phi,
                            'steer_angle_q7' :    delta,
                            # 'pitch_angle_q5' :    theta,
                            'speed' : u,
                            'roll_rate_u4' :      phi_dot,
                            'yaw_rate_u3' :       psi_dot,
                            # 'pitch_rate_u5' :     theta_dot,
                            'steer_rate_u7' :     delta_dot,
                            'roll_acc' :        phi_ddot,
                            'steer_acc' :       delta_ddot,
                           # 'seat_rate':          theta_dot,
                           # 'Acc_x_H' : Acc_x_H,
                           # 'Acc_y_H' : Acc_y_H,
                           # 'Acc_z_H' : Acc_z_H,

                           # 'Gyr_x_H' : Gyr_x_H,
                           # 'Gyr_y_H' : Gyr_y_H,
                           # 'Gyr_z_H' : Gyr_z_H,

                           } #Include here the signals used to feed the opti


        NUM_NODES = len(time)

        self.NUM_NODES = NUM_NODES

        DURATION = time[-1] - time[0]

        self.DURATION = DURATION

        self.interval = DURATION/NUM_NODES  # seconds

        self.x_meas_dict = x_meas_dict

        t, x, r, eoms, p, bicycle = generate_model('model', self.config)

        self.x = x
        self.eoms = eoms
        self.p = p
        self.r = r
        self.t = t
        self.bicycle = bicycle

        self.x_animate = np.zeros((18, len(time)))
        self.x_animate[2,:] = x_meas_dict['yaw_angle_q3']
        self.x_animate[3,:] = x_meas_dict['roll_angle_q4']
        # self.x_animate[4,:] = x_meas_dict['pitch_angle_q5']
        self.x_animate[6,:] = x_meas_dict['steer_angle_q7']
        self.x_animate[8,:] = x_meas_dict['speed']



        # eval_ang_vel_mat_head_cst = ['seat_roll','seat_pitch','seat_yaw','torsojoint_theta']
        # eval_ang_vel_mat_head_cst = ['torsojoint_theta']

        # eval_ang_vel_head_cst_num = [self.data.constants.get(Symbol(cst)) for cst in eval_ang_vel_mat_head_cst]


        # eval_ang_vel_mat_head = sm.lambdify((self.data.x, eval_ang_vel_mat_head_cst), ang_vel_mat_head[:])
        # eval_acc_mat_head = sm.lambdify((self.data.x, eval_ang_vel_mat_head_cst), acc_mat_head[:])
        # print('----------------')
        # print(self.data.eoms[-1])
        # print(self.data.x)
        # print('----------------')


        # print(eval_ang_vel_mat_head(np.random.random(18), eval_ang_vel_head_cst_num))
        # print(eval_acc_mat_head(np.random.random(18), eval_ang_vel_head_cst_num))



        # x_meas_vec = np.array([x_meas_dict[k] for k in x_meas_dict.keys()]).flatten()



    def plot_measurments(self, save, path, figname):

        plt.rcParams['lines.linewidth'] = 1

        fig, axs = plt.subplots(3, 2, figsize=(6*3, 8*3), sharex=True)

        color = ['b','g','r','black']

        time = self.x_meas_dict['time']

        # Trajectories
        # axs[0,0].plot(time, )
        # axs[0,0].set_xlim()
        # axs[0,0].set_aspect('equal', adjustable='box')
        # axs[0,0].set_title('Trajectory')
        axs[0,0].set_xlabel('time [s]')
        axs[0,0].set_ylabel('y [m]')

        # Speed
        u_exp = self.x_meas_dict['speed']
        axs[0,1].plot(time, u_exp,color = color[0], ls = '--',label = 'exp')
        axs[0,1].set_xlim(time[0], time[-1])
        axs[0,1].set_title('Speed')
        axs[0,1].set_ylabel('u [m/s]')
        axs[0,1].set_xlabel('time [s]')
        axs[0,1].legend(bbox_to_anchor=(1.01, 1.05))


        ## Angles

        #Yaw angle
        psi_exp = np.rad2deg(model.x_meas_dict['yaw_angle_q3'])
        axs[1,0].plot(time, psi_exp, ls = '--',label = '$\psi_{meas}$', color = color[0])

        #Roll angle
        phi_exp = np.rad2deg(model.x_meas_dict['roll_angle_q4'])
        axs[1,0].plot(time, phi_exp, ls = '--',label = '$\phi_{meas}$', color = color[1])

        #Steer angle
        delta_exp = np.rad2deg(model.x_meas_dict['steer_angle_q7'])
        axs[1,0].plot(time, delta_exp, ls = '--',label = '$\delta_{meas}$', color = color[2])

        # Lean angle: theta
        # theta_exp = np.rad2deg(model.x_meas_dict['theta_angle_q5'])
        # axs[1,0].plot(time, np.rad2deg(theta_exp), color = color[3], label = '$\\theta_{meas}$', ls = '--')


        axs[1,0].set_xlim(time[0], time[-1])
        axs[1,0].set_title('Angles')
        axs[1,0].set_ylabel('Angles [deg]')
        axs[1,0].set_xlabel('time [s]')
        axs[1,0].legend(bbox_to_anchor=(1.01, 1.05))


        ## Angle Rates

        #Yaw angle rate
        psi_dot_exp = np.rad2deg(model.x_meas_dict['yaw_rate_u3'])
        axs[2,0].plot(time, psi_dot_exp, ls = '--',label = '$\dot{\psi_{meas}}$', color = color[0])

        #Roll angle rate
        phi_dot_exp = np.rad2deg(model.x_meas_dict['roll_rate_u4'])
        axs[2,0].plot(time, phi_dot_exp, ls = '--',label = '$\dot{\phi_{meas}}$', color = color[1])

        #Steer angle rate
        delta_dot_exp = np.rad2deg(model.x_meas_dict['steer_rate_u7'])
        axs[2,0].plot(time, delta_dot_exp, ls = '--',label = '$\dot{\delta_{meas}}$', color = color[2])

        # Pitch angle: theta
        theta_dot_exp = np.rad2deg(model.x_meas_dict['pitch_rate_u5'])
        axs[2,0].plot(time, theta_dot_exp, ls = '--', label = '$\dot{\\theta_{meas}}$', color = color[3])


        axs[2,0].set_xlim(time[0], time[-1])
        axs[2,0].set_title('Angles rates')
        axs[2,0].set_ylabel('Angles rates [deg/s]')
        axs[2,0].set_xlabel('time [s]')
        axs[2,0].legend(bbox_to_anchor=(1.01, 1.05))


        plt.tight_layout()

        if save == 1:
            plt.savefig(f'{path}/{figname}'+'.png')
            plt.close()

    def plot_measurments_animation(self):

        if not os.path.exists(f"results/{model.date}_part_{model.n_part}_trial_{model.n_trial}"):
            os.makedirs(f"results/{model.date}_part_{model.n_part}_trial_{model.n_trial}")


        self.plot_measurments(save = 1, path = f"results/{self.date}_part_{self.n_part}_trial_{self.n_trial}", figname = 'measurments')
        # problem.plot_objective_value()

        # create_animation(data, output = 'test.gif') #Works ok

        create_animation_meas(self.data, self.x_animate, output = f'results/{self.date}_part_{self.n_part}_trial_{self.n_trial}/animation_measurements')



class OLDC_solver():

    #Open Loop Direct Collocation Solver

    def __init__(self, model, N_iteration, x0=None):

        #Weigths set to 1 until now
        K_angles = 15
        K_angle_rates = 1
        K_effort = 1

        NUM_NODES = model.NUM_NODES
        self.NUM_NODES = NUM_NODES

        State_var_names = [
            'yaw_angle_q3',
            'roll_angle_q4',
            'steer_angle_q7',
            'roll_rate_u4',
            'yaw_rate_u3',
            'steer_rate_u7']

        for var in State_var_names:
            model.x_meas_dict[var] = model.x_meas_dict[var]/N_iteration


        def obj(prob, free):
            """Minimize the error in all of the states."""
            d = sol_dict(prob, free, str_as_keys=True)

            C_yaw = (model.x_meas_dict['yaw_angle_q3'] - d['model_q3'])**2
            C_roll = (model.x_meas_dict['roll_angle_q4'] - d['model_q4'])**2
            C_steer = (model.x_meas_dict['steer_angle_q7'] - d['model_q7'])**2
            # C_pitch = (model.x_meas_dict['pitch_angle_q5'] - d['model_q5'])**2


            C_roll_rate = (model.x_meas_dict['roll_rate_u4'] - d['model_u4'])**2
            C_yaw_rate = (model.x_meas_dict['yaw_rate_u3'] - d['model_u3'])**2
            C_steer_rate = (model.x_meas_dict['steer_rate_u7'] - d['model_u7'])**2
            # C_pitch_rate = (model.x_meas_dict['pitch_rate_u5'] - d['model_u5'])**2

            C_speed = (model.x_meas_dict['speed'] - d['model_u1'])**2

            J = (K_angles*np.sum(C_yaw + C_roll + C_steer)
                 # + K_speed*np.sum(C_speed)
                 + K_angle_rates*np.sum(C_speed + C_roll_rate + C_yaw_rate + C_steer_rate ))


            print('J=', round(K_angles*np.sum(C_yaw + C_roll + C_steer),5),('(angles)+'),
                  round( K_angle_rates*np.sum(C_speed + C_roll_rate + C_yaw_rate + C_steer_rate),5), '(angles rates)')


            return model.interval*J

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
            d = sol_dict(prob, free, str_as_keys=True)

            grad = np.zeros_like(free)


            grad[2*NUM_NODES:3*NUM_NODES] = 2.0*model.interval*K_angles*(d['model_q3'] - model.x_meas_dict['yaw_angle_q3'])
            grad[3*NUM_NODES:4*NUM_NODES] = 2.0*model.interval*K_angles*(d['model_q4'] - model.x_meas_dict['roll_angle_q4'])
            grad[6*NUM_NODES:7*NUM_NODES] = 2.0*model.interval*K_angles*(d['model_q7'] - model.x_meas_dict['steer_angle_q7'])
            # grad[4*NUM_NODES:5*NUM_NODES] = 2.0*model.interval*K_angles*(d['model_q5'] - model.x_meas_dict['pitch_angle_q5'])

            grad[11*NUM_NODES:12*NUM_NODES] = 2.0*model.interval*K_angle_rates*(d['model_u4'] - model.x_meas_dict['roll_rate_u4'])
            grad[10*NUM_NODES:11*NUM_NODES] = 2.0*model.interval*K_angle_rates*(d['model_u3'] - model.x_meas_dict['yaw_rate_u3'])
            grad[14*NUM_NODES:15*NUM_NODES] = 2.0*model.interval*K_angle_rates*(d['model_u7'] - model.x_meas_dict['steer_rate_u7'])
            # grad[12*NUM_NODES:13*NUM_NODES] = 2.0*model.interval*K_angles*(d['model_u5'] - model.x_meas_dict['pitch_rate_u5'])

            grad[8*NUM_NODES:9*NUM_NODES] = 2.0*model.interval*K_angle_rates*(d['model_u1'] - model.x_meas_dict['speed'])



            return grad

        q1, q2, q3, q4, q5, q6, q7, q8 = model.x[:8]
        u1, u2, u3, u4, u5, u6, u7, u8 = model.x[8:]

        if model.config['steer_torque'] == False and model.config['roll_control'] == False :

            T_ped = model.r

        if model.config['steer_torque'] == True and model.config['roll_control'] == True :

            T_ped, T_steer, M_x, F_y, F_z = model.r

        if model.config['steer_torque'] == True and model.config['roll_control'] == False :

            T_ped, T_steer = model.r


        if model.config['steer_torque'] == False and model.config['roll_control'] == True :

            T_ped, M_x, F_y, F_z = model.r



        u1_mean = np.mean(model.x_meas_dict['speed'])
        param_str = {str(param) : value for param, value in zip(model.p.keys(),model.p.values())}

        wheel_radius = param_str['rear_wheel_r']


        q3_std = np.std(model.x_meas_dict['yaw_angle_q3'])
        q4_std = np.std(model.x_meas_dict['roll_angle_q4'])
        q7_std = np.std(model.x_meas_dict['steer_angle_q7'])
        # q5_std = np.std(model.x_meas_dict['pitch_angle_q5'])

        # q5_mean = np.mean(model.x_meas_dict['pitch_angle_q5'])


        u3_std = np.std(model.x_meas_dict['yaw_rate_u3'])
        u4_std = np.std(model.x_meas_dict['roll_rate_u4'])
        u7_std = np.std(model.x_meas_dict['steer_rate_u7'])
        # u5_std = np.std(model.x_meas_dict['pitch_rate_u5'])

        N_std = 3


        bounds = {
         q1: (-0.2,  u1_mean*model.DURATION*1.2),
         q2: (-4, 4),
         q3: (-N_std*q3_std, N_std*q3_std),  # bicycle yaw
         q4: (-N_std*q4_std, N_std*q4_std),    # bicycle roll
         # q5: (-N_std*q5_std + q5_mean, q5_mean + N_std*q5_std), # bicycle pitch
         q6: (-1.2*u1_mean*model.DURATION/wheel_radius, 0), # wheel angle
         q7: (-N_std*q7_std, N_std*q7_std),    # steering angle
         q8: (-1.2*u1_mean*model.DURATION/wheel_radius, 0), #wheel angle

         u1: (0.0, 10.0), #longitudinal speed
         u2: (-5.0, 5.0), #lateral speed
         u3: (-N_std*u3_std, N_std*u3_std), #yaw angular rate
         u4: (-N_std*u4_std, N_std*u4_std), #roll angular rate
         # u5: (-N_std*u5_std, N_std*u5_std), #pitch angular rate
         # u6: (-20.0, 0.0), #wheel angular rate
         u7: (-N_std*u7_std, N_std*u7_std), #steer angular rate
         # u8: (-20.0, 0.0), #wheel angular rate

         T_ped : (-20, 20),
        }


        if model.config['steer_torque'] == True:
            bounds[T_steer] = (-25, 25)

        if model.config['roll_control'] == True:
            bounds[M_x] = (-100, 100)
            bounds[F_y] = (-100, 100)
            bounds[F_z] = (-100, 100)



        self.problem = Problem(
            obj,
            obj_grad,
            model.eoms,
            model.x,
            model.NUM_NODES,
            model.interval,
            known_parameter_map = model.p,
            # instance_constraints=data.constraints.instance_constraints,
            bounds = bounds,
            time_symbol = me.dynamicsymbols._t,
            # integration_method = 'midpoint'
            )

        max_item = 10000

        self.problem.add_option('max_iter' , max_item)

        if x0 is None:

            x0 = np.zeros((2*8 + len(model.r), NUM_NODES)).flatten()

            # x0[2*NUM_NODES:3*NUM_NODES] = model.x_meas_dict['yaw_angle_q3']
            # x0[3*NUM_NODES:4*NUM_NODES] = model.x_meas_dict['roll_angle_q4']
            # x0[6*NUM_NODES:7*NUM_NODES] = model.x_meas_dict['steer_angle_q7']

            # x0[11*NUM_NODES:12*NUM_NODES] = model.x_meas_dict['roll_rate_u4']
            # x0[10*NUM_NODES:11*NUM_NODES] = model.x_meas_dict['yaw_rate_u3']
            # x0[14*NUM_NODES:15*NUM_NODES] = model.x_meas_dict['steer_rate_u7']

            # x0[10*NUM_NODES:11*NUM_NODES] = x_meas_dict['speed']
            # x0[13*NUM_NODES:14*NUM_NODES] = - x_meas_dict['speed'] /self.data.constants.get(Symbol('rear_wheel_r'))
            # x0[17*NUM_NODES:18*NUM_NODES] = - x_meas_dict['speed'] /self.data.constants.get(Symbol('front_wheel_r'))


            # x0[18*NUM_NODES:19*NUM_NODES] = x_meas_dict['Acc_x_H']
            # x0[19*NUM_NODES:20*NUM_NODES] = x_meas_dict['Acc_y_H']
            # x0[20*NUM_NODES:21*NUM_NODES] = x_meas_dict['Acc_z_H']

            # x0[21*NUM_NODES:22*NUM_NODES] = x_meas_dict['Gyr_x_H']
            # x0[22*NUM_NODES:23*NUM_NODES] = x_meas_dict['Gyr_y_H']
            # x0[23*NUM_NODES:24*NUM_NODES] = x_meas_dict['Gyr_z_H']






            # x0[9*NUM_NODES:10*NUM_NODES] = model.x_meas_dict['roll_rate_u4']
            # x0[15*NUM_NODES:16*NUM_NODES] = model.x_meas_dict['yaw_rate_u3']
            # x0[11*NUM_NODES:12*NUM_NODES] = model.x_meas_dict['steer_rate_u7']
            # x0[12*NUM_NODES:13*NUM_NODES] = model.x_meas_dict['lean_rate']


            # x0[8*NUM_NODES:9*NUM_NODES] = model.x_meas_dict['speed']
            # x0[11*NUM_NODES:12*NUM_NODES] = -model.x_meas_dict['speed'] /model.data.constants.get(Symbol('rear_wheel_r'))
            # x0[17*NUM_NODES:18*NUM_NODES] = -model.x_meas_dict['speed'] /model.data.constants.get(Symbol('front_wheel_r'))


            # x0[20*NUM_NODES:21*NUM_NODES] = x_meas_dict['Acc_x_H']
            # x0[21*NUM_NODES:22*NUM_NODES] = x_meas_dict['Acc_y_H']
            # x0[22*NUM_NODES:23*NUM_NODES] = x_meas_dict['Acc_z_H']

            # x0[23*NUM_NODES:24*NUM_NODES] = x_meas_dict['Gyr_x_H']
            # x0[24*NUM_NODES:25*NUM_NODES] = x_meas_dict['Gyr_y_H']
            # x0[25*NUM_NODES:26*NUM_NODES] = x_meas_dict['Gyr_z_H']




        self.initial_guess = x0  # u


        self.problem_to_save = {'eoms' : model.eoms,
                                'x' : model.x,
                                'num_nodes' : model.NUM_NODES,
                                'max_item' : max_item,
                                'x0' : x0
                                }

        #idee initialiser avec les derives des vitesses


    def solve_and_save(self):

        self.solution, self.info = self.problem.solve(self.initial_guess)
        model.solution = self.solution
        self.time_simu = np.linspace(0, model.DURATION, num = model.NUM_NODES)



        if not os.path.exists(f"results/{model.date}_part_{model.n_part}_trial_{model.n_trial}"):
            os.makedirs(f"results/{model.date}_part_{model.n_part}_trial_{model.n_trial}")

        try:
            self.problem.plot_constraint_violations(self.solution)
        except:
            plt.savefig(f'results/{model.date}_part_{model.n_part}_trial_{model.n_trial}/constraints_violation.png')
            plt.close()



        #             'metadata' : model.data,
        #               'x_meas_dict' : model.x_meas_dict,
        #               'initial_guess' : self.initial_guess,
        dict_saved = { 'solution' : self.solution,
                      'x' : str(list(model.x)),
                      'inputs' : str(list(model.r)),
                      # 'info' : self.info,
                      'n_part': model.n_part,
                      'n_trial': model.n_trial,
                      'time_simu' : self.time_simu,}
                      # 'problem' : self.problem_to_save}

        # try:
        if not os.path.exists(f"results/{model.date}_part_{model.n_part}_trial_{model.n_trial}"):
            os.makedirs(f"results/{model.date}_part_{model.n_part}_trial_{model.n_trial}")

        dict_saved_file = open(f'results/{model.date}_part_{model.n_part}_trial_{model.n_trial}/{model.date}_part_{model.n_part}_trial_{model.n_trial}.pkl', 'wb')
        pickle.dump(dict_saved, dict_saved_file)
        dict_saved_file.close()

        # except:
        #     print("Something went wrong when saving dict_saved_file")

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
        u_opti = self.solution[8*self.NUM_NODES:(8+1)*self.NUM_NODES]
        u_exp = model.x_meas_dict['speed']
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
        psi_exp = np.rad2deg(model.x_meas_dict['yaw_angle_q3'])
        RMSE_psi = round(RMSE(psi_opti, psi_exp),3)
        axs[1,0].plot(self.time_simu, psi_opti, color = color[0] ,label = '$\psi_{opt}$'+f'- RMSE: {RMSE_psi}')
        axs[1,0].plot(self.time_simu, psi_exp, ls = '--',label = '$\psi_{meas}$', color = color[0])

        #Roll angle
        phi_opti = np.rad2deg(self.solution[3*self.NUM_NODES:(3+1)*self.NUM_NODES])
        phi_exp = np.rad2deg(model.x_meas_dict['roll_angle_q4'])
        RMSE_phi = round(RMSE(phi_opti, phi_exp),3)

        axs[1,0].plot(self.time_simu, phi_opti, color = color[1], label = '$\phi_{opt}$'+ f'- RMSE: {RMSE_phi}')
        axs[1,0].plot(self.time_simu, phi_exp, ls = '--',label = '$\phi_{meas}$', color = color[1])

        #Steer angle
        delta_opti = np.rad2deg(self.solution[6*self.NUM_NODES:(6+1)*self.NUM_NODES])
        delta_exp = np.rad2deg(model.x_meas_dict['steer_angle_q7'])
        RMSE_delta = round(RMSE(delta_opti, delta_exp),3)
        axs[1,0].plot(self.time_simu, delta_opti, color = color[2], label = '$\delta_{opt}$'+f'- RMSE: {RMSE_delta}')
        axs[1,0].plot(self.time_simu, delta_exp, ls = '--',label = '$\delta_{meas}$', color = color[2])

        # Lean angle: theta
        # theta_opti = np.rad2deg(self.solution[4*self.NUM_NODES:(4+1)*self.NUM_NODES])
        # theta_exp = np.rad2deg(model.x_meas_dict['pitch_angle_q5'])
        # RMSE_theta = round(RMSE(theta_opti, theta_exp),3)


        # axs[1,0].plot(self.time_simu, theta_opti, color = color[3], label = '$\\theta_{opt}$' + f'- RMSE: {RMSE_theta}')
        # axs[1,0].plot(self.time_simu, theta_exp, color = color[3], label = '$\\theta_{meas}$', ls = '--')


        axs[1,0].set_xlim(self.time_simu[0], self.time_simu[-1])
        axs[1,0].set_title('Angles')
        axs[1,0].set_ylabel('Angles [deg]')
        axs[1,0].set_xlabel('time [s]')
        axs[1,0].legend(bbox_to_anchor=(1.01, 1.05))


        ## Angle Rates

        #Yaw angle rate
        psi_dot_opti = np.rad2deg(self.solution[10*self.NUM_NODES:(10+1)*self.NUM_NODES])
        psi_dot_exp = np.rad2deg(model.x_meas_dict['yaw_rate_u3'])
        RMSE_psi_dot = round(RMSE(psi_dot_opti, psi_dot_exp),3)
        # RMSE_psi_dot = 0

        axs[2,0].plot(self.time_simu, psi_dot_opti, color = color[0] ,label = '$\dot{\psi_{opt}}$'+f'- RMSE: {RMSE_psi_dot}')
        axs[2,0].plot(self.time_simu, psi_dot_exp, ls = '--',label = '$\dot{\psi_{meas}}$', color = color[0])

        #Roll angle rate
        phi_dot_opti = np.rad2deg(self.solution[11*self.NUM_NODES:(11+1)*self.NUM_NODES])
        phi_dot_exp = np.rad2deg(model.x_meas_dict['roll_rate_u4'])
        RMSE_phi_dot = round(RMSE(phi_dot_opti, phi_dot_exp),3)

        # RMSE_phi_dot = 0

        axs[2,0].plot(self.time_simu, phi_dot_opti, color = color[1], label = '$\dot{\phi_{opt}}$'+ f'- RMSE: {RMSE_phi_dot}')
        axs[2,0].plot(self.time_simu, phi_dot_exp, ls = '--',label = '$\dot{\phi_{meas}}$', color = color[1])

        #Steer angle rate
        delta_dot_opti = np.rad2deg(self.solution[14*self.NUM_NODES:(14+1)*self.NUM_NODES])
        delta_dot_exp = np.rad2deg(model.x_meas_dict['steer_rate_u7'])
        RMSE_dot_delta = round(RMSE(delta_opti, delta_exp),3)
        # RMSE_dot_delta = 0


        axs[2,0].plot(self.time_simu, delta_dot_opti, color = color[2], label = '$\dot{\delta_{opt}}$'+f'- RMSE: {RMSE_dot_delta}')
        axs[2,0].plot(self.time_simu, delta_dot_exp, ls = '--',label = '$\dot{\delta_{meas}}$', color = color[2])

        # Lean angle: theta
        # theta_dot_opti = np.rad2deg(self.solution[12*self.NUM_NODES:(12+1)*self.NUM_NODES])
        # theta_dot_exp = np.rad2deg(model.x_meas_dict['pitch_rate_u5'])

        # RMSE_dot_theta = round(RMSE(theta_dot_opti, theta_dot_exp),3)

        # axs[2,0].plot(self.time_simu, theta_dot_opti, color = color[3], label = '$\dot{\\theta_{opt}}$')
        # axs[2,0].plot(self.time_simu, theta_dot_exp, ls = '--', label = '$\dot{\\theta_{meas}}$' +f'- RMSE: {RMSE_dot_theta}', color = color[3])


        axs[2,0].set_xlim(self.time_simu[0], self.time_simu[-1])
        axs[2,0].set_title('Angles rates')
        axs[2,0].set_ylabel('Angles rates [deg/s]')
        axs[2,0].set_xlabel('time [s]')
        axs[2,0].legend(bbox_to_anchor=(1.01, 1.05))


        #Torques

        # Longitudinal motion control action
        axs[2,1].plot(self.time_simu, self.solution[16*self.NUM_NODES:(16+1)*self.NUM_NODES], label = '$T_{pedal}$', color = color[0])
        axs[2,1].set_xlim(self.time_simu[0], self.time_simu[-1])
        axs[2,1].set_title('Longitudinal motion torques')
        axs[2,1].set_xlabel('time [s]')
        axs[2,1].set_ylabel('Torque [Nm]')
        axs[2,1].legend(bbox_to_anchor=(1.01, 1.05))

        #Balance control actions

        if len(model.r) == 3:

            axs[1,1].plot(self.time_simu, self.solution[17*self.NUM_NODES:(17+1)*self.NUM_NODES], label = '$T_{steer}$', color = color[0])
            axs[1,1].plot(self.time_simu, self.solution[18*self.NUM_NODES:(18+1)*self.NUM_NODES], label = '$T_{roll}$', color = color[1])

            axs[1,1].set_xlim(self.time_simu[0], self.time_simu[-1])
            axs[1,1].set_title('Balance control torques')
            axs[1,1].set_xlabel('time [s]')
            axs[1,1].set_ylabel('Torque [Nm]')
            axs[1,1].legend(bbox_to_anchor=(1.01, 1.05))

        if len(model.r) == 2:

            if config['steer_torque'] == True :

                axs[1,1].plot(self.time_simu, self.solution[17*self.NUM_NODES:(17+1)*self.NUM_NODES], label = '$T_{steer}$', color = color[0])

            if config['roll_torque'] == True :

                axs[1,1].plot(self.time_simu, self.solution[17*self.NUM_NODES:(17+1)*self.NUM_NODES], label = '$T_{roll}$', color = color[0])


            axs[1,1].set_xlim(self.time_simu[0], self.time_simu[-1])
            axs[1,1].set_title('Balance control torques')
            axs[1,1].set_xlabel('time [s]')
            axs[1,1].set_ylabel('Torque [Nm]')
            axs[1,1].legend(bbox_to_anchor=(1.01, 1.05))





        plt.tight_layout()

        if save == 1:
            plt.savefig(f'{path}/{figname}'+'_1_2'+'.svg')
            plt.savefig(f'{path}/{figname}'+'_1_2'+'.png')
            plt.close()


        # fig2, axs2 = plt.subplots(3, 2, figsize=(6*3, 8*3), sharex=True)

        # Acc_x_H_opti = self.solution[18*self.NUM_NODES:(18+1)*self.NUM_NODES]
        # Acc_x_H_exp = self.x_meas_dict['Acc_x_H']
        # RMSE_Acc_x_H = round(RMSE(Acc_x_H_opti, Acc_x_H_exp), 3)
        # axs2[0, 0].plot(self.time_simu, Acc_x_H_opti, color = color[0] ,label = '$Acc_x$'+f'- RMSE: {RMSE_Acc_x_H}')
        # axs2[0, 0].plot(self.time_simu, Acc_x_H_exp, ls = '--',label = '$Accx_{meas}$', color = color[0])

        # Acc_y_H_opti = self.solution[19*self.NUM_NODES:(19+1)*self.NUM_NODES]
        # Acc_y_H_exp = self.x_meas_dict['Acc_y_H']
        # RMSE_Acc_y_H = round(RMSE(Acc_y_H_opti, Acc_y_H_exp), 3)
        # axs2[1, 0].plot(self.time_simu, Acc_y_H_opti, color = color[0] ,label = '$Acc_y$'+f'- RMSE: {RMSE_Acc_y_H}')
        # axs2[1, 0].plot(self.time_simu, Acc_y_H_exp, ls = '--',label = '$Accy_{meas}$', color = color[0])

        # Acc_z_H_opti = self.solution[20*self.NUM_NODES:(20+1)*self.NUM_NODES]
        # Acc_z_H_exp = self.x_meas_dict['Acc_z_H']
        # RMSE_Acc_z_H = round(RMSE(Acc_z_H_opti, Acc_z_H_exp), 3)
        # axs2[2, 0].plot(self.time_simu, Acc_z_H_opti, color = color[0] ,label = '$Acc_z$'+f'- RMSE: {RMSE_Acc_z_H}')
        # axs2[2, 0].plot(self.time_simu, Acc_z_H_exp, ls = '--',label = '$Accz_{meas}$', color = color[0])


        # Gyr_x_H_opti = np.rad2deg(self.solution[21*self.NUM_NODES:(21+1)*self.NUM_NODES])
        # Gyr_x_H_exp = np.rad2deg(self.x_meas_dict['Gyr_x_H'])
        # RMSE_Gyr_x_H = round(RMSE(Gyr_x_H_opti, Gyr_x_H_exp), 3)
        # axs2[0, 1].plot(self.time_simu, Gyr_x_H_opti, color = color[0] ,label = '$Gyr_x$'+f'- RMSE: {RMSE_Gyr_x_H}')
        # axs2[0, 1].plot(self.time_simu, Gyr_x_H_exp, ls = '--',label = '$Gyrx_{meas}$', color = color[0])

        # Gyr_y_H_opti = np.rad2deg(self.solution[22*self.NUM_NODES:(22+1)*self.NUM_NODES])
        # Gyr_y_H_exp = np.rad2deg(self.x_meas_dict['Gyr_y_H'])
        # RMSE_Gyr_y_H = round(RMSE(Gyr_y_H_opti, Gyr_y_H_exp), 3)
        # axs2[1, 1].plot(self.time_simu, Gyr_y_H_opti, color = color[0] ,label = '$Gyr_y$'+f'- RMSE: {RMSE_Gyr_y_H}')
        # axs2[1, 1].plot(self.time_simu, Gyr_y_H_exp, ls = '--',label = '$Gyry_{meas}$', color = color[0])

        # Gyr_z_H_opti = np.rad2deg(self.solution[23*self.NUM_NODES:(23+1)*self.NUM_NODES])
        # Gyr_z_H_exp = np.rad2deg(self.x_meas_dict['Gyr_z_H'])
        # RMSE_Gyr_z_H = round(RMSE(Gyr_z_H_opti, Gyr_z_H_exp), 3)
        # axs2[2, 1].plot(self.time_simu, Gyr_z_H_opti, color = color[0] ,label = '$Gyr_z$'+f'- RMSE: {RMSE_Gyr_z_H}')
        # axs2[2, 1].plot(self.time_simu, Gyr_z_H_exp, ls = '--',label = '$Gyrz_{meas}$', color = color[0])
        # plt.tight_layout()

        # axs2[0,0].legend(bbox_to_anchor=(1.01, 1.05))
        # axs2[1,0].legend(bbox_to_anchor=(1.01, 1.05))
        # axs2[2,0].legend(bbox_to_anchor=(1.01, 1.05))
        # axs2[0,1].legend(bbox_to_anchor=(1.01, 1.05))
        # axs2[1,1].legend(bbox_to_anchor=(1.01, 1.05))
        # axs2[2,1].legend(bbox_to_anchor=(1.01, 1.05))



        if save == 1:
            plt.savefig(f'{path}/{figname}'+'_2_2'+'.svg')
            plt.savefig(f'{path}/{figname}'+'_2_2'+'.png')
            plt.close()


    def plot_results(self, ani=False):


        self.plot_res_type_1(save = 1, path = f"results/{model.date}_part_{model.n_part}_trial_{model.n_trial}", figname = 'results_type_1')
        # problem.plot_objective_value()


        if ani == True:

            ani_name = f'results/{model.date}_part_{model.n_part}_trial_{model.n_trial}/animation'

            sol_reshaped = self.solution.reshape(len(model.x) + len(model.r),-1)

            x_opt = sol_reshaped[:len(model.x),:].T
            r_opt = sol_reshaped[len(model.x):,:].T

            animate_solution(self.time_simu, x_opt, r_opt, model.bicycle, model.x, model.r, model.p, ani_name)






#%% From MOCAP
# PATH = 'C:/Users/ronne/Documents/Recherche/OpenLoopBalanceControl/data/mocap/csv/'

PATH = 'D:/Users/ronne/Documents/4_side_quest/OpenLoopBalanceControl/data/Moore_mocap/csv/'
# run = '2050'
# f'states_{run}'
# Runs = ['3043','3044','3045','3046','3047','3048','3049','3050']
# Runs = ['2043','2044','2045','2046','2047','2048','2049','2050']
Runs = ['2005']

config = {'roll_control' : True,
          'steer_torque' : True,
          'N_sampling' : 1,
          'filtering' : {'use_filtering' : True,
                         'low_cut_freq' : 0.01,
                         'high_cut_freq' : 5},
          'iterative_scaling_factor' : 5
          }


for run in Runs:
    model = OLDC_model(PATH, run, config)

    treadmill_speed = model.treadmill_speed

    #Specific lines of code to compute the speed of the bike on the treadmill
    time = model.df_exp['time']
    x_dot = np.gradient(model.df_exp['1 distance to rear wheel contact'], time)
    y_dot = np.gradient(model.df_exp['2 distance to rear wheel contact'], time)
    psi = model.df_exp['yaw angle']
    model.df_exp['speed'] = (treadmill_speed + x_dot)*np.cos(psi) - y_dot*np.sin(psi)


    model.initialize_model(null_state_solving=True)
    problem = OLDC_solver(model, 1)
    problem.solve_and_save()
    # new_initial_solution = problem.solution
    # problem.plot_res_type_1(0, '', 'fig')



    # model.initialize_model(null_state_solving=False)
    # problem = OLDC_solver(model, 1, new_initial_solution)
    # problem.solve_and_save()

    # new_initial_solution = problem.solution
    # model.initialize_model()
    # problem = OLDC_solver(model, 1)
    # problem.solve_and_save()

# problem.plot_res_type_1(0, '', 'fig')
# problem.plot_results()




