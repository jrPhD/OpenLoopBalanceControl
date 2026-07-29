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
        self.null_state_solving = null_state_solving

        # not use this line for general use
        df_trial = self.df_exp[(self.df_exp['time']>15) & (self.df_exp['time']<17)]

        fs = np.mean(1/np.diff(df_trial['time'].to_numpy()))
        fs_adjusted = fs/self.N_sampling

        time = df_trial['time'].to_numpy()[::self.N_sampling]

        u = df_trial['speed'].to_numpy()[::self.N_sampling]


        if null_state_solving == True: #Sloving the problem for null state
            x_meas_dict  = {'time' : time,
                            'yaw_angle_q3' :     np.zeros(len(time)),
                            'roll_angle_q4' :     np.zeros(len(time)),
                            'steer_angle_q7' :    np.zeros(len(time)),
                            # 'pitch_angle_q5' :    theta,
                            'speed' :   np.mean(u)*np.ones(len(time)),
                            'roll_rate_u4' :      np.zeros(len(time)),
                            'yaw_rate_u3' :       np.zeros(len(time)),
                            # 'pitch_rate_u5' :     theta_dot,
                            'steer_rate_u7' :     np.zeros(len(time)),
                            # 'roll_acc' :        np.zeros(len(df_trial)),
                            # 'steer_acc' :       np.zeros(len(df_trial)),
                           } #Include here the signals used to feed the opti

            self.config['steer_torque'] = False
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
        self.x0_shape = (len(self.x) + len(self.r), self.NUM_NODES)

        # self.x_animate = np.zeros((18, len(time)))
        # self.x_animate[2,:] = x_meas_dict['yaw_angle_q3']
        # self.x_animate[3,:] = x_meas_dict['roll_angle_q4']
        # # self.x_animate[4,:] = x_meas_dict['pitch_angle_q5']
        # self.x_animate[6,:] = x_meas_dict['steer_angle_q7']
        # self.x_animate[8,:] = x_meas_dict['speed']



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

    def __init__(self, model, scaling_factor, x0=None, gains=None):

        if model.null_state_solving == True:
            K_angles = 1
            K_angle_rates = 1
            K_effort = 1
            K_speed = 1
            
        else:
            
            if gains is not None:
                
                K_angles = gains['K_angles']
                K_angle_rates = gains['K_angles_rates']
                K_effort = gains['K_efforts']
                K_speed = gains['K_speed']
                
                
            else:
            
                K_angles = 1
                K_angle_rates = 0.1
                K_effort = 0.0001
                K_speed = 5
            
        NUM_NODES = model.NUM_NODES
        self.NUM_NODES = NUM_NODES
        self.model = model

        State_var_names = [
            'yaw_angle_q3',
            'roll_angle_q4',
            'steer_angle_q7',
            'roll_rate_u4',
            'yaw_rate_u3',
            'steer_rate_u7']

        for var in State_var_names:
            self.model.x_meas_dict[var] = self.model.x_meas_dict[var]/scaling_factor


        def obj(prob, free):
            """Minimize the error in all of the states."""
            d = sol_dict(prob, free, str_as_keys=True)

            C_yaw = (self.model.x_meas_dict['yaw_angle_q3'] - d['model_q3'])**2
            C_roll = (self.model.x_meas_dict['roll_angle_q4'] - d['model_q4'])**2
            C_steer = (self.model.x_meas_dict['steer_angle_q7'] - d['model_q7'])**2
            # C_pitch = (model.x_meas_dict['pitch_angle_q5'] - d['model_q5'])**2


            C_roll_rate = (self.model.x_meas_dict['roll_rate_u4'] - d['model_u4'])**2
            C_yaw_rate = (self.model.x_meas_dict['yaw_rate_u3'] - d['model_u3'])**2
            C_steer_rate = (self.model.x_meas_dict['steer_rate_u7'] - d['model_u7'])**2
            # C_pitch_rate = (model.x_meas_dict['pitch_rate_u5'] - d['model_u5'])**2

            C_speed = (self.model.x_meas_dict['speed'] - d['model_u1'])**2
            
            C_effort = d['pedaling_torque']**2
            
            if self.model.config['steer_torque'] == True:
                C_effort = C_effort + d['steer_torque']**2
                
            if self.model.config['roll_control'] == True:
                C_effort = C_effort + d['M_x']**2 + d['F_y']**2 + d['F_z']**2
            

            J = (K_angles*np.sum(C_yaw + C_roll + C_steer)
                 + K_speed*np.sum(C_speed)
                 + K_angle_rates*np.sum(C_roll_rate + C_yaw_rate + C_steer_rate )) + K_effort*np.sum(C_effort)

            print('J=', round(K_angles*np.sum(C_yaw + C_roll + C_steer),5),('(angles)+'),
                  round( K_angle_rates*np.sum(C_roll_rate + C_yaw_rate + C_steer_rate),5), '(angles rates)',
                  round(K_effort*np.sum(C_effort), 5), '(torques)',
                  round(K_speed*np.sum(C_speed), 5), '(speed)')


            return self.model.interval*J

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


            grad[2*NUM_NODES:3*NUM_NODES] = 2.0*self.model.interval*K_angles*(d['model_q3'] - self.model.x_meas_dict['yaw_angle_q3'])
            grad[3*NUM_NODES:4*NUM_NODES] = 2.0*self.model.interval*K_angles*(d['model_q4'] - self.model.x_meas_dict['roll_angle_q4'])
            grad[6*NUM_NODES:7*NUM_NODES] = 2.0*self.model.interval*K_angles*(d['model_q7'] - self.model.x_meas_dict['steer_angle_q7'])
            # grad[4*NUM_NODES:5*NUM_NODES] = 2.0*model.interval*K_angles*(d['model_q5'] - model.x_meas_dict['pitch_angle_q5'])

            grad[11*NUM_NODES:12*NUM_NODES] = 2.0*self.model.interval*K_angle_rates*(d['model_u4'] - self.model.x_meas_dict['roll_rate_u4'])
            grad[10*NUM_NODES:11*NUM_NODES] = 2.0*self.model.interval*K_angle_rates*(d['model_u3'] - self.model.x_meas_dict['yaw_rate_u3'])
            grad[14*NUM_NODES:15*NUM_NODES] = 2.0*self.model.interval*K_angle_rates*(d['model_u7'] - self.model.x_meas_dict['steer_rate_u7'])
            # grad[12*NUM_NODES:13*NUM_NODES] = 2.0*model.interval*K_angles*(d['model_u5'] - model.x_meas_dict['pitch_rate_u5'])

            grad[8*NUM_NODES:9*NUM_NODES] = 2.0*self.model.interval*K_speed*(d['model_u1'] - self.model.x_meas_dict['speed'])

            grad[16*NUM_NODES:17*NUM_NODES] = 2.0*self.model.interval*K_effort*d['pedaling_torque']


            if self.model.config['steer_torque'] == True:
                grad[17*NUM_NODES:18*NUM_NODES] = 2.0*self.model.interval*K_effort*d['steer_torque']

            if self.model.config['roll_control'] == True:
                grad[18*NUM_NODES:19*NUM_NODES] = 2.0*self.model.interval*K_effort*d['M_x']
                grad[19*NUM_NODES:20*NUM_NODES] = 2.0*self.model.interval*K_effort*d['F_y']
                grad[20*NUM_NODES:21*NUM_NODES] = 2.0*self.model.interval*K_effort*d['F_z']

            return grad

        q1, q2, q3, q4, q5, q6, q7, q8 = self.model.x[:8]
        u1, u2, u3, u4, u5, u6, u7, u8 = self.model.x[8:]

        if self.model.config['steer_torque'] == False and self.model.config['roll_control'] == False :

            T_ped = self.model.r[0]
            # print(T_ped)

        if self.model.config['steer_torque'] == True and self.model.config['roll_control'] == True :

            T_ped, T_steer, M_x, F_y, F_z = model.r

        if self.model.config['steer_torque'] == True and self.model.config['roll_control'] == False :

            T_ped, T_steer = self.model.r


        if self.model.config['steer_torque'] == False and self.model.config['roll_control'] == True :

            T_ped, M_x, F_y, F_z = self.model.r



        u1_mean = np.mean(model.x_meas_dict['speed'])
        param_str = {str(param) : value for param, value in zip(model.p.keys(),model.p.values())}

        wheel_radius = param_str['rear_wheel_r']
        
        if self.model.null_state_solving == True:
            
            bounds = {
             q1: (-0.2,  u1_mean*self.model.DURATION*1.2),
             q2: (-10, 10),
             q3: (-0.3, 0.3),  # bicycle yaw
             q4: (-0.3, 0.3),    # bicycle roll
             # q5: (-N_std*q5_std + q5_mean, q5_mean + N_std*q5_std), # bicycle pitch
             q6: (-1.2*u1_mean*self.model.DURATION/wheel_radius, 0), # wheel angle
             q7: (-0.3, 0.3),    # steering angle
             q8: (-1.2*u1_mean*self.model.DURATION/wheel_radius, 0), #wheel angle
    
             u1: (u1_mean*0.9, u1_mean*1.1), #longitudinal speed
             u2: (-0.5*u1_mean, 0.5*u1_mean), #lateral speed
             u3: (-0.3, 0.3), #yaw angular rate
             u4: (-0.3, 0.3), #roll angular rate
             # u5: (-N_std*u5_std, N_std*u5_std), #pitch angular rate
             # u6: (-20.0, 0.0), #wheel angular rate
             u7: (-0.3, 0.3), #steer angular rate
             # u8: (-20.0, 0.0), #wheel angular rate
    
             T_ped : (-100, 100),
            }
            

            
        else:


            q3_std = np.std(self.model.x_meas_dict['yaw_angle_q3'])
            q4_std = np.std(self.model.x_meas_dict['roll_angle_q4'])
            q7_std = np.std(self.model.x_meas_dict['steer_angle_q7'])
            # q5_std = np.std(model.x_meas_dict['pitch_angle_q5'])
    
            # q5_mean = np.mean(model.x_meas_dict['pitch_angle_q5'])
    
    
            u3_std = np.std(self.model.x_meas_dict['yaw_rate_u3'])
            u4_std = np.std(self.model.x_meas_dict['roll_rate_u4'])
            u7_std = np.std(self.model.x_meas_dict['steer_rate_u7'])
            # u5_std = np.std(model.x_meas_dict['pitch_rate_u5'])
    
            N_std = 3
    
    
    
            bounds = {
             q1: (0,  u1_mean*self.model.DURATION*1.2),
             q2: (-5, 5),
             q3: (-N_std*q3_std, N_std*q3_std),  # bicycle yaw
             q4: (-N_std*q4_std, N_std*q4_std),    # bicycle roll
             # q5: (-N_std*q5_std + q5_mean, q5_mean + N_std*q5_std), # bicycle pitch
             q6: (-1.2*u1_mean*self.model.DURATION/wheel_radius, 0), # wheel angle
             q7: (-N_std*q7_std, N_std*q7_std),    # steering angle
             q8: (-1.2*u1_mean*self.model.DURATION/wheel_radius, 0), #wheel angle
    
             u1: (0.5*u1_mean, u1_mean*1.5), #longitudinal speed
             u2: (-0.5*u1_mean, 0.5*u1_mean), #lateral speed
             u3: (-N_std*u3_std, N_std*u3_std), #yaw angular rate
             u4: (-N_std*u4_std, N_std*u4_std), #roll angular rate
             # u5: (-N_std*u5_std, N_std*u5_std), #pitch angular rate
             # u6: (-20.0, 0.0), #wheel angular rate
             u7: (-N_std*u7_std, N_std*u7_std), #steer angular rate
             # u8: (-20.0, 0.0), #wheel angular rate
    
             T_ped : (-50, 50),
            }


        if self.model.config['steer_torque'] == True:
            bounds[T_steer] = (-50, 50)
        

        if self.model.config['roll_control'] == True:
            bounds[M_x] = (-50, 50)
            bounds[F_y] = (-50, 50)
            bounds[F_z] = (-50, 50)



        self.problem = Problem(
            obj,
            obj_grad,
            self.model.eoms,
            self.model.x,
            self.model.NUM_NODES,
            self.model.interval,
            known_parameter_map = self.model.p,
            # instance_constraints=data.constraints.instance_constraints,
            bounds = bounds,
            time_symbol = me.dynamicsymbols._t,
            # integration_method = 'midpoint'
            )

        max_item = 100000

        self.problem.add_option('max_iter' , max_item)

        if x0 is None:


            x0 = np.zeros(self.model.x0_shape)
            # u_mean = np.mean(model.df_exp['speed'])
            # x0[0,:] = [np.trapezoid(u_mean*np.ones(i), model.x_meas_dict['time'][:i]) for i in range(model.NUM_NODES)]
            # x0[4,:] =  np.ones(model.NUM_NODES)*0.3
            # x0[5,:] = [np.trapezoid(u_mean*np.ones(i)/0.34, model.x_meas_dict['time'][:i]) for i in range(model.NUM_NODES)]
            # x0[7,:] = x0[5,:]
            
            # x0[8,:] = u_mean*np.ones(model.NUM_NODES)
            # x0[13,:] = u_mean*np.ones(model.NUM_NODES)/0.34
            # x0[15,:] = x0[13,:]

        self.initial_guess = x0.flatten()  # u


        self.problem_to_save = {'eoms' : self.model.eoms,
                                'x' : self.model.x,
                                'num_nodes' : self.model.NUM_NODES,
                                'max_item' : max_item,
                                'x0' : x0
                                }

        #idee initialiser avec les derives des vitesses


    def solve_and_save(self):

        self.solution, self.info = self.problem.solve(self.initial_guess)
        self.model.solution = self.solution
        self.time_simu = np.linspace(0, self.model.DURATION, num = self.model.NUM_NODES)



        if not os.path.exists(f"results/{self.model.date}_part_{self.model.n_part}_trial_{self.model.n_trial}"):
            os.makedirs(f"results/{self.model.date}_part_{self.model.n_part}_trial_{self.model.n_trial}")

        try:
            self.problem.plot_constraint_violations(self.solution)
        except:
            plt.savefig(f'results/{self.model.date}_part_{self.model.n_part}_trial_{self.model.n_trial}/constraints_violation.png')
            plt.close()



        #             'metadata' : model.data,
        #               'x_meas_dict' : model.x_meas_dict,
        #               'initial_guess' : self.initial_guess,
        dict_saved = { 'solution' : self.solution,
                      'x' : str(list(self.model.x)),
                      'inputs' : str(list(self.model.r)),
                      # 'info' : self.info,
                      'n_part': self.model.n_part,
                      'n_trial': self.model.n_trial,
                      'time_simu' : self.time_simu,}
                      # 'problem' : self.problem_to_save}

        # try:
        if not os.path.exists(f"results/{self.model.date}_part_{self.model.n_part}_trial_{self.model.n_trial}"):
            os.makedirs(f"results/{self.model.date}_part_{self.model.n_part}_trial_{self.model.n_trial}")

        dict_saved_file = open(f'results/{self.model.date}_part_{self.model.n_part}_trial_{self.model.n_trial}/{self.model.date}_part_{self.model.n_part}_trial_{self.model.n_trial}.pkl', 'wb')
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
        u_exp = self.model.x_meas_dict['speed']
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
        psi_exp = np.rad2deg(self.model.x_meas_dict['yaw_angle_q3'])
        RMSE_psi = round(RMSE(psi_opti, psi_exp),3)
        axs[1,0].plot(self.time_simu, psi_opti, color = color[0] ,label = '$\psi_{opt}$'+f'- RMSE: {RMSE_psi}')
        axs[1,0].plot(self.time_simu, psi_exp, ls = '--',label = '$\psi_{meas}$', color = color[0])

        #Roll angle
        phi_opti = np.rad2deg(self.solution[3*self.NUM_NODES:(3+1)*self.NUM_NODES])
        phi_exp = np.rad2deg(self.model.x_meas_dict['roll_angle_q4'])
        RMSE_phi = round(RMSE(phi_opti, phi_exp),3)

        axs[1,0].plot(self.time_simu, phi_opti, color = color[1], label = '$\phi_{opt}$'+ f'- RMSE: {RMSE_phi}')
        axs[1,0].plot(self.time_simu, phi_exp, ls = '--',label = '$\phi_{meas}$', color = color[1])

        #Steer angle
        delta_opti = np.rad2deg(self.solution[6*self.NUM_NODES:(6+1)*self.NUM_NODES])
        delta_exp = np.rad2deg(self.model.x_meas_dict['steer_angle_q7'])
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
        psi_dot_exp = np.rad2deg(self.model.x_meas_dict['yaw_rate_u3'])
        RMSE_psi_dot = round(RMSE(psi_dot_opti, psi_dot_exp),3)
        # RMSE_psi_dot = 0

        axs[2,0].plot(self.time_simu, psi_dot_opti, color = color[0] ,label = '$\dot{\psi_{opt}}$'+f'- RMSE: {RMSE_psi_dot}')
        axs[2,0].plot(self.time_simu, psi_dot_exp, ls = '--',label = '$\dot{\psi_{meas}}$', color = color[0])

        #Roll angle rate
        phi_dot_opti = np.rad2deg(self.solution[11*self.NUM_NODES:(11+1)*self.NUM_NODES])
        phi_dot_exp = np.rad2deg(self.model.x_meas_dict['roll_rate_u4'])
        RMSE_phi_dot = round(RMSE(phi_dot_opti, phi_dot_exp),3)

        # RMSE_phi_dot = 0

        axs[2,0].plot(self.time_simu, phi_dot_opti, color = color[1], label = '$\dot{\phi_{opt}}$'+ f'- RMSE: {RMSE_phi_dot}')
        axs[2,0].plot(self.time_simu, phi_dot_exp, ls = '--',label = '$\dot{\phi_{meas}}$', color = color[1])

        #Steer angle rate
        delta_dot_opti = np.rad2deg(self.solution[14*self.NUM_NODES:(14+1)*self.NUM_NODES])
        delta_dot_exp = np.rad2deg(self.model.x_meas_dict['steer_rate_u7'])
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

        if self.model.config['steer_torque'] == True:

            axs[1,1].plot(self.time_simu, self.solution[17*self.NUM_NODES:(17+1)*self.NUM_NODES], label = '$T_{steer}$', color = color[0])
            # axs[1,1].plot(self.time_simu, self.solution[18*self.NUM_NODES:(18+1)*self.NUM_NODES], label = '$T_{roll}$', color = color[1])

            axs[1,1].set_xlim(self.time_simu[0], self.time_simu[-1])
            axs[1,1].set_title('Balance control torques')
            axs[1,1].set_xlabel('time [s]')
            axs[1,1].set_ylabel('Torque [Nm]')
            axs[1,1].legend(bbox_to_anchor=(1.01, 1.05))
            
            
        if self.model.config['roll_control'] == True:


            axs[1,1].plot(self.time_simu, self.solution[17*self.NUM_NODES:(17+1)*self.NUM_NODES], label = '$T_{steer}$', color = color[0])
            axs[1,1].plot(self.time_simu, self.solution[18*self.NUM_NODES:(18+1)*self.NUM_NODES], label = '$M_{x}$', color = color[1])
            axs[1,1].plot(self.time_simu, self.solution[19*self.NUM_NODES:(19+1)*self.NUM_NODES], label = '$F_{y}$', color = color[2])
            axs[1,1].plot(self.time_simu, self.solution[20*self.NUM_NODES:(20+1)*self.NUM_NODES], label = '$F_{z}$', color = color[3])


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


        self.plot_res_type_1(save = 1, path = f"results/{self.model.date}_part_{self.model.n_part}_trial_{self.model.n_trial}", figname = 'results_type_1')
        # problem.plot_objective_value()


        if ani == True:

            ani_name = f'results/{self.model.date}_part_{self.model.n_part}_trial_{self.model.n_trial}/animation'

            sol_reshaped = self.solution.reshape(len(self.model.x) + len(self.model.r),-1)

            x_opt = sol_reshaped[:len(self.model.x),:].T
            r_opt = sol_reshaped[len(self.model.x):,:].T

            animate_solution(self.time_simu, x_opt, r_opt, self.model.bicycle, self.model.x, self.model.r, self.model.p, ani_name)






#%% From MOCAP
# PATH = 'C:/Users/ronne/Documents/Recherche/OpenLoopBalanceControl/data/mocap/csv/'

PATH = 'D:/Users/ronne/Documents/4_side_quest/OpenLoopBalanceControl/data/Moore_mocap/csv/'
# run = '2050'
# f'states_{run}'
# Runs = ['3043','3044','3045','3046','3047','3048','3049','3050']
# Runs = ['2043','2044','2045','2046','2047','2048','2049','2050']
Runs = ['2002']
run = Runs[0]

config = {'roll_control' : False,
          'steer_torque' : True,
          'N_sampling' : 4,
          'filtering' : {'use_filtering' : True,
                         'low_cut_freq' : 0.01,
                         'high_cut_freq' : 5}
          }


from scipy.interpolate import CubicSpline

for k in range(20):
    
    ## Define a fisrt model with full steering and litte points
    
    model_1 = OLDC_model(PATH, run, config)
    treadmill_speed = model_1.treadmill_speed
    
    #Specific lines of code to compute the speed of the bike on the treadmill
    time = model_1.df_exp['time']
    x_dot = np.gradient(model_1.df_exp['1 distance to rear wheel contact'], time)
    y_dot = np.gradient(model_1.df_exp['2 distance to rear wheel contact'], time)
    psi = model_1.df_exp['yaw angle']
    
    model_1.df_exp['speed'] = (treadmill_speed + x_dot)*np.cos(psi) - y_dot*np.sin(psi)
    model_1.initialize_model(null_state_solving=False)
    
    problem_1 = OLDC_solver(model_1, 1)
    problem_1.solve_and_save()
    previous_solution = problem_1.solution.reshape(model_1.x0_shape)
    problem_1.plot_res_type_1(0, '', 'fig')
    
    ## Define a second model with more points
    
    config['N_sampling'] = 2
    model_2 = OLDC_model(PATH, run, config)
    model_2.df_exp['speed'] = (treadmill_speed + x_dot)*np.cos(psi) - y_dot*np.sin(psi)
    model_2.initialize_model(null_state_solving=False)
    
    x0 = np.zeros(model_2.x0_shape)
    
    for i in range(previous_solution.shape[0]):
        x0_i_interpolated = CubicSpline(model_1.x_meas_dict['time'], previous_solution[i, :])
        x0[i, :] = x0_i_interpolated(model_2.x_meas_dict['time'])
    
    problem_2 = OLDC_solver(model_2, 1, x0.flatten())
    problem_2.solve_and_save()
    previous_solution = problem_2.solution.reshape(model_2.x0_shape)
    problem_2.plot_res_type_1(0, '', 'fig')
    
    ## Define a new model that include roll control as well
    config['roll_control'] = True
    model_3 = OLDC_model(PATH, run, config)
    model_3.df_exp['speed'] = (treadmill_speed + x_dot)*np.cos(psi) - y_dot*np.sin(psi)
    model_3.initialize_model(null_state_solving=False)
    
    
    random_gains = np.abs(np.random.normal(1, 50, 4))
    gains_name = ['angles','angles_rates','efforts','speed']
    
    for i, gain in enumerate(gains_name):

    
    x0 = np.zeros(model_3.x0_shape)
    x0[:previous_solution.shape[0], :] = previous_solution
    problem_3 = OLDC_solver(model_3, 1, x0.flatten(), gains=gains)
    problem_3.solve_and_save()
    problem_3.plot_res_type_1(0, '', 'fig')





    # model.initialize_model(null_state_solving=False)
    # problem = OLDC_solver(model, 1, x0)
    # problem.solve_and_save()

    # new_initial_solution = problem.solution
    # model.initialize_model()
    # problem = OLDC_solver(model, 1)
    # problem.solve_and_save()

# problem.plot_res_type_1(0, '', 'fig')
# problem.plot_results()










