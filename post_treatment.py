# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 16:30:38 2026

@author: Jules
"""

import os
import pickle
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

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


#%% Hands off 

path = 'D:/Users/ronne/Documents/4_side_quest/OpenLoopBalanceControl/results/hands_off_mocap/'
runInfo = pd.read_csv('D:/Users/ronne/Documents/4_side_quest/OpenLoopBalanceControl/data/Moore_mocap/runInfo.csv', sep=';')
exp_data_path = 'D:/Users/ronne/Documents/4_side_quest/OpenLoopBalanceControl/data/Moore_mocap/csv/'


list_folder = os.listdir(path)
list_solution = []

df_indicators = []

for folder_name in list_folder:
    
    with open(path + f'{folder_name}'+'/'+f'{folder_name}.pkl', 'rb') as file:
        data_loaded = pickle.load(file)
        n_trial = data_loaded['n_trial']

        
        df_exp = pd.read_csv(exp_data_path + f'states_{n_trial}.csv')
        df_exp_sample = df_exp[(df_exp['time']>15) & (df_exp['time']<45)]

        time_exp = df_exp_sample['time'].to_numpy()[::5]
        yaw_angle_exp = df_exp_sample['yaw angle'].to_numpy()[::5]
        roll_angle_exp = df_exp_sample['roll angle'].to_numpy()[::5]
        steer_angle_exp = df_exp_sample['steer angle'].to_numpy()[::5]
        lean_angle_exp = df_exp_sample['lean angle'].to_numpy()[::5]
        
        yaw_angle_rate_exp = np.gradient(yaw_angle_exp, time_exp)
        roll_angle_rate_exp = np.gradient(roll_angle_exp, time_exp)
        steer_angle_rate_exp = np.gradient(steer_angle_exp, time_exp)
        lean_angle_rate_exp = np.gradient(lean_angle_exp, time_exp)
        
        
        time_simu = data_loaded['time_simu']
        columns = data_loaded['x'][1:-1].split(',') + data_loaded['inputs'][1:-1].split(',')
        solution = data_loaded['solution'].reshape((len(columns), -1))
                
        rider = n_trial[0]
        solution = pd.DataFrame(solution.T, columns=columns)
        
        speed = float(runInfo[runInfo['run'] == int(n_trial)]['speed'].iloc[0])
        solution['speed'] = speed
        solution['rider'] = rider
        list_solution.append(solution)
        
        RMSE_sol_yaw = np.rad2deg(RMSE(solution[' q3(t)'], yaw_angle_exp))
        RMSE_sol_roll = np.rad2deg(RMSE(solution[' q4(t)'], roll_angle_exp))
        RMSE_sol_steer = np.rad2deg(RMSE(solution[' q7(t)'], steer_angle_exp))
        RMSE_sol_lean = np.rad2deg(RMSE(solution[' seat_q(t)'], lean_angle_exp))
        
        RMSE_sol_yaw_rate = np.rad2deg(RMSE(solution[' u3(t)'], yaw_angle_rate_exp))
        RMSE_sol_roll_rate = np.rad2deg(RMSE(solution[' u4(t)'], roll_angle_rate_exp))
        RMSE_sol_steer_rate = np.rad2deg(RMSE(solution[' u7(t)'], steer_angle_rate_exp))
        RMSE_sol_lean_rate = np.rad2deg(RMSE(solution[' seat_u(t)'], lean_angle_rate_exp))


        fig, axs = plt.subplots(4, 2)
        signals_exp = [yaw_angle_exp, roll_angle_exp, steer_angle_exp, lean_angle_exp,
                       yaw_angle_rate_exp, roll_angle_rate_exp, steer_angle_rate_exp, lean_angle_rate_exp]
        signals_sol = [' q3(t)',' q4(t)', ' q7(t)',' seat_q(t)',
                       ' u3(t)', ' u4(t)', ' u7(t)', ' seat_u(t)']
        
        fig.suptitle(f'run {n_trial}')
  
        axs[0,0].plot(time_exp, np.rad2deg(yaw_angle_exp), ls='--', color='blue')
        axs[0,0].plot(time_exp, np.rad2deg(solution[' q3(t)']), color='blue')
        
        axs[0,1].plot(time_exp, np.rad2deg(yaw_angle_rate_exp), ls='--', color='red')
        axs[0,1].plot(time_exp, np.rad2deg(solution[' u3(t)']), color='red')
        
          
        axs[1,0].plot(time_exp, np.rad2deg(roll_angle_exp), ls='--', color='blue')
        axs[1,0].plot(time_exp, np.rad2deg(solution[' q4(t)']), color='blue')
        
        axs[1,1].plot(time_exp, np.rad2deg(roll_angle_rate_exp), ls='--', color='red')
        axs[1,1].plot(time_exp, np.rad2deg(solution[' u4(t)']), color='red')
        
        axs[2,0].plot(time_exp, np.rad2deg(steer_angle_exp), ls='--', color='blue')
        axs[2,0].plot(time_exp, np.rad2deg(solution[' q7(t)']), color='blue')
        
        axs[2,1].plot(time_exp, np.rad2deg(steer_angle_rate_exp), ls='--', color='red')
        axs[2,1].plot(time_exp, np.rad2deg(solution[' u7(t)']), color='red')
        
          
        axs[3,0].plot(time_exp, np.rad2deg(lean_angle_exp), ls='--', color='blue')
        axs[3,0].plot(time_exp, np.rad2deg(solution[' seat_q(t)']), color='blue')
        
        axs[3,1].plot(time_exp, np.rad2deg(lean_angle_rate_exp), ls='--', color='red')
        axs[3,1].plot(time_exp, np.rad2deg(solution[' seat_u(t)']), color='red')
            
            
        # ax[0].set_xlabel('Time [s]')
        
        #     if k<5:
        #         ax.set_ylabel(f'{signal_sol} [deg]')
        #     else:
        #         ax.set_ylabel(f'{signal_sol} [deg/s]')



        
        
        
        RMS_T_sls = float(np.sqrt(np.mean((solution[' T_sls(t)']**2))))
        RMS_T_ext_roll = float(np.sqrt(np.mean((solution['T_ext_roll(t)']**2))))
        RMS_T_ped = float(np.sqrt(np.mean((solution[' T_ped(t)']**2))))

        # std_roll = float(np.std(solution[' q4(t)']))
        # std_roll_rate = float(np.std(solution[' u4(t)']))
        # std_steer = float(np.std(solution[' q7(t)']))
        # std_steer_rate = float(np.std(solution[' u7(t)']))
        
        # plt.plot(data_loaded['solution'])

        df_indicators.append([speed,
                              rider,
                              n_trial,
                              RMSE_sol_yaw,
                              RMSE_sol_roll,
                              RMSE_sol_steer,
                              RMSE_sol_lean,
                              RMSE_sol_yaw_rate,
                              RMSE_sol_roll_rate,
                              RMSE_sol_steer_rate,
                              RMSE_sol_lean_rate,
                              RMS_T_sls,
                              RMS_T_ext_roll,
                              RMS_T_ped])

data = pd.concat(list_solution)
df_indicators = pd.DataFrame(df_indicators, columns = ['speed',
                      'rider',
                      'run',
                      'RMSE_sol_yaw',
                      'RMSE_sol_roll',
                      'RMSE_sol_steer',
                      'RMSE_sol_lean',
                      'RMSE_sol_yaw_rate',
                      'RMSE_sol_roll_rate',
                      'RMSE_sol_steer_rate',
                      'RMSE_sol_lean_rate',
                      'RMS_T_sls',
                      'RMS_T_ext_roll',
                      'RMS_T_ped'])

metrics = ['RMSE_sol_yaw',
'RMSE_sol_roll',
'RMSE_sol_steer',
'RMSE_sol_lean',
'RMSE_sol_yaw_rate',
'RMSE_sol_roll_rate',
'RMSE_sol_steer_rate',
'RMSE_sol_lean_rate']

for metric in metrics:
    
    mean = df_indicators[metric].mean()
    std = df_indicators[metric].std()
    
    print(metric, 'mean=',round(mean,3),'std=', round(std,3))




cols_to_extract = df_indicators.columns[-3:]  # RMS_T_sls, RMS_T_ext_roll, RMS_T_ped

df_new = df_indicators[cols_to_extract].melt(var_name='metric', value_name='value')

sns.boxplot(df_new, x='metric', y='value')




