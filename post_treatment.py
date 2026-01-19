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


#%% Hands off 

path = 'C:/Users/ronne/Documents/Recherche/OpenLoopBalanceControl/results/hands_off_mocap/'
runInfo = pd.read_csv('C:/Users/ronne/Documents/Recherche/OpenLoopBalanceControl/data/mocap/runInfo.csv', sep=';')

list_folder = os.listdir(path)
list_solution = []

df_indicators = []

for folder_name in list_folder:
    
    with open(path + f'{folder_name}'+'/'+f'{folder_name}.pkl', 'rb') as file:
        data_loaded = pickle.load(file)
        time_simu = data_loaded['time_simu']
        columns = data_loaded['x'][1:-1].split(',') + data_loaded['inputs'][1:-1].split(',')
        n_trial = data_loaded['n_trial']
        solution = data_loaded['solution'].reshape((len(time_simu), len(columns)))
        rider = n_trial[0]
        solution = pd.DataFrame(solution, columns=columns)
        
        speed = float(runInfo[runInfo['run'] == int(n_trial)]['speed'].iloc[0])
        solution['speed'] = speed
        solution['rider'] = rider
        list_solution.append(solution)
        
        RMSE_T_sls = float(np.sqrt(np.mean((solution[' T_sls(t)']**2))))
        RMSE_T_ext_roll = float(np.sqrt(np.mean((solution['T_ext_roll(t)']**2))))
        std_roll = float(np.std(solution[' q4(t)']))
        std_roll_rate = float(np.std(solution[' u4(t)']))
        std_steer = float(np.std(solution[' q7(t)']))
        std_steer_rate = float(np.std(solution[' u7(t)']))
        
        plt.plot(data_loaded['solution'])

        df_indicators.append([speed, RMSE_T_sls, RMSE_T_ext_roll, rider, std_roll, std_roll_rate, std_steer, std_steer_rate])

data = pd.concat(list_solution)
df_indicators = pd.DataFrame(df_indicators, columns = ['speed', 'RMSE_T_sls', 'RMSE_T_ext_roll', 'rider', 'std_roll', 'str_roll_rate', 'std_steer', 'std_steer_rate'])

# data['T_sls_abs'] = np.abs(data[' T_sls(t)'])
# data['T_ext_roll_abs'] = np.abs(data['T_ext_roll(t)'])

# sns.relplot(df_indicators, x='speed', y='RMSE_T_sls', hue='rider')
# sns.relplot(df_indicators, x='speed', y='RMSE_T_ext_roll', hue='rider')
# sns.relplot(df_indicators, x='speed', y='std_roll', hue='rider')
# sns.relplot(df_indicators, x='speed', y='str_roll_rate', hue='rider')
# sns.relplot(df_indicators, x='speed', y='std_steer', hue='rider')
# sns.relplot(df_indicators, x='speed', y='std_steer_rate', hue='rider')
