# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 14:02:09 2026

@author: Jules
"""
import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt


PATH = 'C:/Users/ronne/Documents/Recherche/OpenLoopBalanceControl/data/mocap/npy/' 

runinfo = pd.read_csv('C:/Users/ronne/Documents/Recherche/OpenLoopBalanceControl/data/mocap/runInfo.csv', sep=';')

sub_trials = runinfo[(runinfo['run']>2000) & (runinfo['condition'] != 'static')]

Runs = sub_trials['run']
data = []

for run in Runs:
    df_trial = np.load(PATH+f'{run}s.npy')
    
    rider = str(sub_trials[sub_trials['run'] == run]['rider'].iloc[0])
    bike = str(sub_trials[sub_trials['run'] == run]['bike'].iloc[0])
    condition = str(sub_trials[sub_trials['run'] == run]['condition'].iloc[0])
    speed = float(sub_trials[sub_trials['run'] == run]['speed'].iloc[0])
    
    r3_16 = df_trial[2,:,15]
    r3_20 = df_trial[2,:,19]
    r2_16 = df_trial[1,:,15]
    r2_20 = df_trial[1,:,19]
    
    d = np.sqrt((r3_20-r3_16)**2 + (r2_20-r2_16)**2)
    d_mean = np.mean(d)
    
    tSteps = df_trial.shape[1]
    t = np.linspace(0, 59.99, tSteps)
    head_roll_angle = np.rad2deg(np.asin((r3_20-r3_16)/d_mean))
    head_roll_angular_velocity = np.gradient(head_roll_angle, t)
    std_roll_angular_velocity = np.std(head_roll_angular_velocity)
    max_roll_angular_velocity = np.max(head_roll_angular_velocity)

    
    data.append([rider, bike, condition, speed, std_roll_angular_velocity, max_roll_angular_velocity])
    
data = pd.DataFrame(data, columns = ['rider', 'bike', 'condition', 'speed', 'std_roll_angular_velocity', 'max_roll_angular_velocity'])

sns.boxplot(data, x='speed', y='std_roll_angular_velocity', hue='condition')
sns.histplot(data, x='std_roll_angular_velocity')

data['S'] = 1


data.to_csv('data_moore.csv')


# plt.plot(t, theta_d )
