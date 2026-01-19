# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 12:16:57 2026

@author: Jules
"""

import numpy as np
import pandas as pd

imported_runInfo = pd.read_csv('C:/Users/ronne/Documents/Recherche/OpenLoopBalanceControl/data/mocap/runInfo.csv', sep=';')
PATH = 'C:/Users/ronne/Documents/Recherche/OpenLoopBalanceControl/data/mocap/npy/states/'

qName = ['1 distance to rear wheel contact', '2 distance to rear wheel' +
' contact', 'yaw angle', 'roll angle', 'pitch angle', 'steer angle',
'1 distance to front wheel contact', '2 distance to front wheel contact',
'crank rotation', 'right knee lateral distance', 'left knee lateral distance',
'butt lateral distance', 'lean angle', 'twist angle']

r_names = ['right foot',
'right ankle',
'right knee',
'right hip',
'left foot',
'left ankle',
'left knee',
'left hip',
'butt',
'lower spine',
'shoulder blades',
'neck (helmet)',
'right wrist',
'right elbow',
'right shoulder',
'right temple (helmet)',
'left wrist',
'left elbow',
'left shoulder',
'left temple (helmet)',
'right front wheel',
'right head tube',
'right handlebar',
'right seat tube',
'right rear wheel',
'seat post',
'left front wheel',
'left head tube',
'left handlebar',
'left seat stay',
'left rear wheel',
'front wheel center', 'headtube center', 'handlebar', 'seat stay' +
' center', 'rear wheel center', 'bottom bracket', 'handlebar steer axis',
'rear wheel contact point', 'front wheel contact']

for run in imported_runInfo[imported_runInfo['run']>2000]['run']:
    
    state_file_name = f'states{str(run)}q.npy'
    state_file_array = np.load(PATH + state_file_name)
    tSteps = state_file_array.shape[1]
    t = np.linspace(0, 59.99, tSteps)
    df_state = pd.DataFrame(state_file_array.T, columns = qName)
    df_state['time'] = t
    
    df_state.to_csv(f'states_{str(run)}.csv')
    print('CSV file of',run,'saved')



