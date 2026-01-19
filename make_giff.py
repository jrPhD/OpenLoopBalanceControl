# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 10:49:47 2026

@author: Jules
"""

import glob
from PIL import Image

path = 'C:/Users/ronne/Documents/Recherche/OpenLoopBalanceControl/results/2026_01_17_19_08_07_part_3_trial_3045_good/animation_frames'

def make_gif(frame_folder, giff_name, n_loop):
    images = glob.glob(f"{frame_folder}/*.png")
    images.sort()
    frames = [Image.open(image) for image in images]
    frame_one = frames[0]
    frame_one.save(f"{giff_name[:-4]}.gif", format="GIF", append_images=frames,
                   save_all=True, duration=5, loop=n_loop)
    print('Giff generated')
    
    
    
    
make_gif(path, 'giff', 4)