#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 16:46:09 2018

@author: pt
"""
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import os
import pandas as pd
import keras
import cv2



data_dir = '/home/pt/Desktop/debug_data'
img_dir = data_dir + '/IMG'
driving_log_csv = data_dir + '/' + 'driving_log.csv'


#driving_log = update_driving_log(data_dir, driving_log_csv, debug = True)

import os
files = os.listdir(data_dir + '/IMG')



from os import listdir
from os.path import isfile, join

driving_log_pd = pd.read_csv(driving_log_csv, header= None)
driving_log_pd.columns = ['center','left', 'right', 'steering','throttle','brake','speed']    

max_index = 0


filenames = listdir(img_dir)

for f in filenames:
    file_name = f.split('.jpg')[0]
    camera = file_name[0]
    file_index = file_name[1:]
    file_index = int(file_index)
    
    if(file_index > max_index):
        max_index = file_index
    
    full_file_path = img_dir + '/' + f
    
    if camera == 'L':
        driving_log_pd.loc[file_index, 'left'] = full_file_path

    elif camera == 'C':
        driving_log_pd.loc[file_index, 'center'] = full_file_path
        
    elif camera == 'R':
        driving_log_pd.loc[file_index, 'right'] = full_file_path       
    
    else:
        print('error')
        break
    
    driving_log_pd.loc[file_index, 'steering'] = file_index
    driving_log_pd.loc[file_index, 'throttle'] = file_index*2
    driving_log_pd.loc[file_index, 'brake'] = file_index*3
    driving_log_pd.loc[file_index, 'speed'] = file_index*4



    
    
driving_log_pd = driving_log_pd[0:max_index]
driving_log_pd.to_csv(driving_log_csv, index = False, header = False)    























