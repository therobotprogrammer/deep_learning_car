#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 14:24:12 2019

@author: pt
"""

import sys
import math

import numpy as np
import pandas as pd


data_dir = '/media/pt/ramdisk/data_big/data'
code_dir = '/home/pt/Documents/deep_learning_car/utils'

sys.path.insert(0, code_dir) 
import custom_utils as custom_utils

    

driving_log_csv = data_dir + '/' + 'driving_log.csv'
driving_log = custom_utils.update_driving_log(data_dir, driving_log_csv)
driving_log['steering'].hist()

#sorted_targets = np.sort(driving_log['steering'])

#rmin = sorted_targets[0]
#rmax = sorted_targets[-1]
#
#driving_log['random_distribution'] = driving_log['steering'].apply(lambda x: np.random.uniform(rmin,rmax))
#
#
#driving_log['random_distribution'].hist()
#
#sorted_targets = pd.DataFrame
#sorted_targets = driving_log['steering']
##sorted_targets['old_index'] = driving_log['Index']
#sorted_targets['old_index']=driving_log.index.copy()
#
#
#sorted_targets = sorted_targets.sort_values()
#
#
#
#def find_nearest(value):
#    closest_id = np.searchsorted(sorted_targets,value, side = 'left')    
#    return sorted_targets[closest_id]
#
#
##stratified_ids =  find_nearest(driving_log['random_distribution'] )
#
#stratified_ids =  find_nearest(1)
#
#
#driving_log['stratfied_targets'] = stratified_ids
#
#print('Raw Steering Angles')
#driving_log['steering'].hist()
#
#print('Balanced Steering Angles')
#driving_log['stratfied_targets'].hist()
#
#
#

stratified = driving_log.groupby('steering', group_keys=False)
stratified.apply(lambda x: x.sample(stratified.size().min()).reset_index(drop=True))
stratified = stratified.head()

stratified.hist()














