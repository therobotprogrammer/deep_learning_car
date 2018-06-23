#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 19:22:26 2018

@author: pt
"""

from skimage.io import imread
from skimage.transform import resize
import numpy as np
import os
import pandas as pd
import keras

# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.

class DataGenerator(keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([
            resize(imread(file_name), (160, 320))
               for file_name in batch_x]), np.array(batch_y)
    



    
def strip_filenames(old_path):
    old_path = old_path.split("\\") #handles windows generated files
    *directory, filename = old_path        
    directory, filename = os.path.split(filename) #handles linux generated files
    return("/" + filename) 
    

def update_driving_log(data_dir):    
    driving_log_csv = data_dir + '/' + 'driving_log.csv'
    new_image_path = data_dir + '/IMG'
    driving_log_pd = pd.read_csv(driving_log_csv)
    driving_log_pd_temp = driving_log_pd.iloc[:,0:3]    
    driving_log_pd_temp = driving_log_pd_temp.applymap(strip_filenames)
    driving_log_pd_temp = new_image_path + driving_log_pd_temp.astype(str)    
    driving_log_pd.iloc[:,0:3] = driving_log_pd_temp  
    
    driving_log_pd.columns = ['center','left', 'right', 'x','y','z','steering']
    
    driving_log_pd.to_csv(driving_log_csv, index = False, header = False)    
    
    
    return driving_log_pd


data_dir = '/home/pt/Desktop/data'

driving_log = update_driving_log(data_dir)

batch_size = 5
test = DataGenerator(driving_log['center'], driving_log['steering'], batch_size)


iterator = test.__iter__()

batch = next(iterator)

#batch = test.__getitem__(5)


from matplotlib import pyplot as plt

for i in range(0,batch_size):
    plt.imshow(batch[0][i], interpolation='nearest')
    
    plt.show()





















 
