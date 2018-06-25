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
import cv2

# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.

class DataGenerator(keras.utils.Sequence):

    def __init__(self, input_data_set, labels_set, batch_size, image_dimention = (160,320), n_channels = 3, shuffle = True):
        self.input_data_set = input_data_set
        self.labels_set = labels_set
        self.batch_size = batch_size
        self.image_dimention = image_dimention
        self.n_channels = n_channels


    def __len__(self):
        return int(np.ceil(len(self.input_data_set) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.input_data_set[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels_set[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        
        batch_input_data_image_array = np.empty((self.batch_size, *self.image_dimention, self.n_channels))     
        
        batch_input_data = self.input_data_set[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_label_data = self.labels_set[idx * self.batch_size:(idx + 1) * self.batch_size]
            
        for i, file_name in enumerate(batch_input_data):
            print((file_name))
            batch_input_data_image_array[i] = cv2.imread(file_name)                    
        return batch_input_data_image_array, np.array(batch_label_data)
        
        
 



    
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

batch_size = 16

params = {
          'image_dimention': (160,320),
          'batch_size': 16,
          'n_channels': 3,
          'shuffle': True,
         }

test = DataGenerator(driving_log['center'], driving_log['steering'], **params)


iterator = test.__iter__()

batch = next(iterator)

#batch = test.__getitem__(5)


'''
from matplotlib import pyplot as plt

for i in range(0,batch_size):
    plt.imshow(batch[0][i], interpolation='nearest')
    
    plt.show()

'''



















 
