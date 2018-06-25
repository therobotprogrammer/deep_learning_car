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
from keras.preprocessing.sequence import TimeseriesGenerator


# Here, `input_data_set` is list of path to the images
# and `labels_set` are the associated classes.




        


class DataGenerator(keras.preprocessing.sequence.TimeseriesGenerator):
    id_list = []
    imagename_list = []

    def __init__(self, input_data_set, labels_set, length=100, sampling_rate=1, stride=1, start_index=0, end_index=None, shuffle=False, reverse=False, batch_size=16, image_dimention = (160,320), n_channels = 3):
        self.input_data_set = input_data_set
        self.labels_set = labels_set
        
        self.image_dimention = image_dimention
        self.n_channels = n_channels
        self.batch_size=16

#The following have not been implemented
        self.length=100
        self.sampling_rate=1
        self.stride=1
        self.start_index=0
        self.end_index=None
        self.shuffle=False
        self.reverse=False

      

    def __len__(self):
        return int(np.ceil(len(self.input_data_set) / float(self.batch_size)))

    def __getitem__(self, idx):
        if self.length == 0:
            batch_input_data_image_array = np.empty((self.batch_size, *self.image_dimention, self.n_channels))     
            
            batch_input_data = self.input_data_set[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_label_data = self.labels_set[idx * self.batch_size:(idx + 1) * self.batch_size]
                
            for i, file_name in enumerate(batch_input_data):
                print(file_name)
                batch_input_data_image_array[i] = cv2.imread(file_name)                    
            return batch_input_data_image_array, np.array(batch_label_data)

        
        
        
        
'''   
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.input_data_set))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
'''
    
        #return np.array([resize(imread(file_name), (160, 320)) for file_name in self.batch_input_data]), np.array(batch_label_data)
    

    
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

params = {
          'image_dimention': (160,320),
          'batch_size': 16,
          'n_channels': 3,
          'shuffle': True,
         }

test = DataGenerator(driving_log['center'], driving_log['steering'], **params) #why **
test.on_epoch_end()

iterator = test.__iter__()

batch = next(iterator)


#batch = test.__getitem__(5)









'''
from matplotlib import pyplot as plt

for i in range(0,batch_size):
    plt.imshow(batch[0][i], interpolation='nearest')
    
    plt.show()




'''
















 
