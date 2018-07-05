#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 09:37:18 2018

@author: pt
"""

from skimage.io import imread
from skimage.transform import resize
import numpy as np
import os
import pandas as pd
import keras
import cv2
from matplotlib import pyplot as plt


     
    
def strip_filenames(old_path):
    old_path = old_path.split("\\") #handles windows generated files
    *directory, filename = old_path        
    directory, filename = os.path.split(filename) #handles linux generated files
    return("/" + filename) 
    


def update_driving_log(data_dir, driving_log_csv = None, relative_path = False):  
    if driving_log_csv == None:
        driving_log_csv = data_dir + '/' + 'driving_log.csv'
        
    new_image_path = ''
    if relative_path == False:
        new_image_path = data_dir + '/IMG'
    
    driving_log_pd = pd.read_csv(driving_log_csv, header= None)
    driving_log_pd_temp = driving_log_pd.iloc[:,0:3]    
    driving_log_pd_temp = driving_log_pd_temp.applymap(strip_filenames)    
    driving_log_pd_temp = new_image_path + driving_log_pd_temp.astype(str)    
    driving_log_pd.iloc[:,0:3] = driving_log_pd_temp          
    driving_log_pd.to_csv(driving_log_csv, index = False, header = False)    
    driving_log_pd.columns = ['center','left', 'right', 'x','y','z','steering']       

    return driving_log_pd



def show_batch(batch, figsize=(15, 3)):
    multi_camera_samples_batch = batch[0]
    multi_camera_labels_batch =  batch[1]    
    total_cameras = len(multi_camera_samples_batch)
    
    for camera in range(0,total_cameras):
        samples_batch = multi_camera_samples_batch[camera]        
        volume_shape = samples_batch.shape
        
        index = {'sample':0,'time':1,'height':2,'width':3,'channels':4}        
        
        total_samples = volume_shape[index['sample']] 
        total_timesteps = volume_shape[index['time']]        
        
        plt.figure(figsize=figsize)        
        image_count = 1
        
        for s, sample in enumerate(samples_batch):    
            for t, timestep in enumerate(sample):        
                plt.subplot(total_samples, total_timesteps, image_count)
                image_count = image_count + 1
                plt.imshow(timestep)
                plt.axis('off')
        plt.show()

    
    print('\nLabels Batch')
    
    for label in multi_camera_labels_batch:
        print(label)