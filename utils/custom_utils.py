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
    if driving_log_pd.iloc[0,0] == 'center':
       driving_log_pd = driving_log_pd.drop([0])
    
    driving_log_pd_temp = driving_log_pd.iloc[:,0:3]    
    driving_log_pd_temp = driving_log_pd_temp.applymap(strip_filenames)    
    driving_log_pd_temp = new_image_path + driving_log_pd_temp.astype(str)    
    driving_log_pd.iloc[:,0:3] = driving_log_pd_temp      
    driving_log_pd.columns = ['center','left', 'right', 'steering','throttle','brake','speed']        
    driving_log_pd.to_csv(driving_log_csv, index = False, header = True)   
    driving_log_pd = driving_log_pd.reset_index()  
    return driving_log_pd



def show_batch(batch, batch_generator_params, save_dir = None, file_name_prefix = None ):
    multi_camera_samples_batch = batch[0]
    multi_camera_labels_batch =  batch[1]    
    total_cameras = len(multi_camera_samples_batch)
    
    index = {'sample':0,'time':1,'height':2,'width':3,'channels':4}        

    if batch_generator_params['time_axis'] == False:
        total_timesteps = 1
       
        
    for camera in range(0,total_cameras):
        samples_batch = multi_camera_samples_batch[camera]        
        volume_shape = samples_batch.shape        
        
        if batch_generator_params['time_axis'] == False:
           assert batch_generator_params['length'] == 1, 'Time axis is False but time series length is not 1' 
           samples_batch = np.expand_dims(samples_batch, axis = 1)   
        else:
           total_timesteps = volume_shape[index['time']]   
        
        total_samples = volume_shape[index['sample']]          

        
        #plt.figure(figsize=figsize) 
        #plt.figure()        


        image_count = 1

        for s, sample in enumerate(samples_batch):    
            for t, timestep in enumerate(sample):        
                plt.subplot(total_samples, total_timesteps, image_count)
                image_count = image_count + 1
                plt.imshow(timestep)
                plt.axis("off")
                #plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            #plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)

            #image_count = 1  
    #plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = .05, wspace = 0)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)

        if not save_dir == None:
            if file_name_prefix == None:
                file_name = save_dir + '/' + '_batch_cam_' + str(camera) + '.jpg'
            else:
                file_name = save_dir + '/' + file_name_prefix + '_batch_cam_' + str(camera) + '.jpg'
                
            plt.axis("off")
            #plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.savefig(file_name, box_inches='tight', pad_inches=0, dpi = 2000)
        plt.show()


    plt.close()

    
    print('\nLabels Batch')
    
    for label in multi_camera_labels_batch:
        print(label)