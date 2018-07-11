#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 15:55:49 2018

@author: pt
"""

import os
from glob import glob
import pytz
import datetime
from keras.models import load_model

    
'''
#google_drive = '/home/pt/Desktop/google_drive'
### Model save location ###


save_load_params = {
                            'continue_training_last_model' = True,
                            'create_time_stamped_directories' = True                                
                   }

model_save_top_directory = google_drive + '/deep_learning/01_Self_Driving_Car_Nvidia_Paper/saved_models' 
#model_save_directory = model_save_top_directory 
'''

class model_load_and_save:
    def __init__(self, model_save_top_directory, continue_training_last_model = True, create_time_stamped_dirs = True, local_timezone = 'Asia/Kolkata'):
        self.model_save_top_directory = model_save_top_directory
        self.continue_training_last_model = continue_training_last_model 
        self.create_time_stamped_dirs = create_time_stamped_dirs
        self.local_timezone = local_timezone
        self.model = None

        self.setup_paths()

        
        
    def setup_paths(self):
                                   
        if os.path.isdir(self.model_save_top_directory):
            self.cleanup(self.model_save_top_directory)
        else:
            os.makedirs(self.model_save_top_directory)


        model_save_directory = self.model_save_top_directory
        
        if self.continue_training_last_model == True:
          last_model_dir = self.getlastmodeldir(self.model_save_top_directory) #assumes sub dir names are appended with time strings
          if last_model_dir == None:
            self.continue_training_last_model == False
          else:
            model_save_directory = last_model_dir
            saved_model_filename = self.get_last_file(model_save_directory, file_type = 'h5')
            if saved_model_filename != None:
                try:    
                    self.model = load_model(saved_model_filename)
                    print('>>> Previously saved model found. Training will continue from file: \n' + saved_model_filename) 
                except OSError:
                    print('>>> Error loading saved model. A new model will be trained')

            else:
                print('>>> No saved models found. A new model will be trained')
                self.continue_training_last_model = False  
            
        if self.continue_training_last_model == False:
            if self.create_time_stamped_dirs:
                model_save_directory =  self.model_save_top_directory + '/' + self.get_time_string()
            else:
                model_save_directory = self.model_save_top_directory + '/default_model'
        
            if os.path.isdir(model_save_directory):
                print('Model directory emptied for new experiment. It was created in the last minute or continue training was false. ')
                os.rmdir(model_save_directory)
                
            os.makedirs(model_save_directory)
            print('Created directory for this experiment: ' + model_save_directory)
            #model_save_file = model_save_directory + '/' + 'model-{epoch:03d}.h5'
        
        self.model_save_file = model_save_directory + '/' + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
        self.csv_save_file = model_save_directory + '/' + 'log.csv'
        self.tensorboard_log_dir = model_save_directory

        
        '''
        if not os.path.isdir(tensorboard_log_dir):
          os.makedirs(tensorboard_log_dir)
        '''
        

      
        
     
   
    def deep_cleanup(self, currentDir):
      index = 0
    
      for root, dirs, files in os.walk(currentDir):
          for dir in dirs:
              newDir = os.path.join(root, dir)
              index += 1
              print (str(index) + " ---> " + newDir)
    
              try:
                  os.removedirs(newDir)
                  print ("Directory empty! Deleting...")
                  print (" ")
              except:
                  print ("Directory not empty and will not be removed")
                  print (" ")
    
    
    
    def cleanup(self, main_dir, show_relative_dir_names = False):
        sub_dirs = glob(main_dir + '/*/')
        
        if sub_dirs == None:
            return
        else:
            for index, directory in enumerate(sub_dirs):
              
              if show_relative_dir_names:  
                  relative_dirname = directory.split('/')[-3:]
                  relative_dirname = '/'.join(relative_dirname)
                  relative_dirname = '/' + relative_dirname
                  #ralative_dirname = os.path.split(ralative_dirname)[0]  
                  print (str(index) + " ---> " + relative_dirname)
              else:
                  print (str(index) + " ---> " + directory)
    
              try:
                  os.removedirs(directory)
                  print()
                  print ("Directory empty! Deleting...")
                  print (" ")
              except:
                  print ("Directory not empty and will not be removed")
                  print (" ") 
    
    
                  
    def getlastmodeldir(self, main_dir):
      sub_dirs = glob(main_dir + '/*/')      
      if sub_dirs == []:
        return None
      else:
        return sub_dirs[-1]
    
    
    
    def get_last_file(self, directory, file_type = 'h5' ):    
        last_model = glob(directory + '/*.' + file_type) 
        
        if last_model == []:
            print('No saved model found')
            return None
        
        return last_model[-1]
    
    
    def get_time_string(self):
        
        utc_time = pytz.utc.localize(datetime.datetime.utcnow())
        local_time = utc_time.astimezone(pytz.timezone(self.local_timezone))  
                
        year = str(local_time.year)  
        month = str(local_time.month) 
        day = str(local_time.day) 
        hour = str(local_time.hour)
        minute = str(local_time.minute)
        
        time_string = year + '-' + month + '-' + day + '-' + hour + '-' + minute
        return(time_string)