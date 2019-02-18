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
import pandas as pd

    
'''
#google_drive = '/home/pt/Desktop/google_drive'
### Model save location ###


save_load_params = {
                            'use_last_model' = True,
                            'create_time_stamped_directories' = True                                
                   }

model_save_top_directory = google_drive + '/deep_learning/01_Self_Driving_Car_Nvidia_Paper/saved_models' 
#model_save_directory = model_save_top_directory 
'''

class model_load_and_save:
    def __init__(self, model_save_top_directory, use_last_model = True, create_time_stamped_dirs = True, local_timezone = 'Asia/Kolkata', inference_mode = False):
        self.model_save_top_directory = model_save_top_directory
        self.use_last_model = use_last_model 
        self.create_time_stamped_dirs = create_time_stamped_dirs
        self.local_timezone = local_timezone
        self.model_loaded_sucessfully = False
        self.initial_epoch = 1
        
        self.setup_paths()
        
        
            

        
        
    def setup_paths(self):
                  
        self.model = None
        
        if not self.use_last_model:         
            if os.path.isdir(self.model_save_top_directory):
                self.deep_cleanup(self.model_save_top_directory)
            else:
                os.makedirs(self.model_save_top_directory)


        model_save_directory = self.model_save_top_directory
        
        
        if (self.use_last_model) == True:
          last_model_dir = self.getlastmodeldir(self.model_save_top_directory) #assumes sub dir names are appended with time strings
          if last_model_dir == None:
            self.use_last_model = False
          else:
            model_save_directory = last_model_dir
            saved_model_filename = self.get_last_model(model_save_directory)
            saved_csv_save_file = self.get_last_file(model_save_directory, file_type = 'csv')

            try: 
                if saved_model_filename != None:                       
                        self.model = load_model(saved_model_filename)  
                        if self.model != None:
                            self.model_loaded_sucessfully = True
                        else:                            
                            self.model_loaded_sucessfully = False
                            raise(OSError) 
                            
                        if saved_csv_save_file != None:
                            log_pd = pd.read_csv(saved_csv_save_file, header = 0)  
                            self.initial_epoch = 1 + int(log_pd['epoch'].iloc[-1])                                                
                            print('>>> Previously saved model found. Training will continue from file: \n' + saved_model_filename) 
                            print('Initial epoch: ' + str(self.initial_epoch))
                        else:
                            print('Saved CSV file not found or was invalid with size = 0 bytes. Creating new csv filename and setting initial_spoch to 1')
                            self.initial_epoch = 1                
    
                else:
                    print('>>> No saved models found.')
                    raise(OSError)  
            except OSError:
                        print('>>> Error loading saved model.')
            
        if self.use_last_model == False:
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
        
        #self.model_save_file = model_save_directory + '/' + 'weights-{epoch:02d}-{val_loss:.2f}.h5'
        #self.csv_save_file = model_save_directory + '/' + 'log-' + str(self.initial_epoch) + '.csv'
        
        #self.csv_save_file = model_save_directory + '/' + 'training_log.csv'
        
        self.model_save_file = os.path.join(model_save_directory, 'weights-{epoch:02d}-{val_loss:.2f}.h5' )
        self.csv_save_file = os.path.join(model_save_directory, 'log.csv')
        self.working_directory = model_save_directory
        self.tensorboard_log_dir = model_save_directory
       
        '''
        if not os.path.isdir(tensorboard_log_dir):
          os.makedirs(tensorboard_log_dir)
        '''
        
   
   
    def deep_cleanup(self, currentDir):
      index = 0
    
      for root, dirs, files in os.walk(currentDir):
        # remove empty files. 
        #Usecase: when keras callback CSVLogger is enabled and training interupted 
        # due to lack or resources on google colab, then before this callback saves anything, 
        # it created an empty .csv file. This causes problems when we resume training and 
        # pandas tries to read this empty file
        
        for file in files:
            file_full_path = os.path.join(root, file)
            if os.path.getsize(file_full_path) == 0:
                try: 
                    os.remove(file_full_path)
                except:
                    print('File size is 0, but unable to remove it')
    
        #remove empty directories
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
      sub_dirs = sorted(sub_dirs)
      if sub_dirs == []:
        return None
      else:
        return sub_dirs[-1]
    
    
    #returns last model. Worls with both h5 and hdf5 
    def get_last_model(self, directory):
        last_model_h5 = self.get_last_file(directory, 'h5')
        last_model_hdf5 = self.get_last_file(directory, 'hdf5')
        
        if (last_model_h5 == None) and (last_model_hdf5 == None):
            return None
        elif (last_model_h5 == None) and (last_model_hdf5 != None):
            return last_model_hdf5
        elif (last_model_h5 != None) and (last_model_hdf5 == None):
            return last_model_h5
        else:
            h5_file_without_extension = last_model_h5.split('.h5')
            hdf5_file_without_extension = last_model_hdf5.split('.hdf5')
            
            if h5_file_without_extension > hdf5_file_without_extension:
                return last_model_h5
            else:
                return last_model_hdf5
            

    
    def get_last_file(self, directory, file_type):    
        
        last_model = glob(directory + '/*.' + file_type) 
        last_model = sorted(last_model)            
        
        if last_model == []:           
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
        
        

    def get_files_of_type(self, directory, file_type):    
        
        files = glob(directory + '/*.' + file_type) 
        files = sorted(files)            
        
        if files == []:           
            return None
        
        return files

    
    def __get_filename_from_epoch_lookup(self,split_char_in_filename = '-'):
        all_files = self.get_files_of_type(self.working_directory, 'h5')
        lookup = {}
        
        for file in all_files:
            directory, filename = os.path.split(file)
            epoch = filename.split(split_char_in_filename)[1]
            lookup[epoch] = file
        return lookup
    
    
    def get_n_best_models(self, N = 1):
        #try:       
        best_epochs_pd = pd.read_csv(self.csv_save_file)
        if N >= best_epochs_pd.shape[0]:
            N = best_epochs_pd.shape[0]  
            
        best_epochs_pd = best_epochs_pd.sort_values('val_loss')
        best_epochs_pd = best_epochs_pd.iloc[0:N,]
        best_epochs_pd = best_epochs_pd.reset_index(drop = True)
        filename_from_epoch_lookup = self.__get_filename_from_epoch_lookup()
        
        best_epochs_pd['model'] = None
        best_epochs_pd['file_name'] = None
        
        for index, epoch in enumerate(best_epochs_pd['epoch']):
            try:
                filename = filename_from_epoch_lookup[str(epoch)]
                best_epochs_pd.loc[[index], 'file_name'] = filename
                best_epochs_pd.loc[[index], 'model'] = load_model(filename)
            except KeyError:
                print('>>>ERROR: Best model is at epoch ' + str(epoch) + ' but the model could not be found')        
        return best_epochs_pd   
    
    
    
    def get_model_at_epoch(self, epoch):
        #try:       
        filename_from_epoch_lookup = self.__get_filename_from_epoch_lookup()
        filename = filename_from_epoch_lookup[str(epoch)]
        
        model = None
        try:
            model = load_model(filename) 
        except KeyError:
            print('>>>ERROR: Model not found at epoch: ' + str(epoch))        
        
        return model
       


    
    
if (__name__) == '__main__':
     
    google_drive = '/home/pt/fake_google_drive'

    
    model_save_top_directory = google_drive + '/deep_learning/01_Self_Driving_Car_Nvidia_Paper/saved_models' 
    #model_save_top_directory = '~/my_deep_learning/01_Self_Driving_Car_Nvidia_Paper/saved_models' 
    
    
    save_load_params = {
                                'use_last_model' : True,
                                'create_time_stamped_dirs' : True                                
                       }
    
    model_saver = model_load_and_save(model_save_top_directory, **save_load_params)
    print('log file: ' + model_saver.csv_save_file)  

    #best_model = model_saver.get_n_best_models(1)    
    
#t = model_load_and_save(model_save_top_directory, **save_load_params)
#t.csv_save_file