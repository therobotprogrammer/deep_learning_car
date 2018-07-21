from skimage.io import imread
from skimage.transform import resize
import numpy as np
import os
import pandas as pd
import keras
import cv2
from matplotlib import pyplot as plt
import random
from keras.preprocessing import image


class MultiSensorTimeSeriesGenerator(keras.utils.Sequence):
#Q: Is is good practive to send dict in a dict?
    def __init__(self, multi_sensor_data, targets, 
                 length,
                 sampling_rate=1,
                 stride=1,
                 start_index=0,
                 end_index=None,
                 shuffle=False,
                 reverse=False,
                 batch_size=16,
                 image_dimention = (160,320),
                 n_channels = 3,
                 time_axis = True,
                 augmentation_parameters =  {
                                                'theta_range':0,                                         
                                                'tx_range':0,                                         
                                                'ty_range':0,                                 
                                                'shear_range':0,
                                                'zx_range':0,
                                                'zy_range':0,
                                                'flip_horizontal':False,
                                                'flip_vertical':False,
                                                'channel_shift_intencity_range': 0,
                                                'brightness_range':0
                                            },
                 seed = 0
                 ):
        
        self.multi_sensor_data = multi_sensor_data
        self.single_sensor_data_sample = multi_sensor_data[0] #change this if different sensors give different dimentions of data
        self.targets = targets
        self.length = length
        self.sampling_rate = sampling_rate
        self.stride = stride
        self.start_index = start_index + length
        if end_index is None:
            end_index = len(self.single_sensor_data_sample) - 1
        self.end_index = end_index
        self.shuffle = shuffle
        self.reverse = reverse
        self.batch_size = batch_size        
        self.image_dimention = image_dimention
        self.n_channels = n_channels
        self.n_sensors = len(self.multi_sensor_data)        
        self.time_axis = time_axis
        self.augmentation_parameters = augmentation_parameters           
        self.seed = seed
        random.seed = self.seed
        
        self.use_augmentation = self.__get_decision_use_augmentation(augmentation_parameters)
        



        if self.start_index > self.end_index:
            raise ValueError('`start_index+length=%i > end_index=%i` '
                             'is disallowed, as no part of the sequence '
                             'would be left to be used as current step.'
                             % (self.start_index, self.end_index))
            

    def __get_decision_use_augmentation(self, augmentation_parameters):
        default_augmentation_params =  {
                                                'theta_range':0,                                         
                                                'tx_range':0,                                         
                                                'ty_range':0,                                 
                                                'shear_range':0,
                                                'zx_range':0,
                                                'zy_range':0,
                                                'flip_horizontal':False,
                                                'flip_vertical':False,
                                                'channel_shift_intencity_range': 0,
                                                'brightness_range':0
                                            }
        if augmentation_parameters == default_augmentation_params:
            return False
        else:
            return True
        
        
    def __len__(self):
        return (self.end_index - self.start_index +
                self.batch_size * self.stride) // (self.batch_size * self.stride)

    def _empty_batch(self, num_rows):
        samples_shape = [num_rows, self.length // self.sampling_rate, *self.image_dimention, self.n_channels]
        samples_shape.extend(self.single_sensor_data_sample.shape[1:])
        targets_shape = [num_rows]
        targets_shape.extend(self.targets.shape[1:]) 
        return np.empty(samples_shape), np.empty(targets_shape)

    def __getitem__(self, index):
        all_sensor_transforms = {}           

        if self.shuffle:
            rows = np.random.randint(
                self.start_index, self.end_index + 1, size=self.batch_size)
        else:
            i = self.start_index + self.batch_size * self.stride * index
            rows = np.arange(i, min(i + self.batch_size * self.stride, self.end_index + 1), self.stride)
            
            
        if self.use_augmentation:
            all_sensor_transforms = self.__get_all_sensor_transforms(rows)
                
        multi_camera_tensor = []
        
        
        for sensor_index in range(0,self.n_sensors):
            samples,targets = self.__getcameraTensor(sensor_index, rows, all_sensor_transforms) # is this bad programming? - to do: code review
            multi_camera_tensor.append(samples)
        

        return multi_camera_tensor, targets
    
    
    def __get_all_sensor_transforms(self, rows):
        all_sensor_transforms = {}
        
        for row in rows:
            all_sensor_transforms[row]  = self.__get_random_sensor_transform()
            
        return all_sensor_transforms
        
        
    #Target transform has to be calculated based on sensor transform. For example, if sensor camera is flipped
    #then steering angle has to be flipped too. 
    def __apply_sensor_target_transform_mapping(self, sensor_transform, target):
        if sensor_transform['flip_horizontal']:
            target = -target
        
        
    def __get_random_sensor_transform(self):
        #random_transform = self.augmentation_parameters
        random_transform = {}
        for key, value in self.augmentation_parameters.items():
            if type(value) == bool:
                if value == True: # we should flip with random probability
                    random_number = random.random()
                    if random_number >.5:
                        random_transform[key] = True
                    else:
                        random_transform[key] = False
                else:
                    random_transform[key] = False # Done flip
                    
            else:
                if value == 0:
                    random_transform[key] = 0
                else:
                    range_min, range_max = self.get_range(value)
                    random_transform[key] = random.uniform(range_min, range_max)                
        return random_transform
    
    
    def get_range(self,item):
        #we have a specific range
        if type(item) == list:
            if len(item) == 2:
                range_min = min(item)
                range_max = max(item)
            else:
                raise NameError('Invalid range for image augmentation. It can either be a single range value of type Int or Float for a +-range  or a list with min & max values')
        elif (type(item) == int) or (type(item) == float):
            range_min = -abs(item)
            range_max = abs(item)                      
        else:
            raise NameError('Invalid range for type. Enter Int or Float or a list with min and max range')
        
        return range_min, range_max
        

        
    def __getcameraTensor(self, sensor_index, rows, all_sensor_transforms):       
        single_sensor_data = self.multi_sensor_data[sensor_index]
        single_sensor_targets = self.targets
        
        single_sensor_samples_batch, single_sensor_targets_batch = self._empty_batch(len(rows))       
        
        
        data_den_obj = keras.preprocessing.image.ImageDataGenerator()
        
        for j, row in enumerate(rows):
            #Note: Be careful of off-by-one errors with rows. 
            
            #indices are for timestamps
            indices_consecutive_timesteps = range(rows[j] - self.length, rows[j], self.sampling_rate)
            file_names_consecutive_timesteps = single_sensor_data[list(indices_consecutive_timesteps)]    
            
            
            for t, file_at_timestep in enumerate(file_names_consecutive_timesteps): #we did not use time 0 to t because length will be different if we skip frames
                loaded_image = resize(imread(file_at_timestep), self.image_dimention, mode='constant')
                
                if self.use_augmentation:                    
                    transform = all_sensor_transforms[row]
                    loaded_image = data_den_obj.apply_transform(loaded_image, transform)
                    
                single_sensor_samples_batch[j, t] = loaded_image
            
            if self.reverse:
                #This will give steering corresponding to first timestep for timestaps 0 to 9.                 
                #single_sensor_targets[indices_consecutive_timesteps[0]] ## To Do: code review
                single_sensor_targets_batch[j] = single_sensor_targets[row-1 - t] 
            else:
                #This will give steering corresponding to last timestep for timestaps 0 to 9.
                # here we want the last element
                 #single_sensor_targets[indices_consecutive_timesteps[-1]] ## To Do: code review
                single_sensor_targets_batch[j] = single_sensor_targets[row-1]
               
                
        if self.reverse:
            single_sensor_samples_batch = single_sensor_samples_batch[:, ::-1, ...]
        
        #if non time series data is needed. This will work as a normal generator
        if self.time_axis == False:
            assert self.length == 1, 'Time axis is False but time series length is not 1'
            #Drop the time axis            
            single_sensor_samples_batch = np.squeeze(single_sensor_samples_batch, axis=1)
            
        return single_sensor_samples_batch, single_sensor_targets_batch   
 







           
if (__name__) == '__main__':
 
    ##########################################################################
    import sys
    sys.path.insert(0, '/home/pt/repository/deep_learning_car/utils') 
    import custom_utils as custom_utils
    

    augmentation_parameters =   {
                            'theta_range':0,                                         
                            'tx_range':0,                                         
                            'ty_range':0,                                 
                            'shear_range':0,
                            'zx_range':0,
                            'zy_range':0,
                            'flip_horizontal':True,
                            'flip_vertical':True,
                            'channel_shift_intencity_range': 0.0,
                            'brightness_range':0.0
                        }
    batch_generator_params = {
                 'length' : 10,
                 'sampling_rate':1,
                 'stride':1,
                 'start_index':0,
                 'end_index':None,
                 'shuffle':True,
                 'reverse':False,
                 'batch_size':5,
                 'image_dimention' : (160,320),
                 'n_channels' : 3,
                 'time_axis':True,
                 'augmentation_parameters': augmentation_parameters
             }
    
    
    
    data_dir = '/home/pt/Desktop/debug_data'
    driving_log_csv = data_dir + '/' + 'driving_log.csv'
    driving_log = custom_utils.update_driving_log(data_dir, driving_log_csv)


    generator = MultiSensorTimeSeriesGenerator([driving_log['center'], driving_log['left'], driving_log['right']], driving_log['steering'], **batch_generator_params)
    
    iterator = generator.__iter__()
    batch = next(iterator)
    custom_utils.show_batch(batch, batch_generator_params, save_dir = data_dir)


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


