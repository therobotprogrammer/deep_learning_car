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
                 seed = None,
                 image_data_gen_obj = None,
                 swap_sensors_on_horizontal_flip = False
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
        self.seed = seed

        self.image_data_gen_obj = image_data_gen_obj
        self.swap_sensors_on_horizontal_flip = swap_sensors_on_horizontal_flip


        if self.start_index > self.end_index:
            raise ValueError('`start_index+length=%i > end_index=%i` '
                             'is disallowed, as no part of the sequence '
                             'would be left to be used as current step.'
                             % (self.start_index, self.end_index))
                   
        
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
            
            
            all_sensor_transforms = self.__get_all_sensor_transforms(rows)
                
        multi_camera_tensor = []        
        
        for sensor_index in range(0,self.n_sensors):
            samples,targets = self.__getcameraTensor(sensor_index, rows, all_sensor_transforms) # is this bad programming? - to do: code review
            multi_camera_tensor.append(samples)
        
        
        if self.image_data_gen_obj != None:
            #Q: Code review - To Do:
            #Q: Code reivew: should we pass multi_camera_tensor or use self. ? do we create an extra copy by not using self. ? 
            #Q: Dows this use twice the ram? May be significant for latge datasets
            #Q: Is it better to use a callback funciton here or just take image_data_gen_obj obj
            
            multi_camera_tensor = self.__applyAugmentations(multi_camera_tensor, rows)
            
        return multi_camera_tensor, targets
    
    
    def __get_all_sensor_transforms(self, rows):
        all_sensor_transforms = {}
        
        for row in rows:
            all_sensor_transforms[row]  = self.image_data_gen_obj.get_random_transform(self.image_dimention, self.seed)
            
        return all_sensor_transforms
        
        
    #Target transform has to be calculated based on sensor transform. For example, if sensor camera is flipped
    #then steering angle has to be flipped too. 
    def __apply_sensor_target_transform_mapping(self, sensor_transform, target):
        if sensor_transform['flip_horizontal']:
            target = -target
        

    '''
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
    '''
        

    def __applyAugmentations(self, multi_camera_tensor, rows):
        all_sensor_row_transforms = self.__get_all_sensor_transforms(rows)
        all_sensor_row_transforms = list(all_sensor_row_transforms.values())
        
        use_batch_level_augmentation_flag = self.__get_batch_level_augmentation_flag()

        for sensor_index in range(0,self.n_sensors):
            single_sensor_samples_batch = multi_camera_tensor[sensor_index] 
            
            if use_batch_level_augmentation_flag:
                sensor_image_data_gen_obj = self.__get_sensor_image_data_gen_obj
                
            for j, sample in enumerate(single_sensor_samples_batch, start = 0): 
                transform = all_sensor_row_transforms[j]
                
                for t, image_at_timestep in enumerate(sample):    
                    image_at_timestep = self.image_data_gen_obj.apply_transform(image_at_timestep, transform)
                    
                    if use_batch_level_augmentation_flag:
                        image_at_timestep = sensor_image_data_gen_obj.standardize(image_at_timestep)
                    
                    multi_camera_tensor[sensor_index][ j, t] = image_at_timestep


        #To Do: Implement multiple swaps for multi camera
        if self.swap_sensors_on_horizontal_flip:
            left_sensor_index = 1
            right_sensor_index = 2
            
            for j, transform in enumerate(all_sensor_row_transforms, start = 0):
                if transform['flip_horizontal']:
                    '''
                    multi_camera_tensor[left_sensor_index][j], multi_camera_tensor[right_sensor_index][j] = self.__swap_row(multi_camera_tensor[left_sensor_index][j], multi_camera_tensor[right_sensor_index][j])
                    
                    '''
                    temp_row = np.empty(multi_camera_tensor[j][0].shape)
                    temp_row = multi_camera_tensor[left_sensor_index][j] 
                    self.__show_row(temp_row)
                    
                    multi_camera_tensor[left_sensor_index][j] = multi_camera_tensor[right_sensor_index][j] 
                    multi_camera_tensor[right_sensor_index][j] = temp_row
                    
                    self.__show_row(temp_row)                

                        
        return multi_camera_tensor

    '''
    def __swap_row(self,row1,row2):
        return row2,row1 
    '''


    def __show_row(self,temp_row):
        #from matplotlib import pyplot as plt
        
        plt.figure(figsize=(25,25)) 
        
        image_count = 1 
        
        for t, timestep in enumerate(temp_row):        
            plt.subplot(1, len(temp_row), image_count)
            image_count = image_count + 1
            plt.imshow(timestep)
            plt.axis("off")
            
        plt.show()
        
        
    
    
    def __get_sensor_image_data_gen_obj(self,single_sensor_samples_batch):
        all_images_in_sensor_samples_batch = []
        
        for j, sample in enumerate(single_sensor_samples_batch): 
            for image_at_timestep in sample:
                all_images_in_sensor_samples_batch.append(image_at_timestep)
        
        sensor_image_data_gen_obj = self.image_data_gen_obj
        sensor_image_data_gen_obj.fit(all_images_in_sensor_samples_batch)
        return sensor_image_data_gen_obj
        
        
    def __get_batch_level_augmentation_flag(self):
        
        if (image_data_gen_obj.featurewise_center or 
            image_data_gen_obj.samplewise_center or 
            image_data_gen_obj.featurewise_std_normalization or 
            image_data_gen_obj.samplewise_std_normalization or 
            image_data_gen_obj.zca_whitening):
            return True
        else:
            return False
        
    def __getcameraTensor(self, sensor_index, rows, all_sensor_transforms):       
        single_sensor_data = self.multi_sensor_data[sensor_index]
        single_sensor_targets = self.targets
        
        single_sensor_samples_batch, single_sensor_targets_batch = self._empty_batch(len(rows))       
        
        
        
        for j, row in enumerate(rows):
            #Note: Be careful of off-by-one errors with rows. 
            
            #indices are for timestamps
            indices_consecutive_timesteps = range(rows[j] - self.length, rows[j], self.sampling_rate)
            file_names_consecutive_timesteps = single_sensor_data[list(indices_consecutive_timesteps)]    
            

            
            for t, file_at_timestep in enumerate(file_names_consecutive_timesteps): #we did not use time 0 to t because length will be different if we skip frames
                loaded_image = resize(imread(file_at_timestep), self.image_dimention, mode='constant')
                
                '''
                # To apply augmentations within this function
                if self.image_data_gen_obj != None:                    
                    transform = all_sensor_transforms[row]
                    loaded_image = self.image_data_gen_obj.apply_transform(loaded_image, transform)
                '''
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
    
    #Q Note: this is different from Keras API. as they use zoom_range etc. Later this can be changed
    
    image_generator_params =    {   
                                     'featurewise_center':False, 
                                     'samplewise_center':False, 
                                     'featurewise_std_normalization':False, 
                                     'samplewise_std_normalization':False, 
                                     'zca_whitening':False, 
                                     'zca_epsilon':1e-06, 
                                     'rotation_range':0.0, 
                                     'width_shift_range':0.0, 
                                     'height_shift_range':0.0, 
                                     'brightness_range':None, 
                                     'shear_range':0.0, 
                                     'zoom_range':0.0, 
                                     'channel_shift_range':0.0, 
                                     'fill_mode':'nearest', 
                                     'cval':0.0, 
                                     'horizontal_flip':True, 
                                     'vertical_flip':False, 
                                     'rescale':None, 
                                     'preprocessing_function':None, 
                                     'data_format':None, 
                                     'validation_split':0.0
                                 }

    
    image_data_gen_obj = keras.preprocessing.image.ImageDataGenerator(**image_generator_params)


    batch_generator_params = {
                 'length' : 5,
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
                 'image_data_gen_obj': image_data_gen_obj,
                 'swap_sensors_on_horizontal_flip': True
             }
    
    
    
    data_dir = '/home/pt/Desktop/debug_data'
    driving_log_csv = data_dir + '/' + 'driving_log.csv'
    driving_log = custom_utils.update_driving_log(data_dir, driving_log_csv)


    generator = MultiSensorTimeSeriesGenerator([driving_log['center'], driving_log['left'], driving_log['right']], driving_log['steering'], **batch_generator_params)
    
    iterator = generator.__iter__()
    batch = next(iterator)
    custom_utils.show_batch(batch, batch_generator_params, save_dir = data_dir)


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


