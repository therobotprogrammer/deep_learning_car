from skimage.io import imread
from skimage.transform import resize
import numpy as np
import os
import pandas as pd
import keras
import cv2
from matplotlib import pyplot as plt


class MultiSensorTimeSeriesGenerator(keras.utils.Sequence):

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
                 time_axis = True):
        
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
        if self.shuffle:
            rows = np.random.randint(
                self.start_index, self.end_index + 1, size=self.batch_size)
        else:
            i = self.start_index + self.batch_size * self.stride * index
            rows = np.arange(i, min(i + self.batch_size *
                                    self.stride, self.end_index + 1), self.stride)

        multi_camera_tensor = []
        
        for sensor_index in range(0,self.n_sensors):
            samples,targets = self.__getcameraTensor(sensor_index, rows) # is this bad programming? - to do: code review
            multi_camera_tensor.append(samples)
        
        return multi_camera_tensor, targets
    
     
            
    def __getcameraTensor(self, sensor_index, rows):       
        single_sensor_data = self.multi_sensor_data[sensor_index]
        single_sensor_targets = self.targets
        
        single_sensor_samples_batch, single_sensor_targets_batch = self._empty_batch(len(rows))       
        
        for j, row in enumerate(rows):
            #Note: Be careful of off-by-one errors with rows. 
            
            #indices are for timestamps
            indices_consecutive_timesteps = range(rows[j] - self.length, rows[j], self.sampling_rate)
            file_names_consecutive_timesteps = single_sensor_data[list(indices_consecutive_timesteps)]    
            
            
            for t, file_at_timestep in enumerate(file_names_consecutive_timesteps): #we did not use time 0 to t because length will be different if we skip frames
                single_sensor_samples_batch[j, t] = resize(imread(file_at_timestep), self.image_dimention, mode='constant')
                
            
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
 







           

   
def strip_filenames(old_path):
    old_path = old_path.split("\\") #handles windows generated files
    *directory, filename = old_path        
    directory, filename = os.path.split(filename) #handles linux generated files
    return("/" + filename) 
    


def update_driving_log(data_dir, driving_log_csv = None, ralative_path = False):  
    if driving_log_csv == None:
        driving_log_csv = data_dir + '/' + 'driving_log.csv'
        
    new_image_path = ''
    if ralative_path == False:
        new_image_path = data_dir + '/IMG'
    
    driving_log_pd = pd.read_csv(driving_log_csv, header= None)
    driving_log_pd_temp = driving_log_pd.iloc[:,0:3]    
    driving_log_pd_temp = driving_log_pd_temp.applymap(strip_filenames)    
    driving_log_pd_temp = new_image_path + driving_log_pd_temp.astype(str)    
    driving_log_pd.iloc[:,0:3] = driving_log_pd_temp          
    driving_log_pd.to_csv(driving_log_csv, index = False, header = False)    
    driving_log_pd.columns = ['center','left', 'right', 'x','y','z','steering']       

    return driving_log_pd


'''

def show_batch(batch, figsize=(15, 3), time_axis = False):
    multi_camera_samples_batch = batch[0]
    multi_camera_labels_batch =  batch[1]    
    total_cameras = len(multi_camera_samples_batch)
    
    for camera in range(0,total_cameras):
        samples_batch = multi_camera_samples_batch[camera]        
        volume_shape = samples_batch.shape        
        
        if time_axis == False:
           samples_batch = np.expand_dims(samples_batch, axis = 1) 
        
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


##########################################################################

data_dir = '/home/pt/Desktop/debug_data'
driving_log_csv = data_dir + '/' + 'driving_log.csv'
driving_log = update_driving_log(data_dir, driving_log_csv)

params = {
             'length' : 1,
             'sampling_rate':1,
             'stride':1,
             'start_index':0,
             'end_index':None,
             'shuffle':False,
             'reverse':False,
             'batch_size':5,
             'image_dimention' : (160,320),
             'n_channels' : 3,
             'time_axis':False
         }

input_data = [driving_log['left'], driving_log['center'], driving_log['right']]
test = MultiSensorTimeSeriesGenerator(input_data, driving_log['steering'], **params)

iterator = test.__iter__()

batch = next(iterator)

params['time_axis']

show_batch(batch,  figsize = (400,4) , time_axis = False )

'''
