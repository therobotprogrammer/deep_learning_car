from skimage.io import imread
from skimage.transform import resize
import numpy as np
import os
import pandas as pd
import keras
import cv2
from matplotlib import pyplot as plt


class TimeseriesGenerator_new(keras.utils.Sequence):

    def __init__(self, data, targets, length,
                 sampling_rate=1,
                 stride=1,
                 start_index=0,
                 end_index=None,
                 shuffle=False,
                 reverse=False,
                 batch_size=4):
        self.data = data
        self.targets = targets
        self.length = length
        self.sampling_rate = sampling_rate
        self.stride = stride
        self.start_index = start_index + length
        if end_index is None:
            end_index = len(data) - 1
        self.end_index = end_index
        self.shuffle = shuffle
        self.reverse = reverse
        self.batch_size = batch_size
        
        self.image_dimention = (160,320)
        self.n_channels = 3
        self.n_cameras = 3
        

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
        samples_shape.extend(self.data.shape[1:])
        targets_shape = [num_rows]
        targets_shape.extend(self.targets.shape[1:]) 

        multi_camera_samples_shape = [self.n_cameras, num_rows, self.length // self.sampling_rate, *self.image_dimention, self.n_channels]         
        return np.empty(samples_shape), np.empty(targets_shape)

    def __getitem__(self, index):
        if self.shuffle:
            rows = np.random.randint(
                self.start_index, self.end_index + 1, size=self.batch_size)
        else:
            i = self.start_index + self.batch_size * self.stride * index
            rows = np.arange(i, min(i + self.batch_size *
                                    self.stride, self.end_index + 1), self.stride)

        samples, targets = self._empty_batch(len(rows))
        
        
        #for camera in range(0,self.n_cameras):
        multi_camera_tensor = []
        
        for camera in range(0,self.n_cameras):
            samples, targets = self._empty_batch(len(rows))
            samples,targets = self.__getcameraTensor(rows,samples,targets)
            multi_camera_tensor.append(samples)
        
        
            
            
            

        return multi_camera_tensor, targets
    
     
            
    def __getcameraTensor(self, rows, samples, targets):
        for j, row in enumerate(rows):
            indices = range(rows[j] - self.length, rows[j], self.sampling_rate)
            file_names_time_series = self.data[list(indices)]  
            #print(*file_names_time_series)
            #get_images(c)             
            
            for t, file_at_timestep in enumerate(file_names_time_series): #we did not use time 0 to t because length will be different if we skip frames
                samples[j, t] = resize(imread(file_at_timestep), self.image_dimention)
                #print(t,file_at_timestep)
            targets[j] = self.targets[rows[j]]
            
        if self.reverse:
            return samples[:, ::-1, ...], targets
        return samples, targets    
            

    
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




##########################################################################
    
data_dir = '/home/pt/Desktop/debug_data'
driving_log_csv = data_dir + '/' + 'driving_log.csv'


driving_log = update_driving_log(data_dir, driving_log_csv)


batch_size = 16

'''
params = {
          'image_dimention': (160,320),
          'batch_size': 16,
          'n_channels': 3,
          'shuffle': True,
          'length': 0
         }
'''

#test = DataGenerator(driving_log['center'], driving_log['steering'], **params)

params = {
            'length': 10
         }


#from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np
test = TimeseriesGenerator_new(driving_log['center'], driving_log['steering'], **params)

iterator = test.__iter__()

batch = next(iterator)

samples_batch = batch[0]
labels_batch =  batch[1]
'''

volume_shape = samples_batch.shape
index = {'sample':0,'time':1,'height':2,'width':3,'channels':4}

total_samples = volume_shape[index['sample']] 
total_timesteps = volume_shape[index['time']]
total_images = total_samples * total_timesteps                         


plt.figure(figsize=(15, 3))

image_count = 1
for s, sample in enumerate(samples_batch):    
    for t, timestep in enumerate(sample):

        plt.subplot(total_samples, total_timesteps, image_count)
        image_count = image_count + 1
        plt.imshow(timestep)
        plt.axis('off')
plt.show()
'''