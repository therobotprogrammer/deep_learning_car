#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 17:00:49 2018

@author: pt
"""

import numpy as np
import keras
import os

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, image_directory, batch_size=32, dim=(160,320), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.image_directory = image_directory

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __load_image(self, infilename ) :
        img = Image.open( infilename )
        img.load()
        data = np.asarray( img, dtype="int32" )
        return data

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            #X[i,] = np.load( self.image_directory +  '/' + ID )
            filename = self.image_directory +  '/' + ID
            image_np_array = self.__load_image(filename)
            X[i,] = image_np_array 


            # Store class
            y[i] = self.labels[ID]

        return X, y
    
    
    

import numpy as np
import pandas as pd

from PIL import Image
import numpy as np

from keras.models import Sequential
#from my_classes import DataGenerator

# Parameters
params = {'dim': (160,320),
          'batch_size': 64,
          'n_classes': 6,
          'n_channels': 3,
          'shuffle': True,
          'image_directory': '/home/pt/repository/deep_learning_car/data/IMG'}


data_df = pd.read_csv('/home/pt/repository/deep_learning_car/data/driving_log.csv')
center = data_df.iloc[:,0]
left = data_df.iloc[:,1]
right = data_df.iloc[:,2]

steering = data_df.iloc[:,3]

training_generator = DataGenerator(center, steering, **params)

a = training_generator.__getitem__(7)
b = training_generator.__getitem__(8)
c = training_generator.__getitem__(9)



'''
a[0].shape()


partition = pd.DataFrame

# Datasets
partition.train = center
labels = # Labels

# Generators


 
training_generator.

training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

# Design model
model = Sequential()
[...] # Architecture
model.compile()

# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6)


'''



















