#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 13:43:45 2019

@author: pt
"""
from tensorflow import keras

import MultiSensorTimeSeriesGenerator
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

#Profiler tools
import time
from line_profiler import LineProfiler
import pickle
import time
from functools import wraps
 
##########################################################################

import sys
sys.path.insert(0, '/home/pt/repository/deep_learning_car/utils') 
import custom_utils as custom_utils

#Q Note: this is different from Keras API. as they use zoom_range etc. Later this can be changed

import custom_utils as custom_utils
from MultiSensorTimeSeriesGenerator import MultiSensorTimeSeriesGenerator

import os
import tensorflow as tf

image_generator_params =    {   
                             'featurewise_center':False, 
                             'samplewise_center':False, 
                             'featurewise_std_normalization':False, 
                             'samplewise_std_normalization':False, 
                             'zca_whitening':False, 
                             'zca_epsilon':1e-06, 
                             'rotation_range':20.0, 
                             'width_shift_range':0.0, 
                             'height_shift_range':0.0, 
                             'brightness_range':None, 
                             'shear_range':00.0, 
                             'zoom_range':0.0, 
                             'channel_shift_range':0.3, 
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
image_data_gen_obj = None

batch_generator_params = {
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
             'time_axis':False,
             'image_data_gen_obj': image_data_gen_obj,
             'swap_sensors_on_horizontal_flip': True,
             'predict_mode': False
         }



data_dir = '/media/pt/ramdisk/data_big/data'
#data_dir = '/media/pt/ramdisk/data/data'
#data_dir = '/media/pt/ramdisk/debug_data'
driving_log_csv = data_dir + '/' + 'driving_log.csv'
driving_log = custom_utils.update_driving_log(data_dir, driving_log_csv)
driving_log = driving_log.reset_index()

generator = MultiSensorTimeSeriesGenerator([driving_log['center'], driving_log['left'], driving_log['right']], driving_log['steering'], **batch_generator_params)

iterator = generator.__iter__()

batch1 = generator.__getitem__(0)
#custom_utils.show_batch(batch1, batch_generator_params)


tf_data_dir = data_dir + '/' + 'tf_records'

if not os.path.isdir(tf_data_dir):
    os.makedirs(tf_data_dir)
else:
    os.removedirs(tf_data_dir)
    os.makedirs(tf_data_dir)


#for batch_index in range (0,total_batches -2):    
#    #dataset = tf.data.Dataset.from_tensor_slices(next(iterator))
#    temp_batch = generator.__getitem__(batch_index)
#
#    dataset = tf.data.Dataset.from_tensor_slices({'i1': temp_batch[0][0], 'i2':temp_batch[0][0], 'i3':temp_batch[0][0] }, temp_batch[1]))
#    print(batch_index)
#    if batch_index == 1:
#        break
#    











#features_placeholder = tf.placeholder(features.dtype, features.shape)
#labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
#

#
#
#i1 = batch1[0][0].tolist()
#i2 = batch1[0][0].tolist()
#i3 = batch1[0][0].tolist()
#
#i_all = [i1,i2,i3]
#
#b = batch1[1].tolist()
#
#
#dataset = tf.data.Dataset.from_tensor_slices(((i1,i2,i3), b))


total_batches = generator.__len__()
combined_dataset = tf.data.Dataset

batch =  generator.__getitem__(0)
b = batch[1].tolist()

i1 = batch[0][0]
i2 = batch[0][1]
i3 = batch[0][2]
dataset = tf.data.Dataset.from_tensor_slices((   (i1,i2,i3)   , b))
combined_dataset = dataset


for batch_index in range (0,total_batches -2):
    
    batch =  generator.__getitem__(batch_index)
    b = batch[1].tolist()
    
    i1 = batch[0][0]
    i2 = batch[0][1]
    i3 = batch[0][2]
    dataset = tf.data.Dataset.from_tensor_slices((   (i1,i2,i3)   , b))
    #dataset = dataset.batch(128)

    if(batch_index == 0):
        combined_dataset = dataset
    combined_dataset = combined_dataset.concatenate(dataset)
    
#    with tf.python_io.TFRecordWriter('batch_' + str(batch_index) + '.tfrecord') as writer:
#        writer.write(dataset.SerializeToString())
    
    
    print('processing batch: ', batch_index)

##print(dataset)
#
#
#
#
##dataset = tf.data.Dataset.range(100)
#sess = tf.Session()
#
##combined_dataset = combined_dataset.batch(12)
#
#iterator = combined_dataset.make_one_shot_iterator()
#
#
#next_element = iterator.get_next()
#
##value = sess.run(next_element)
##print(value[1])
#
#
#
#for batch_index in range (0,2):
#  value = sess.run(next_element)
#  print(value[1])    
#
#
#
#dataset_batch_size = 5
#
#batched_dataset = combined_dataset.batch(dataset_batch_size)
#iterator = batched_dataset.make_one_shot_iterator()
#next_element = iterator.get_next()
#
#
#for batch_index in range (0,total_batches//dataset_batch_size):
#  value = sess.run(next_element)
#  print(value[1])    
#
#
#
#
#
    
#sess = tf.Session()
#dataset_batch_size = 128
#
#batched_dataset = combined_dataset.batch(128)
#
#
#iterator = batched_dataset.make_one_shot_iterator()
#next_element = iterator.get_next()
#
#
#for batch_index in range (0,total_batches//dataset_batch_size):
#  value = sess.run(next_element)
#  
#  print(value[1])    
#  #break
#
#
#







#def wrapper():
#    yield next(iterator)
#
#temp = next(iterator)
##custom_utils.show_batch(temp, batch_generator_params)
#
#ds = tf.data.Dataset.from_generator(wrapper, ((tf.uint8, tf.uint8, tf.uint8), tf.float32) )
#print(ds)

#
#iter = dataset.make_one_shot_iterator()
#el = iter.get_next()
#
#with tf.Session() as sess:
#    ans2 = (sess.run(el)) # output: [ 0.42116176  0.40666069]
#    


combined_dataset = combined_dataset.repeat(10000)
combined_dataset = combined_dataset.batch(128)

history = model.fit(combined_dataset, epochs=100, verbose=1, steps_per_epoch=8500//128)

#history = model.fit_generator(generator=train_generator, validation_data = validation_generator, epochs = model_params['epochs'], initial_epoch = model_saver.initial_epoch, callbacks = callbacks, use_multiprocessing = True, workers=24, max_queue_size=10, verbose=1  )



#
#
#
#custom_utils.show_batch(ans2, batch_generator_params)
#
#
#
#
#

#print(ds)









