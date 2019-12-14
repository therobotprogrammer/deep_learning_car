#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 18:28:52 2019

@author: pt
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 14:27:06 2019

@author: pt
"""


import sys
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt 
from scipy.stats import truncnorm, norm
import seaborn as sns
import numpy as np
import bisect



data_dir = '/media/pt/ramdisk/data_big/data'
code_dir = '/home/pt/Documents/deep_learning_car/utils'

sys.path.insert(0, code_dir) 
import custom_utils as custom_utils
    

driving_log_csv = data_dir + '/' + 'driving_log.csv'
driving_log = custom_utils.update_driving_log(data_dir, driving_log_csv)
#driving_log['steering'].hist()
original_sample_size = driving_log.shape[0]
sample_df = driving_log.groupby('steering')
bins = sample_df.groups

sorted_bins = np.array(list(bins.keys()))
sorted_bins.sort()

max_random_sampling_point = -2

def find_nearest_bin(key_to_find):
    #These if statements are to make a search inclusive of -1 and +1. 
    #This inclusion is done by sclecting side. Otherwise -.999 becomes -.95 bin and not -1 bin and vice versa
    if key_to_find > 0 and key_to_find <= 1:
        closest_id = np.searchsorted(sorted_bins,key_to_find, side = 'left')          
    elif key_to_find <= 0:
        closest_id = np.searchsorted(sorted_bins,key_to_find, side = 'right')
        closest_id = closest_id-1
    elif key_to_find > 1:    
        closest_id = np.searchsorted(sorted_bins,key_to_find, side = 'left')
        closest_id = closest_id-1
    elif key_to_find < 1:    
        closest_id = np.searchsorted(sorted_bins,key_to_find, side = 'left')
        closest_id = closest_id-1
    else:
        closest_id = np.searchsorted(sorted_bins,key_to_find, side = 'left')
    return sorted_bins[closest_id]


def find_nearest_4(key_to_find):
    #These if statements are to make a search inclusive of -1 and +1. 
    #This inclusion is done by sclecting side. Otherwise -.999 becomes -.95 bin and not -1 bin and vice versa
    closest_id = np.searchsorted(sorted_bins,key_to_find, side = 'right')
    closest_id = closest_id-1
    return sorted_bins[closest_id]


def find_nearest_bin_2(key_to_find):
    closest_id_left = np.searchsorted(sorted_bins,key_to_find, side = 'left')   
    closest_id_right = np.searchsorted(sorted_bins,key_to_find, side = 'left')   
#
#    print('id_l',closest_id_left)
#    print('id_r', closest_id_right)
#
#    print('value_l',closest_id_left)
#    print('value_r', closest_id_right)    
    return sorted_bins[closest_id_left]


def get_random_sample_from_nearest_bin(nearest_bin):
    values = bins[nearest_bin]
    random_sample = np.random.choice(values)
    return random_sample

def get_sample_from_distribution(distribution):
    random_sampling_point = np.random.choice(distribution)
    global max_random_sampling_point 
    if (random_sampling_point > max_random_sampling_point):
        max_random_sampling_point = random_sampling_point
        
    #nearest_bin = find_nearest_3(random_sampling_point)
    nearest_bin = find_nearest_3(random_sampling_point)
    random_sample = get_random_sample_from_nearest_bin(nearest_bin)
    return nearest_bin, random_sample

def get_random_distribution(rmin = -1, rmax = 1, distribution_sampling_points = 100000):
    random_distribution = np.random.uniform(rmin, rmax, size = distribution_sampling_points)    
    return random_distribution

def get_normalised_distribution(rmin = -1, rmax = 1, distribution_sampling_points = 100000):
    normal_distribution = np.random.normal(loc = 0, scale = .25, size = distribution_sampling_points )       
    return normal_distribution
    

def get_closest(sorted_bins, values):
    #make sure array is a numpy array
    array = np.array(sorted_bins)

    # get insert positions
    idxs = np.searchsorted(array, values, side="left")

    # find indexes where previous index is closer
    prev_idx_is_less = ((idxs == len(array))|(np.fabs(values - array[np.maximum(idxs-1, 0)]) < np.fabs(values - array[np.minimum(idxs, len(array)-1)])))
    idxs[prev_idx_is_less] -= 1

    return array[idxs]


def get_closest2(value):
    xarr = sorted_bins    
    srt_ind = xarr.argsort()
    xar = xarr.copy()[srt_ind]
    xlist = xar.tolist()
    index = bisect.bisect_left(xlist, value)
    return (sorted_bins[index])


def find_nearest_3(value):
#    if value> sorted_bins[39] and value < len(sorted_bins) -1 :
#        print('value: ', value)
    array = sorted_bins
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    
#    if value> sorted_bins[39] and value < len(sorted_bins) -1 :
#        print('idx: ', idx)
#        print('')
#        print('')
    return array[idx]


# Here edge correction is done. For example if a random distribution is 
# generated between -1 and +1, and the bins have increments of .05
# then for numbers between .95 is sclected as the nearest bin for numbers
# on range .925 to .975. But the edges i.e. 1 is only chosen for distribution
# range from .975 to 1. This is half the range that other numbers get. 
# this causes the edges to have half the probability of getting sclected. 
# this method extends the distribution range beyond from 1 to 1.024 so 
# boundry conditions also have the same probability
    
def edge_correction_for_distribution(distribution_range_min, distribution_range_max):
    array = sorted_bins
    array = np.asarray(array)
    average_bin_width = (array.max() - array.min() ) / (array.size -1) 
    new_distribution_range_min = distribution_range_min - average_bin_width/2
    new_distribution_range_max = distribution_range_max + average_bin_width/2
    return new_distribution_range_min, new_distribution_range_max

distribution_resolution = 10000
distribution_range_min = -1
distribution_range_max = 1
distribution_type = 'random'
correct_edges_for_random_distribution = True



import random
random.seed(100)
distribution = 0

if distribution_type == 'normal':
    if correct_edges_for_random_distribution:
        distribution_range_min, distribution_range_max = edge_correction_for_distribution(distribution_range_min, distribution_range_max)
    distribution = truncnorm.rvs(distribution_range_min, distribution_range_max, size = distribution_resolution)

elif distribution_type == 'random':
    if correct_edges_for_random_distribution:
        distribution_range_min, distribution_range_max = edge_correction_for_distribution(distribution_range_min, distribution_range_max)
    distribution = get_random_distribution(distribution_range_min,distribution_range_max,distribution_resolution)

else:
    distribution = get_random_distribution(distribution_range_min,distribution_range_max,distribution_resolution)


sample_size = driving_log.shape[0]

samples = []
nearest_bins = [] 

for sample in range(0, sample_size):
    nearest_bin, sample = get_sample_from_distribution(distribution)
    nearest_bins.append(nearest_bin)    
    samples.append(sample)


sampled_driving_log = driving_log.iloc[samples]




plt.figure()
plt.title('Original Targets')
sns.distplot(driving_log['steering'], hist=True, kde=False, rug = True,  color = 'blue', hist_kws={'edgecolor':'black'}, bins = sorted_bins.size)


plt.figure()
plt.title('Source distribution used for sampling')
sns.distplot(distribution, hist=True, kde=True, color = 'green', hist_kws={'edgecolor':'black'}, bins = sorted_bins.size)


plt.figure()
plt.title('Targets - Samples using distribution')
sns.distplot(sampled_driving_log['steering'], hist=True, kde=True, color = 'red', hist_kws={'edgecolor':'black'}, bins = sorted_bins.size)


sampled_driving_log.groupby('steering').count().plot()

#temp = balanced_driving_log.sort_index()
sampled_driving_log_csv = data_dir + '/' + 'sampled_driving_log.csv'
sampled_driving_log.to_csv(sampled_driving_log_csv, index = False, header = True)   
#balanced_driving_log = driving_log_pd.reset_index(drop = True)  

new_bins = sampled_driving_log.groupby('steering').groups
sampled_driving_log.groupby('steering').size()

assert (len(bins) == len(new_bins)), 'Some samples were lost. Try increasing resolution of distribution used. '







    
    
    
    
    
    
    
    
    
    


