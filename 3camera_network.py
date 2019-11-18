#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 10:22:25 2018

@author: pt
"""

#To Do: Implement with cv2.imread


import numpy
import scipy.ndimage
import matplotlib.pyplot as plt 

# coding: utf-8

# In[1]:

#compute modes: colab, google_cloud, local_dev
compute_mode = 'local_dev'

if compute_mode == 'colab':
    google_drive = '/content/google_drive'
    project_dir = '/content/Deep_Learning'
    script_dir = '/content/scripts'
    download_code = True
    download_data = True
    
elif compute_mode == 'google_cloud':
    google_drive = '/home/pt/fake_google_drive'
    script_dir = '/home/pt/scripts'
    project_dir = '/home/pt/Deep_Learning_nvidia'
    download_code = True
    download_data = True
    
elif compute_mode == 'local_dev':
    google_drive = '/home/pt/Documents/Results'
    #script_dir = '/home/pt/scripts'
    project_dir = '/home/pt/Documents/deep_learning_car'
    auto_downloader_dir = '/home/pt/Documents/Auto-Downloader/Code'
    data_dir = '/home/pt/Documents/DATA/data_big/data'
    code_dir = '/home/pt/Documents/deep_learning_car/utils'
    download_code = False
    download_data = False
    
else:
    print('Invalid compute mode')

# In[2]:

#========================================== 
# Author: Pranav
# If you find this code useful, 
# please include this section and give 
# credit to original author.  
#==========================================


import os

if not os.path.isdir(google_drive):
    os.makedirs(google_drive) 

if (compute_mode == 'local_dev') or (compute_mode == 'google_cloud'):
    import wget, shutil, sys 


if compute_mode == 'colab':
    get_ipython().system('pip install -q pydot')
    get_ipython().system('pip install graphviz')

    get_ipython().system('pip install wget')

    import wget, shutil, sys 

    setup_kaggle_token_from_google_drive = False



    if os.path.isdir(script_dir):
        shutil.rmtree(script_dir)
    os.makedirs(script_dir)

    os.chdir(script_dir)
    script_url = 'https://raw.githubusercontent.com/therobotprogrammer/Auto-Downloader/master/Code/setup.sh'
    wget.download(script_url, out = script_dir + '/setup.sh')

    get_ipython().system('chmod +x setup.sh')
    get_ipython().system('bash setup.sh')
    get_ipython().system('clear')


    # Download AutoDownloader

    if not os.path.isdir(script_dir):
        os.makedirs(script_dir)

    if os.path.isfile(script_dir + '/AutoDownloader.py'):
        os.remove(script_dir + '/AutoDownloader.py')

    AutoDownloader_url = 'https://raw.githubusercontent.com/therobotprogrammer/Auto-Downloader/master/Code/AutoDownloader.py'
    wget.download(AutoDownloader_url, out = script_dir + '/AutoDownloader.py')

    sys.path.insert(0, script_dir) 

    get_ipython().system('clear')

    

    if os.path.isdir(google_drive):
      print('Google drive previously mounted.')

    else:

        # To mount Google Drive
      from google.colab import auth
      auth.authenticate_user()
      from oauth2client.client import GoogleCredentials
      print("Sometimes you may have to repeat verification process & enter new token")
      creds = GoogleCredentials.get_application_default()
      import getpass
      get_ipython().system('google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL')
      vcode = getpass.getpass()
      get_ipython().system('echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}')

      if not os.path.isdir(google_drive):
        os.mkdir(google_drive)
        get_ipython().system('google-drive-ocamlfuse' + '/content/google_drive')
        print("Google Drive mounted at: " + google_drive)
      else:
        print("Google Drive already mounted at: " + google_drive)         

      get_ipython().system('clear  ')

    if setup_kaggle_token_from_google_drive == True:
      # To get Kaggle token from Google Drive
      from googleapiclient.discovery import build
      import io, os
      from googleapiclient.http import MediaIoBaseDownload
      from google.colab import auth

      auth.authenticate_user()
      drive_service = build('drive', 'v3')
      results = drive_service.files().list(
              q="name = 'kaggle.json'", fields="files(id)").execute()
      kaggle_api_key = results.get('files', [])

      filename = "/content/.kaggle/kaggle.json"
      os.makedirs(os.path.dirname(filename), exist_ok=True)

      request = drive_service.files().get_media(fileId=kaggle_api_key[0]['id'])
      fh = io.FileIO(filename, 'wb')
      downloader = MediaIoBaseDownload(fh, request)
      done = False
      while done is False:
          status, done = downloader.next_chunk()
          print("Download Kaggle authentication token from Google Drive %d%%." % int(status.progress() * 100))
          print("Done setting up Kaggle authentication token")
      os.chmod(filename, 600)  


    print('Setup Complete !!!')

# In[3]:


# Download AutoDownloader
if (compute_mode == 'google_cloud') or (compute_mode == 'colab'):   
    if not os.path.isdir(script_dir):
        os.makedirs(script_dir)
        
    if os.path.isfile(script_dir + '/AutoDownloader.py'):
        os.remove(script_dir + '/AutoDownloader.py')
    
    AutoDownloader_url = 'https://raw.githubusercontent.com/therobotprogrammer/Auto-Downloader/master/Code/AutoDownloader.py'
    wget.download(AutoDownloader_url, out = script_dir + '/AutoDownloader.py')
    sys.path.insert(0, script_dir) 
    
elif compute_mode == 'local_dev':
    sys.path.insert(0, auto_downloader_dir)
    
else:
    print('Invalid compute mode')


# In[4]:
print('Project dir: ' + project_dir)

if not project_dir:
    os.makedirs(project_dir)

#    /COMMON_UTILS is a special directory. All its contents are deleted & downloaded for latest copy. 
#    Use it for dependencies
#    All other directorys dont download if data already exists. Use it for training/test data etc
#    All zip files are automatically unzipped & original zip is deleted.

#    To create sharable link for file on Google Drive: https://drive.google.com/uc?export=download&id=YOUR_FILE_ID


#These are relative paths
if (compute_mode == 'google_cloud') or (compute_mode == 'colab'):       
    common_utils_dir = '/COMMON_UTILS'
    data_sub_dir = '/DATA'
    code_sub_dir = '/CODE'
    
    data_to_download =  {    
        
        data_sub_dir:                       [
                                            'https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip'
                                       ],
    
    
        code_sub_dir:                      [
                                            'https://github.com/therobotprogrammer/deep_learning_car/archive/master.zip',                                        
                                       ],
        
        common_utils_dir:              [
                                            
                                       ],
        
                        }
    
    
    if download_code == False:
        del data_to_download[code_sub_dir]
    if download_data == False:
        del data_to_download[data_sub_dir]


from AutoDownloader import AutoDownloader
auto_dl = AutoDownloader(local_timezone =  'Asia/Kolkata')
if(download_code or download_data): 
    auto_dl.initiate(project_dir, data_to_download, recreate_dir = True )
    data_subdir = data_sub_dir.split('/')[-1].split('.zip')[0]    
    data_dir = project_dir + '/' + data_subdir + '/data'
    code_dir = project_dir + '/' + code_sub_dir

auto_dl.recursively_add_to_path(code_dir ) #Warning: Do not add code recursively if you dont trust the source.

 


# Push Notifications
user_key = "umnxi6yau47cpp9evkn6nfa5zbpipt"
token ="aev81xsojcq2ggdpevsia5rzuw5x12"
#auto_dl.setup_pushover_credintials(user_key,token)

#To send push notificaiton:
#auto_dl.send_notification('Hello form Colab!!!')

# To see any directory  
#auto_dl.showDirectory(project_dir + '/CODE') 

#To get localtime as string
#auto_dl.get_time_string()



# In[5]:

from model_load_and_save import model_load_and_save
from colab_CSVLogger import colab_CSVLogger


model_save_top_directory = google_drive + '/deep_learning/01_Self_Driving_Car_Nvidia_Paper/saved_models' 
#model_save_top_directory = '~/my_deep_learning/01_Self_Driving_Car_Nvidia_Paper/saved_models' 


save_load_params = {
                            'use_last_model' : False,
                            'create_time_stamped_dirs' : True                                
                   }

model_saver = model_load_and_save(model_save_top_directory, **save_load_params)
print('log file: ' + model_saver.csv_save_file)

### Override tensorboard log dir to a fixed directory
model_saver.tensorboard_log_dir = '/home/pt/Documents/Results/deep_learning/01_Self_Driving_Car_Nvidia_Paper/saved_models'

# In[7]:

#from keras.backend.common import set_floatx
#set_floatx('float16')

#import tensorflow as tf
#
#from tensorflow.python.keras.models import Model
#from tensorflow.python.keras.layers import Input,Dense, Lambda, concatenate, SeparableConv1D, MaxPooling1D, Dropout, Conv2D, Flatten, BatchNormalization, ELU
#from IPython.display import SVG
#from tensorflow.python.keras.utils.vis_utils import model_to_dot
#from tensorflow.python.keras.utils import plot_model
#from sklearn.model_selection import train_test_split #to split out training and testing data 
#from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard, ReduceLROnPlateau
#from tensorflow.python.keras.optimizers import Adam
#from tensorflow.python.keras.preprocessing import image
#import keras









from keras.models import Model
from keras.layers import Input,Dense, Lambda, concatenate, SeparableConv1D, MaxPooling1D, Dropout, Conv2D, Flatten, BatchNormalization, ELU
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from sklearn.model_selection import train_test_split #to split out training and testing data 
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing import image






import os


import sys



#utils_dir = project_dir + '/CODE/deep_learning_car-master/utils'
import custom_utils as custom_utils
from MultiSensorTimeSeriesGenerator import MultiSensorTimeSeriesGenerator


def show_sample_from_generator(generator, batch_generator_params):
    iterator = generator.__iter__()
    batch = next(iterator)
    custom_utils.show_batch(batch, batch_generator_params)
    return batch


#Q Why this notmalisation
def image_normalization(x):
    return ( (x/127.5) - 1.0 )
    
#To Do: Verify if Lambda is correctly used
def build_single_sensor_network(x):    
    #x = Lambda(image_normalization)(x)
    #x = Dense(32)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(24, (5,5), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    
    
    
        #x = Lambda(lambda x: x/127.5-1.0)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(24, (5,5), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    
    
    x = Conv2D(36, (5,5), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    
    x = Conv2D(48, (5,5), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    
    x = Conv2D(64, (3,3), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    
    x = Conv2D(64, (3,3), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    
    return x

def show_model(model):
    display(SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg')))


def build_nvidia_paper_model(sensor_count, single_sensor_input_shape, overfit_training = False):
    
    sensor_inputs = []
    sensor_outputs = []
    
    
    for sensor_id in range(0, sensor_count):
        #Q Will it if we make input_sensor = Input() and then appent list.append(input_sensor) 3 times. will it be same data or different for eact sensor?
        #Q How to do this as an array?
        
        single_sensor_input = Input(shape=single_sensor_input_shape)
        single_sensor_output = build_single_sensor_network(single_sensor_input)
        
        sensor_inputs.append(single_sensor_input)        
        sensor_outputs.append(single_sensor_output)
        
    ### To Do: Enable Batch Normalisation back
    x = concatenate(sensor_outputs)
    


    x = Flatten()(x)
    
    if overfit_training == False:
      x = Dropout(.5)(x)
      
    x = Dense(100)(x)
    x = Dropout(.5)(x)
    x = ELU()(x)
    
    
    x = Dense(50)(x)
    x = Dropout(.5)(x)
    x = ELU()(x)
    
    
    x = Dense(10)(x)
    x = ELU()(x)
       
    x = Dense(1)(x)
    x = ELU()(x)

    x = Dense(1, activation = 'tanh')(x)
    
    
    #x = SeparableConv1D(5,1)(x)



    model = Model(inputs = sensor_inputs, outputs = x)
    #model.summary()   
    #show_model(model) #To Do: install pydot on google colab and do this
    
    return model
    


def get_sensor_count_and_input_shape(sample_generator):
    iterator = sample_generator.__iter__()    
      
    batch = next(iterator)
    input_data = batch[0]   
    
    sensor_count = len(input_data)    
    single_sensor_input_shape = input_data[0][0].shape
    
    return sensor_count, single_sensor_input_shape
    

    
    
image_generator_params =    {   
                                 'featurewise_center':False, 
                                 'samplewise_center':False, 
                                 'featurewise_std_normalization':False, 
                                 'samplewise_std_normalization':False, 
                                 'zca_whitening':False, 
                                 'zca_epsilon':1e-06, 
                                 'rotation_range':0, 
                                 'width_shift_range':0.0, 
                                 'height_shift_range':0.0, 
                                 'brightness_range':[.8,1], 
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
                                 'validation_split':0.0,                                 
                             }
image_data_gen_obj = image.ImageDataGenerator(**image_generator_params)

### To Do: Change image_data_gen_obj so its not None


overfit_training = False
if overfit_training == True:
  image_data_gen_obj = None
  
#image_data_gen_obj = None  

batch_generator_params =    {
                                 'length' : 1,
                                 'sampling_rate':1,
                                 'stride':1,
                                 'start_index':0,
                                 'end_index':None,
                                 'shuffle':True,
                                 'reverse':False,
                                 'batch_size':16,
                                 'image_dimention' : (160,320),
                                 'n_channels' : 3,
                                 'time_axis':False,
                                 'image_data_gen_obj': image_data_gen_obj,
                                 'swap_sensors_on_horizontal_flip': True,
                                 'typecast_target_datatype': 'float16',
                             }
    

model_params =              {
                                'train_to_test_split_ratio' : .8,
                                'random_seed' : 0,
                                'learning_rate': 1.0e-4,
                                'epochs': 100
                            }



driving_log_csv = data_dir + '/' + 'driving_log.csv'

#driving_log_csv = data_dir + '/' + 'sampled_driving_log.csv'


driving_log = custom_utils.update_driving_log(data_dir, driving_log_csv)
plt.figure()
driving_log.groupby('steering').count().plot()



split_point = int(len(driving_log) * model_params['train_to_test_split_ratio'])
driving_log_train = driving_log[0:split_point]
driving_log_validation = driving_log[split_point + batch_generator_params['length'] * 2:] # length*2 so that car has moved far away from train data spot

#if we done reset index, then test data has index split_point onwards. This will cause problem with timeseriesgenerator
driving_log_train = driving_log_train.reset_index()
driving_log_validation = driving_log_validation.reset_index() 


### To Do: These are read only parameters
batch_generator_params['create_tensors'] =  False
batch_generator_params['read_tensors'] =  False
#batch_generator_params['batch_size'] =  128


#
#### Put this in other class & datastructure so it doesnt interfere with time series data
#print('driving_log_train -> Steering values before normalisaiton:')
#driving_log_train['steering'].hist()
#balanced_dataset = driving_log_train.groupby('steering', group_keys=False)
#balanced_dataset.apply(lambda x: x.sample(balanced_dataset.size().min())).reset_index(drop=True)
#
#
#balanced_dataset = balanced_dataset.head()
#balanced_dataset = balanced_dataset.reset_index()
#print('driving_log_train -> Steering values after normalisaiton:')
#balanced_dataset['steering'].hist()
#
#
#
#

train_generator = MultiSensorTimeSeriesGenerator([driving_log_train['center'], driving_log_train['left'], driving_log_train['right']], driving_log_train['steering'], **batch_generator_params)

validation_batch_generator_params = batch_generator_params
validation_batch_generator_params['batch_size'] = 512
validation_batch_generator_params['image_data_gen_obj'] = None

validation_batch_generator_params['create_tensors'] =  False
validation_batch_generator_params['read_tensors'] =  True

validation_generator = MultiSensorTimeSeriesGenerator([driving_log_validation['center'], driving_log_validation['left'], driving_log_validation['right']], driving_log_validation['steering'], **validation_batch_generator_params)


### To DO: fix this
sample_batch = show_sample_from_generator(train_generator, batch_generator_params)

sensor_count, input_shape = get_sensor_count_and_input_shape(train_generator)


if model_saver.use_last_model and model_saver.model_loaded_sucessfully:
    model = model_saver.last_model
else:
    model = build_nvidia_paper_model(sensor_count, input_shape, overfit_training)


show_model(model)

#callbacks
callbacks = []
if overfit_training == True:
  callbacks.append(CSVLogger(model_saver.csv_save_file, append = True))
else:
  callbacks.append(ModelCheckpoint(filepath=model_saver.model_save_file, monitor='val_loss',  save_best_only=False, mode='auto'))
  callbacks.append(colab_CSVLogger(model_saver.csv_save_file, append = True))
if (compute_mode == 'colab') or (compute_mode=='google_cloud'):
	callbacks.append(colab_TensorBoard(log_dir=model_saver.tensorboard_log_dir, write_images=True, histogram_freq = 1, write_grads = True, write_graph = True, batch_size = validation_batch_generator_params['batch_size'] ))
if compute_mode == 'local_dev':
    callbacks.append(TensorBoard(log_dir=model_saver.tensorboard_log_dir + '/' + 'tensorboard', 
                                                 histogram_freq=0, 
                                                 batch_size=batch_generator_params['batch_size'], 
                                                 write_graph=True, 
                                                 write_grads=False, 
                                                 write_images=False, 
                                                 embeddings_freq=0, 
                                                 embeddings_layer_names=None, 
                                                 embeddings_metadata=None, 
                                                 embeddings_data=None, 
                                                 update_freq='epoch')   )
    


#callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=1.0e-16))
#push_notification = auto_dl.send_notification('Training')


model.compile(optimizer= Adam(lr=model_params['learning_rate']), loss='mean_squared_error')

if overfit_training == True:
  model.fit(x=sample_batch[0], y=sample_batch[1], epochs = model_params['epochs']*1,  callbacks = callbacks, verbose=2, validation_split = .2 )
else:
  history = model.fit_generator(generator=train_generator, validation_data = validation_generator, epochs = model_params['epochs'], initial_epoch = model_saver.initial_epoch, callbacks = callbacks, use_multiprocessing = True, workers=12, max_queue_size=5, verbose=1  )

#batch = train_generator.__getitem__(0)
#custom_utils.show_batch(batch, batch_generator_params)

#history = model.fit(x = batch[0], y = batch[1], epochs = 200,  batch_size=128, validation_split = .1, verbose=1  )

# In[8]:


import matplotlib.pyplot as plt



# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


