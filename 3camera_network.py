google_drive = '/home/pt/Desktop/fake_google_drive'


import wget, shutil, os, sys 

# Download AutoDownloader
script_dir = '/home/pt/Desktop/fake_colab/scripts'
if not os.path.isdir(script_dir):
    os.mkdir(script_dir)
    
if os.path.isfile(script_dir + '/AutoDownloader.py'):
    os.remove(script_dir + '/AutoDownloader.py')

AutoDownloader_url = 'https://raw.githubusercontent.com/therobotprogrammer/Auto-Downloader/master/Code/AutoDownloader.py'
wget.download(AutoDownloader_url, out = script_dir + '/AutoDownloader.py')

sys.path.insert(0, script_dir) 

#######################################################################





project_dir = ('/home/pt/Desktop/fake_colab/Deep_Learning')

#    /COMMON_UTILS is a special directory. All its contents are deleted & downloaded for latest copy. 
#    Use it for dependencies
#    All other directorys dont download if data already exists. Use it for training/test data etc
#    All zip files are automatically unzipped & original zip is deleted.

#    To create sharable link for file on Google Drive: https://drive.google.com/uc?export=download&id=YOUR_FILE_ID


#These are relative paths
common_utils_dir = '/COMMON_UTILS'

data_to_download =  {    
    
    '/DATA':                       [
                                        'https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip'
                                   ],


    '/CODE':                      [
                                        'https://github.com/therobotprogrammer/deep_learning_car/archive/master.zip',                                        
                                   ],
    
    '/COMMON_UTILS':              [
                                        
                                   ],
    
                    }




from AutoDownloader import AutoDownloader
auto_dl = AutoDownloader(local_timezone =  'Asia/Kolkata')
auto_dl.initiate(project_dir, data_to_download, recreate_dir = True )
auto_dl.recursively_add_to_path(project_dir + '/CODE') #Warning: Do not add code recursively if you dont trust the source.


# Push Notifications
user_key = "umnxi6yau47cpp9evkn6nfa5zbpipt"
token ="aev81xsojcq2ggdpevsia5rzuw5x12"
auto_dl.setup_pushover_credintials(user_key,token)

#To send push notificaiton:
auto_dl.send_notification('Hello form Colab!!!')

# To see any directory  
#auto_dl.showDirectory(project_dir + '/CODE') 

#To get localtime as string
#auto_dl.get_time_string()



######################################################################
from model_load_and_save import model_load_and_save

model_save_top_directory = google_drive + '/deep_learning/01_Self_Driving_Car_Nvidia_Paper/saved_models' 

save_load_params = {
                            'continue_training_last_model' : True,
                            'create_time_stamped_dirs' : True                                
                   }

model_saver = model_load_and_save(model_save_top_directory, **save_load_params)
###########################




from keras.models import Model
from keras.layers import Input,Dense, Lambda, concatenate, Dropout, Conv2D, Flatten
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
#from sklearn.model_selection import train_test_split #to split out training and testing data 
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard
from keras.optimizers import Adam
from IPython.display import SVG



import custom_utils as custom_utils
from MultiSensorTimeSeriesGenerator import MultiSensorTimeSeriesGenerator


def show_sample_from_generator(generator, batch_generator_params):
    generator = train_generator
    iterator = generator.__iter__()
    batch = next(iterator)
    custom_utils.show_batch(batch, batch_generator_params, figsize=(15, 3))


#Q Why this notmalisation
def image_normalization(x):
    return ( (x/127.5) - 1.0 )
    
#To Do: Verify if Lambda is correctly used
def build_single_sensor_network(sensor_input):    
    x = Lambda(image_normalization)(sensor_input)
    #x = Dense(32)(x)
    return x

def show_model(model):
    SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))


def build_nvidia_paper_model(sensor_count, single_sensor_input_shape):
    
    sensor_inputs = []
    sensor_outputs = []
    
    
    for sensor_id in range(0, sensor_count):
        #Q Will it if we make input_sensor = Input() and then appent list.append(input_sensor) 3 times. will it be same data or different for eact sensor?
        #Q How to do this as an array?
        
        single_sensor_input = Input(shape=single_sensor_input_shape)
        single_sensor_output = build_single_sensor_network(single_sensor_input)
        
        sensor_inputs.append(single_sensor_input)        
        sensor_outputs.append(single_sensor_output)
        

    x = concatenate(sensor_outputs)
    x = Conv2D(24, (5,5), strides=(2, 2), activation='elu')(x)
    x = Conv2D(36, (5,5), strides=(2, 2), activation='elu')(x)
    x = Conv2D(48, (5,5), strides=(2, 2), activation='elu')(x)
    x = Conv2D(64, (5,5), activation='elu')(x)
    x = Conv2D(64, (3,3), activation='elu')(x)
    x = Dropout(.5)(x)
    x = Flatten()(x)
    x = Dense(100, activation = 'elu')(x)
    x = Dense(50, activation = 'elu')(x)
    x = Dense(10, activation = 'elu')(x)
    x = Dense(1, activation = 'elu')(x)
    
    
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
    

    
    

model_params = {
            'train_to_test_split_ratio' : .8,
            'random_seed' : 0,
            'learning_rate': 1.0e-5,
            'epochs': 1000
        }

batch_generator_params = {
             'length' : 1,
             'sampling_rate':1,
             'stride':2,
             'start_index':0,
             'end_index':None,
             'shuffle':False,
             'reverse':False,
             'batch_size':16,
             'image_dimention' : (160,320),
             'n_channels' : 3,
             'time_axis':False
         }



data_dir = project_dir + '/DATA/data'
driving_log_csv = data_dir + '/' + 'driving_log.csv'
driving_log = custom_utils.update_driving_log(data_dir, driving_log_csv)

split_point = int(len(driving_log) * model_params['train_to_test_split_ratio'])
driving_log_train = driving_log[0:split_point]
driving_log_validation = driving_log[split_point + batch_generator_params['length'] * 2:] # length*2 so that car has moved far away from train data spot

#if we done reset index, then test data has index split_point onwards. This will cause problem with timeseriesgenerator
driving_log_train = driving_log_train.reset_index()
driving_log_validation = driving_log_validation.reset_index() 


train_generator = MultiSensorTimeSeriesGenerator([driving_log_train['center'], driving_log_train['left'], driving_log_train['right']], driving_log_train['steering'], **batch_generator_params)

validation_batch_generator_params = batch_generator_params
validation_batch_generator_params['batch_size'] = 16
validation_generator = MultiSensorTimeSeriesGenerator([driving_log_validation['center'], driving_log_validation['left'], driving_log_validation['right']], driving_log_validation['steering'], **validation_batch_generator_params)

#show_sample_from_generator(train_generator, batch_generator_params)

sensor_count, input_shape = get_sensor_count_and_input_shape(train_generator)


if model_saver.continue_training_last_model and model_saver.model_loaded_sucessfully:
    model = model_saver.model
else:
    model = build_nvidia_paper_model(sensor_count, input_shape)


#show_model(model)


#callbacks
save_weights = ModelCheckpoint(filepath=model_saver.model_save_file, monitor='val_loss', verbose =0, save_best_only=True, mode='auto')
csv_logger = CSVLogger(model_saver.csv_save_file, append = True)
#tensorboard = TensorBoard(log_dir=model_saver.tensorboard_log_dir, histogram_freq=0, write_graph=True, write_images=True)
#callbacks = [save_weights, csv_logger, tensorboard]
callbacks = [save_weights, csv_logger]
#push_notification = auto_dl.send_notification('Training')


model.compile(optimizer= Adam(lr=model_params['learning_rate']), loss='mean_squared_error')
model.fit_generator(generator=train_generator, validation_data = validation_generator, epochs = model_params['epochs'], initial_epoch = model_saver.initial_epoch, callbacks = callbacks, use_multiprocessing = False  )






















