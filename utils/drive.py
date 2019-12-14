#parsing command line arguments
import argparse
#decoding camera images
import base64
#for frametimestamp saving
from datetime import datetime
#reading and writing files
import os
#high level file operations
import shutil
#matrix math
import numpy as np
#real-time server
import socketio
#concurrent networking 
import eventlet
#web server gateway interface
import eventlet.wsgi
#image manipulation
from PIL import Image
#web framework
from flask import Flask
#input output
from io import BytesIO

#load our saved model
from keras.models import load_model

import time

import cv2
#helper class
#import utils

#initialize our server
sio = socketio.Server()
#our flask (web) app
app = Flask(__name__)
#init our model and image array as empty
model = None
prev_image_array = None

#set min/max speed for our autonomous car
MAX_SPEED = 30
MIN_SPEED = 5

STEERING_MAX = 1
STEERING_MIN = -1
#and a speed limit
speed_limit = MAX_SPEED

scaled_speed_mode = False

################################################

class SimplePIController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.
        self.old_time = time.time()
        self.old_error = 0
        self.first_update = True


    def set_desired(self, desired):
        self.set_point = desired


    def update(self, measurement):      
        # proportional error
        self.new_error = self.set_point - measurement
        #print('new error:', self.new_error)
        proportional_correction = self.Kp * self.new_error
        #print('p correction:', proportional_correction)

        differential_correction = 0
        
        # derivative error
        if self.first_update:
            self.first_update = False            
        else:
            new_time = time.time()
            differential_correction = (self.new_error - self.old_error) / (new_time - self.old_time)
            differential_correction = self.Kd * differential_correction
        
        #print('D correction:>>>>>', differential_correction)
        
        self.old_error = self.error
        self.old_time = new_time  
        
        # integral error
        self.integral += self.new_error
        self.integral_correction = self.Ki * self.integral

        output =   proportional_correction + differential_correction
        
#        if output < 0:
#            output = 0
        return output
        #return 25
    
    
controller = SimplePIController(.01,0,.01)
set_speed = 25
controller.set_desired(set_speed)


def get_scaled_desired_speed(steering_angle):
    deviation_from_center = abs(steering_angle)
    percent_deviation_in_steering = deviation_from_center/ ((STEERING_MAX - STEERING_MIN)/2)
    desired_speed_percent = 1 - percent_deviation_in_steering
    scaled_speed = desired_speed_percent * (MAX_SPEED - MIN_SPEED) + MIN_SPEED
    return scaled_speed
 
    
####################################################

#registering event handler for the server
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car, how hard to push peddle
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image_left_front = Image.open(BytesIO(base64.b64decode(data["image_front_left"])))
        image_right_front = Image.open(BytesIO(base64.b64decode(data["image_front_right"])))

        try:
            image = np.asarray(image)       # from PIL image to numpy array
            image_left_front = np.asarray(image_left_front)       # from PIL image to numpy array
            image_right_front = np.asarray(image_right_front)       # from PIL image to numpy array
            
            #expand dims as we have 1 different networks expecting batch,width,height,channel shape.
            #batch_size = 1 in model.predict doesnt handle this. Hence batch = 1 has to be done for
            #every sub network
            image = np.expand_dims(image, axis = 0)
            image_left_front = np.expand_dims(image_left_front, axis = 0)       # from PIL image to numpy array
            image_right_front = np.expand_dims(image_right_front, axis = 0)      # from PIL image to numpy array

            '''
            from matplotlib import pyplot as plt
            plt.imshow(image_left_front)
            plt.show()
            '''

            #steering_angle = float(model.predict([image, image_left_front, image_right_front], batch_size=1))
            #steering_angle = steering_angle[0]
            #image = utils.preprocess(image) # apply the preprocessing
            #image = np.array([image])       # the model expects 4D array

            # predict the steering angle for the image
            steering_angle = float(model.predict([image, image_left_front, image_right_front], batch_size=1))
            # lower the throttle as the speed increases
            # if the speed is above the current speed limit, we are on a downhill.
            # make sure we slow down first and then go back to the original max speed.
            #steering_angle = -.5
            '''
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # slow down
            else:
                speed_limit = MAX_SPEED
            throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2
            '''
#            if scaled_speed_mode:
#                desired_speed = get_scaled_desired_speed(steering_angle)
#                controller.set_desired(desired_speed)
            throttle = controller.update(speed)
            #steering_angle = steering_angle - 1
            #throttle = 0
            print('{}      {}      {}'.format(steering_angle, throttle, speed))
            send_control(steering_angle, throttle)
        except Exception as e:
            print(e)

        # save frame
        '''
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
        '''
    else:
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    
    default_model = '/home/pt/Documents/Results/deep_learning/01_Self_Driving_Car_Nvidia_Paper/saved_models/2019-2-13-10-0/weights-20-0.05.h5'
    default_image_folder = '/home/pt/Desktop/fake_google_drive/deep_learning/run'
    
    parser.add_argument(
        '--model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.',
        default=default_model
    )
    parser.add_argument(
        '--image_folder',
        type=str,
        nargs='?',
        help='Path to image folder. This is where the images from the run will be saved.',
        default=default_image_folder,

    )
    args = parser.parse_args()



    #load model
    model = load_model(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
