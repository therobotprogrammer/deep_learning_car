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
MAX_SPEED = 8
MIN_SPEED = 3

#and a speed limit
speed_limit = MAX_SPEED


################################################

class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral
    
    
controller = SimplePIController(0.01, 0)
set_speed = 15
controller.set_desired(set_speed)


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

            #image = utils.preprocess(image) # apply the preprocessing
            #image = np.array([image])       # the model expects 4D array

            # predict the steering angle for the image
            steering_angle = float(model.predict([image, image_left_front, image_right_front], batch_size=1))
            # lower the throttle as the speed increases
            # if the speed is above the current speed limit, we are on a downhill.
            # make sure we slow down first and then go back to the original max speed.
            '''
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # slow down
            else:
                speed_limit = MAX_SPEED
            throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2
            '''
            throttle = controller.update(float(7))

            #throttle = 0
            print('{} {} {}'.format(steering_angle, throttle, speed))
            send_control(steering_angle, throttle)
        except Exception as e:
            print(e)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
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
    
    default_model = '/home/pt/Desktop/fake_google_drive/deep_learning/01_Self_Driving_Car_Nvidia_Paper/saved_models/2018-7-14-22-6/weights-46-0.01.h5'
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