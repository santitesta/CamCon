#!/usr/bin/env python
"""
Raspberry Pi Live Image Inference
Continuously captures image from Raspberry Pi Camera module and perform 
inference using provided .eim model file. Outputs probabilities in console.
Author: EdgeImpulse, Inc.
Date: June 8, 2021
License: Apache-2.0 (apache.org/licenses/LICENSE-2.0)
"""

import os, sys, time
import cv2
import numpy as np
from picamera import PiCamera
from picamera.array import PiRGBArray
from edge_impulse_linux.runner import ImpulseRunner

##----- Added for filtering -----##

import matplotlib.pyplot as plt
import random, os, PIL, json, time, hmac, hashlib, requests, threading, queue

from scipy import ndimage as ndi

from skimage.util import random_noise
from skimage import feature
from skimage.transform import resize                      # Used to scale/resize image arrays

##-------------------------------##
# Settings
model_file = "modelfile.eim"            # Trained ML model from Edge Impulse
draw_fps = True                         # Draw FPS on screen
res_width = 96                          # Resolution of camera (width)
res_height = 96                         # Resolution of camera (height)
rotation = 0                            # Camera rotation (0, 90, 180, or 270)
img_width = 28                          # Resize width to this for inference
img_height = 28                         # Resize height to this for inference

# The ImpulseRunner module will attempt to load files relative to its location,
# so we make it load files relative to this program instead
dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(dir_path, model_file)

# Load the model file
runner = ImpulseRunner(model_path)

# Initialize model
try:

    # Print model information
    model_info = runner.init()
    print("Model name:", model_info['project']['name'])
    print("Model owner:", model_info['project']['owner'])
    
# Exit if we cannot initialize the model
except Exception as e:
    print("ERROR: Could not initialize model")
    print("Exception:", e)
    if (runner):
            runner.stop()
    sys.exit(1)
    
# Initial framerate value
fps = 0

# Start the camera
with PiCamera() as camera:
    
    # Configure camera settings
    camera.resolution = (res_width, res_height)
    camera.rotation = rotation
    
    # Container for our frames
    raw_capture = PiRGBArray(camera, size=(res_width, res_height))
    
    # Continuously capture frames (this is our while loop)
    for frame in camera.capture_continuous(raw_capture, 
                                            format='bgr', 
                                            use_video_port=True):
                                            
        # Get timestamp for calculating actual framerate
        timestamp = cv2.getTickCount()
        
        # Get Numpy array that represents the image
        img_array = frame.array
        
##------------------- Added By JE --------------------##        
        
        # Filtering test 
              
        #print(file_path)
        blurred = cv2.GaussianBlur(img_array, (3, 3), 3)
        edges = feature.canny(blurred, sigma=2)

        # Crop image to the area of interest
          # Geting axis weigth distribution        
        y_dim=np.sum(edges,axis=1)        
        y_axis=np.ones(len(y_dim))        
        #print(y_axis)
        i=0
        for i in range(len(y_dim)):
          y_axis[i]=i
         # Calculing where the area of interest is in y
        y_center=np.argmax(y_dim)              
         # Cropping Image
        TARGET_H_UP = 20
        TARGET_H_DOWN = 20
        #print(y_center)

        im1 = edges[(y_center-TARGET_H_DOWN):(y_center+TARGET_H_UP),:]


          # Calculing where the area of interest is in x
        x_dim=np.sum(im1,axis=0)
        x_axis=np.ones(len(x_dim)) 
        i=0
        Edge_Flag=0
        for i in range(len(x_dim)):
          x_axis[i]=i
          if (x_dim[i]>1) and (Edge_Flag==0):
            Edge_Flag=1
            Edge_Index=i
        #print("x_dim:")
        #print(x_dim)
        x_center=Edge_Index  
         # Cropping Image
        TARGET_W_LEFT = 20
        TARGET_W_RIGTH = 350
        #print("x_center:")
        #print(x_center)
        a=(x_center-TARGET_W_LEFT)
        b=(x_center+TARGET_W_RIGTH)
        im2 = im1[:,a:b]

        
           
        # show the images ## WORKS INSIDE WHILE LOOP????      !!!!!!!!!!!!!!!!!!!!!!!!!!
        y_dim = y_dim[::-1] #Invert array to be able to compare axis sum
                            #with the photograph.
        plt.figure(figsize=(30,20))
        plt.subplot(141), plt.imshow(img_array, cmap = 'gray')
        plt.title('Original Image')
        plt.subplot(142), plt.imshow(edges, cmap = 'gray')
        plt.title('Canny filter output (edges)')
      
                  
        plt.figure(figsize=(30,20))
        plt.subplot(143), plt.imshow(im1, cmap = 'gray')
        plt.title('Cropped on heigth')
        plt.subplot(144), plt.imshow(im2, cmap = 'gray')#plt.plot(x_axis,x_dim)
        plt.title('Cropped on width')

##-------------- End of added By JE ------------------##

        # Resize captured image
        #img_resize = cv2.resize(img, (img_width, img_height))

        # Convert image to grayscale
        #img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)

        # Convert image to 1D vector of floating point numbers
        features = np.reshape(im2, (img_width * img_height)) / 255

        # Edge Impulse model expects features in list format
        features = features.tolist()
        
        # Perform inference
        res = None
        try:
            res = runner.classify(features)
        except Exception as e:
            print("ERROR: Could not perform inference")
            print("Exception:", e)
            
        # Display predictions and timing data
        print("Output:", res)
        
        # Display prediction on preview
        if res is not None:
        
            # Find label with the highest probability
            predictions = res['result']['classification']
            max_label = ""
            max_val = 0
            for p in predictions:
                if predictions[p] > max_val:
                    max_val = predictions[p]
                    max_label = p
                    
            # Draw predicted label on bottom of preview
            cv2.putText(img_array,
                        max_label,
                        (0, res_height - 20),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (255, 255, 255))
                        
            # Draw predicted class's confidence score (probability)
            cv2.putText(img_array,
                        str(round(max_val, 2)),
                        (0, res_height - 2),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (255, 255, 255))
        
        # Draw framerate on frame
        if draw_fps:
            cv2.putText(img_array, 
                        "FPS: " + str(round(fps, 2)), 
                        (0, 12),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (255, 255, 255))
        
        # Show the frame
        cv2.imshow("Frame", img_array)
        
        # Clear the stream to prepare for next frame
        raw_capture.truncate(0)
        
        # Calculate framrate
        frame_time = (cv2.getTickCount() - timestamp) / cv2.getTickFrequency()
        fps = 1 / frame_time
        
        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break
            
# Clean up
cv2.destroyAllWindows()