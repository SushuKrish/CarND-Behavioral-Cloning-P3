import csv
import cv2
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import sys
from scipy import ndimage

lines = []
with open ('self_data/driving_log.csv')as csvfile:
    reader= csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
for line in lines:
    #process images from the center camera
    source_path_C = line[0]
    filename_C = source_path_C.split('/')[-1]
    current_path_C = 'self_data/IMG/'+ filename_C
    image_C = ndimage.imread(current_path_C)
    #image_C = cv2.imread(current_path_C)
    measurement_C = float(line[3])
    
    #measurement correction value
    correction = 0.2
    #process images from the left camera
    source_path_L = line[1]
    filename_L = source_path_L.split('/')[-1]
    current_path_L = 'self_data/IMG/'+ filename_L
    image_L = ndimage.imread(current_path_L)
    #image_L = cv2.imread(current_path_L)
    measurement_L = float(line[3])+correction
    #process images from the right camera
    source_path_R = line[2]
    filename_R = source_path_R.split('/')[-1]
    current_path_R= 'self_data/IMG/'+ filename_R
    image_R = ndimage.imread(current_path_R)
    #image_R = cv2.imread(current_path_R)
    measurement_R = float(line[3])-correction

    #Extend image and measurement
    # image.extend(image_C, image_L, image_R)
    # measurement.extend(measurement_C, measurement_L, measurement_R)
    
    # Augment multiple camera images
    images.append(image_C)
    measurements.append(measurement_C)
    images.append(image_L)
    measurements.append(measurement_L)
    images.append(image_R)
    measurements.append(measurement_R)

# Flip image and augment
augment_images, augment_measurements = [], []
for image,measurement in zip(images,measurements):
    augment_images.append(image)
    augment_measurements.append(measurement)
    augment_images.append(cv2.flip(image,1))
    augment_measurements.append(measurement*-1.0)
    
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Convolution2D, Cropping2D, MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape= (160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
# Nvidia model
model.add(Convolution2D(24, (5, 5), strides=(2, 2), activation="relu"))
model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation="relu"))
model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation="relu"))
model.add(Convolution2D(64, (3, 3), activation="relu"))
model.add(Convolution2D(64, (3, 3), activation="relu"))
model.add(Dropout(0.8))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Train model
X_train = np.array(augment_images)
y_train = np.array(augment_measurements)

model.compile(optimizer='adam',loss ='mse')
model.fit(X_train, y_train, epochs=7, validation_split=0.2, shuffle=True)
model.save('model.h5')
#exit()