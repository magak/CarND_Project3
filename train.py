#path = 'my_data'
path = 'data/data'
#delimeter = '\\'
delimeter = '/'

def getImagesFromLines(line):
    
    centerImgPath = line[0]
    filename = centerImgPath.split(delimeter)[-1]
    center_path = '../'+path+'/IMG/' + filename
    centerImg = cv2.imread(center_path)

    leftImgPath = line[1]
    filename = leftImgPath.split(delimeter)[-1]
    left_path = '../'+path+'/IMG/' + filename
    leftImg = cv2.imread(left_path)

    rightImgPath = line[2]
    filename = rightImgPath.split(delimeter)[-1]
    right_path = '../'+path+'/IMG/' + filename
    rightImg = cv2.imread(right_path)

    return centerImg, leftImg, rightImg

import csv
import cv2
import numpy as np
from sklearn.utils import shuffle

lines = []
with open('../'+path+'/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    # skip the header
    next(reader, None)
    for line in reader:
        lines.append(line)
    
images = []
measurements = []
# correction for left and right camera
correction = 0.2
for line in lines:
    centerImg, leftImg, rightImg = getImagesFromLines(line)            
    measurement = float(line[3])    
    
    images.append(centerImg)
    measurements.append(measurement)

    images.append(leftImg)
    measurements.append(measurement+correction)

    images.append(rightImg)
    measurements.append(measurement-correction)

####################################
# augmenting by flipping
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)
####################################

X_train = np.array(images)
y_train = np.array(measurements)

#Shuffle the training data
X_train, y_train = shuffle(X_train, y_train)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))

model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))

model.add(Flatten())

model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer ='adam')
#print(X_train[6])
#print(current_path)
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)
model.save('model.h5')