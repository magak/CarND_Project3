import preproc as prc
#path = 'my_data'
#path = 'data/data'
path = 'my_data/1205'
delimeter = '\\'
#delimeter = '/'

def getImagesFromLines(line):
    
    centerImgPath = line[0]
    
    #print(centerImgPath)
    
    filename = centerImgPath.split(delimeter)[-1]
    center_path = '../'+path+'/IMG/' + filename
    centerImg = prc.preproc(cv2.imread(center_path))

    leftImgPath = line[1]
    filename = leftImgPath.split(delimeter)[-1]
    left_path = '../'+path+'/IMG/' + filename
    leftImg = prc.preproc(cv2.imread(left_path))

    rightImgPath = line[2]
    filename = rightImgPath.split(delimeter)[-1]
    right_path = '../'+path+'/IMG/' + filename
    rightImg = prc.preproc(cv2.imread(right_path))

    return centerImg, leftImg, rightImg

import csv
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle


#############################################################################################################

#lines = []
#with open('../'+path+'/driving_log.csv') as csvfile:
#    reader = csv.reader(csvfile)
#    # skip the header
#    next(reader, None)
#    for line in reader:
#        lines.append(line)
#    
#images = []
#measurements = []
## correction for left and right camera
#correction = 0.5
#for line in lines:
#    centerImg, leftImg, rightImg = getImagesFromLines(line)            
#    measurement = float(line[3])    
#    
#    images.append(centerImg)
#    measurements.append(measurement)
#
#    images.append(leftImg)
#    measurements.append(measurement+correction)
#
#    images.append(rightImg)
#    measurements.append(measurement-correction)
#
#####################################
## augmenting by flipping
#augmented_images, augmented_measurements = [], []
#for image, measurement in zip(images, measurements):
#    augmented_images.append(image)
#    augmented_measurements.append(measurement)
#    augmented_images.append(cv2.flip(image,1))
#    augmented_measurements.append(measurement*-1.0)
######################################
#
#X_train = np.array(images)
#y_train = np.array(measurements)
#
##Shuffle the training data
#X_train, y_train = shuffle(X_train, y_train)

#############################################################################################################



#############################################################################################################

lines = []
with open('../'+path+'/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    # skip the header
    next(reader, None)
    for line in reader:
        lines.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def generator(samples, batch_size=32):
    correction = 0.4
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            augmented_images = []
            augmented_measurements = []
            for batch_sample in batch_samples:
                
                centerImg, leftImg, rightImg = getImagesFromLines(batch_sample)            
                measurement = float(batch_sample[3])
    
                images.append(centerImg)
                angles.append(measurement)

                images.append(leftImg)
                angles.append(measurement+correction)

                images.append(rightImg)
                angles.append(measurement-correction)
                
                # augmenting by flipping
                augmented_images, augmented_measurements = [], []
                for image, measurement in zip(images, angles):
                    augmented_images.append(image)
                    augmented_measurements.append(measurement)
                    augmented_images.append(cv2.flip(image,1))
                    augmented_measurements.append(measurement*-1.0)

            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)

#############################################################################################################



from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(60,5,5,subsample=(2,2),activation="relu"))

model.add(Convolution2D(76,3,3,activation="relu"))
model.add(Convolution2D(76,3,3,activation="relu"))

model.add(Flatten())

model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer ='adam')
#print(X_train[6])
print(len(train_samples))
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=4)
model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*6 , validation_data=validation_generator,nb_val_samples=len(validation_samples)*6, nb_epoch=2)
model.save('model.h5')