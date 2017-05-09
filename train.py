import csv
import cv2
import numpy as np


#path = 'my_data'
path = 'data/data'
#delimeter = '\\'
delimeter = '/'

lines = []
with open('../'+path+'/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    # skip the header
    next(reader, None)
    for line in reader:
        lines.append(line)
    
images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split(delimeter)[-1]
    current_path = '../'+path+'/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

print(current_path)
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer ='adam')
#print(X_train[6])
#print(current_path)
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=1)
model.save('model.h5')