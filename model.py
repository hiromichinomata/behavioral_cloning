import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, \
    Dropout, BatchNormalization
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

def add_bias(i, steering_angle):
    bias = 0.3
    if i ==0:
        return steering_angle
    elif i ==1:
        return steering_angle + bias
    else:
        return steering_angle - bias


lines = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Get teacher data from reverse drive too
with open('./reverse_driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

with open('./course2.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Get teacher data from reverse drive too
with open('./course2_reverse.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


images = []
measurements = []
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = './IMG/' + filename
        image = cv2.imread(current_path)
        measurement = float(line[3])
        biased_measurement = add_bias(i, measurement)
        if biased_measurement != 0:
            images.append(image)
            measurements.append(biased_measurement)

X_train = np.array(images)
y_train = np.array(measurements)


model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

model.add(Convolution2D(6,5,5,activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Convolution2D(6,5,5,activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(120))
model.add(Dropout(0.5))
model.add(Dense(84))
model.add(Dropout(0.5))
model.add(Dense(1))

model.summary()

adam = Adam(lr=0.001)
model.compile(loss='mse', optimizer=adam)

history_object = model.fit(X_train, y_train, validation_split=0.3, shuffle=True, nb_epoch=5)

model.save('model.h5')

import matplotlib.pyplot as plt

print(history_object.history.keys())

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
