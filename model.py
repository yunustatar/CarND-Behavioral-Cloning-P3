import csv
import cv2
import numpy as np
import os
import random
from random import shuffle
import sklearn

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, MaxPooling2D, Activation, Dropout
from keras.optimizers import Adam

import gc
from keras import backend as K


from sklearn.model_selection import train_test_split

global data_folder
data_folder = 'data'

#generator to process the data on the fly
def generator(samples, batch_size=32):
    global data_folder
    num_samples = len(samples)
    curr_dir = os.path.dirname(__file__)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            augmented_images = []
            augmented_measurements = []

            for batch_sample in batch_samples:
                source_path = batch_sample[0]
                filename = source_path.split('/')[-1]
                current_path = os.path.join(curr_dir, data_folder, 'IMG', filename)

                measurement = float(batch_sample[3])
                org_image = cv2.imread(current_path)
                # convert image to RGB. imread uses BGR format, input from simulator is RGB
                image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)

                augmented_images.append(image)
                augmented_measurements.append(measurement)

                # augment data by flipping:
                flipped_image = cv2.flip(image,1)
                augmented_images.append(flipped_image)
                augmented_measurements.append(measurement * -1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

def resize_img(input):
    '''
    resizer function to resize input images to 32x32x3
    '''
    from keras.backend import tf as ktf
    return ktf.image.resize_images(input, (32, 32))

lines = []

curr_dir = os.path.dirname(__file__)

# randomizer will be used for eliminating excessive data with no steering
random.seed(None)

with open(os.path.join(curr_dir, data_folder, 'driving_log.csv')) as csv_file:
    reader = csv.reader(csv_file)

    for line in reader:
        if line[3] == 'steering': # getting rid of first line in Udacity driving_log.csv
            continue

        measurement = float(line[3])

        if abs(measurement) < 0.05:  # assume 0 steering if absolute steering input is less than 0.05
            random_checker = random.uniform(0, 1)

            if random_checker > 0.9: # use only 10% of data with zero steering
                lines.append(line)
            else:
                continue
        else:
            lines.append(line)

# use 20% of the data as validation dataset
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)

# number of epochs to be used
EPOCHS = 3

learning_rate = 0.001
my_ADAM = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# setup the model in Keras
model = Sequential()

# crop from top and bottom (output shape: 65x320x3)
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))

# normalize and zero center
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160-95,320,3)))

# resize for LeNet architecture
model.add(Lambda(resize_img))  # resize to 32x32x3

# first Conv Layer  (kernel : 5x5, depth: 64)
model.add(Conv2D(64, 5, 5, input_shape=(3, 32, 32), border_mode='valid', activation='relu'))

# Max pooling
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))

# second Conv Layer (kernel: 5x5, depth: 32)
model.add(Conv2D(32, 5, 5, input_shape=(64, 14, 14), border_mode='valid', activation='relu'))

# Max pooling
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))

# flatten the data
model.add(Flatten(input_shape=(5, 5, 32)))

# add dropout to overcome overfitting
model.add(Dropout(0.5))

# fully connected layer output: 400
model.add(Dense(400, input_dim=800))
model.add(Activation('relu'))

# fully connected layer output: 200
model.add(Dense(200, input_dim=400))
model.add(Activation('relu'))

# fully connected layer, output: steering angle
model.add(Dense(1))

model.compile(loss='mse', optimizer=my_ADAM)
model.fit_generator(train_generator,
                    samples_per_epoch= len(train_samples),
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), nb_epoch=EPOCHS)


model.save('model.h5')
print('Model saved')

gc.collect()
print('Garbage collected')
K.clear_session()
print('Keras session cleared')
