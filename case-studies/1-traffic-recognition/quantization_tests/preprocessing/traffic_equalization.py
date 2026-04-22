import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src import tensorfi2 as tfi
import time, sys


TRAINING_FILE_LOCATION = 'GTSRB/Final_Training/Images/'
TEST_FILE_LOCATION = "GTSRB/Final_Test/Images/"


# load data from pickle
trainx = np.load('./Data/trainx_processed.npy', allow_pickle=True)
trainy = np.load('./Data/trainy.npy', allow_pickle=True)

# load data from pickle
X_test = np.load('./Data/testx_processed.npy', allow_pickle=True)
y_test = np.load('./Data/testy.npy', allow_pickle=True)

# shuffle training data and split them into training and validation
indices = np.random.permutation(trainx.shape[0])
# 20% to val
split_idx = int(trainx.shape[0]*0.8)
train_idx, val_idx = indices[:split_idx], indices[split_idx:]
X_train, X_validation = trainx[train_idx,:], trainx[val_idx,:]
y_train, y_validation = trainy[train_idx], trainy[val_idx]

# get overall stat of the whole dataset
n_train = X_train.shape[0]
n_validation = X_validation.shape[0]
n_test = X_test.shape[0]
image_shape = X_train[0].shape
n_classes = len(np.unique(y_train))
print("There are {} training examples ".format(n_train))
print("There are {} validation examples".format(n_validation))
print("There are {} testing examples".format(n_test))
print("Image data shape is {}".format(image_shape))
print("There are {} classes".format(n_classes))



# Normalize data
X_train_normalized_gry = (X_train-128)/128
X_validation_normalized_gry = (X_validation-128)/128
X_test_normalized_gry = (X_test-128)/128


# update the train, val and test data with normalized gray images
X_train = X_train_normalized_gry
X_validation = X_validation_normalized_gry
X_test = X_test_normalized_gry

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

model = models.Sequential()
# Conv 32x32x1 => 28x28x6.
model.add(layers.Conv2D(filters = 6, kernel_size = (5, 5), strides=(1, 1), padding='valid', 
                        activation='relu', data_format = 'channels_last', input_shape = (32, 32, 1)))
# Maxpool 28x28x6 => 14x14x6
model.add(layers.MaxPooling2D((2, 2)))
# Conv 14x14x6 => 10x10x16
model.add(layers.Conv2D(16, (5, 5), activation='relu'))
# Maxpool 10x10x16 => 5x5x16
model.add(layers.MaxPooling2D((2, 2)))
# Flatten 5x5x16 => 400
model.add(layers.Flatten())
# Fully connected 400 => 120
model.add(layers.Dense(120, activation='relu'))
# Fully connected 120 => 84
model.add(layers.Dense(84, activation='relu'))
# Dropout
model.add(layers.Dropout(0.2))
# Fully connected, output layer 84 => 43
model.add(layers.Dense(43, activation='softmax'))

# specify optimizer, loss function and metric
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# training batch_size=128, epochs=10
conv = model.fit(X_train, y_train, batch_size=128, epochs=10,
        validation_data=(X_validation, y_validation))

model.save_weights('h5/traffic-trained.weights.h5')

model.load_weights('h5/traffic-trained.weights.h5')

tesX = []
for i in range(43):
	tesX.append([])

for i in range(len(X_test)):
	loss, acc = model.evaluate(X_test[i:i+1], y_test[i:i+1], verbose=0)
	if(acc == 1.):
		tesX[int(y_test[i:i+1])].append(i)

with open("Data/tesX.txt", "wb") as fp:
	pickle.dump(tesX, fp)


with open("Data/tesX.txt", "rb") as fp:
    tesX = pickle.load(fp)

for i in range(43):
    tesX[i] = tesX[i][:30]

countX = []

for i in range(43):
    countX.append(0.)

start = time.time()

conf = sys.argv[1]
filePath = sys.argv[2]
filePath = os.path.join(filePath, "res.csv")

f = open(filePath, "w")
numFaults = int(sys.argv[3])
print('numfaults =', numFaults)
for k in range(numFaults):
    print("Trial",k+1,"of",numFaults)
    for i in range(43):
        count = 0.
        tesXi = tesX[i]
        for j in range(30):
            res = tfi.inject(model=model, x_test=X_test[tesXi[j:j+1]], confFile=conf)
            if (res.res == i):
                count = count + 1.
            else:
                continue
        countX[i] = countX[i] + count

for i in range(43):
    countX[i] = countX[i]/numFaults

f.write(str(countX))
f.write("\n")
f.write("Time for %d injections: %f seconds" % (numFaults, time.time() - start))
f.close()
