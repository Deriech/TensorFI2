import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import glob
import pickle
import numpy as np
import pandas as pd

from src import tensorfi2 as tfi
import time, sys

IMG_WIDTH, IMG_HEIGHT = (32, 32)
NUM_CLASSES = 43


PRETRAIN = False
if(PRETRAIN):
    TRAINING_FILE_LOCATION = 'GTSRB/Final_Training/Images/'
    TEST_FILE_LOCATION = "GTSRB/Final_Test/Images/"
    im = cv2.imread(TRAINING_FILE_LOCATION + '00000/00000_00000.ppm') 
    print(im.shape)

    #function to read and resize images, get labels and store them into np array
    def get_image_label_resize(label, filelist, dim = (32, 32), dataset = 'Train'):
        x = np.array([cv2.resize(cv2.imread(fname), dim, interpolation = cv2.INTER_AREA) for fname in filelist])
        y = np.array([label] * len(filelist))
            
        print('{} examples loaded for label {}'.format(x.shape[0], label))
        return (x, y)    
        
    #data for label 0. I store them in parent level so that they won't be uploaded to github
    filelist = glob.glob(TRAINING_FILE_LOCATION+'00000'+'/*.ppm')
    trainx, trainy = get_image_label_resize(0, glob.glob(TRAINING_FILE_LOCATION+"00000"+'/*.ppm'))

    #go throgh all others labels and store images into np array
    for label in range(1, 43):
        label_file = "{:05}".format(label)
        filelist = glob.glob(TRAINING_FILE_LOCATION+str(label_file)+'/*.ppm')
        x, y = get_image_label_resize(label, filelist)
        trainx = np.concatenate((trainx ,x))
        trainy = np.concatenate((trainy ,y))

    #save data into a pickle to later use
    trainx.dump('./Data/trainx.npy')
    trainy.dump('./Data/trainy.npy')

# load data from pickle
trainx = np.load('./Data/trainx.npy', allow_pickle=True)
trainy = np.load('./Data/trainy.npy', allow_pickle=True)

if(PRETRAIN):
    #get path for test images
    testfile = pd.read_csv(TEST_FILE_LOCATION +'GT-final_test.test.csv', sep=";")['Filename'].apply(lambda x: TEST_FILE_LOCATION + x).tolist()
    #print(testfile)

    X_test = np.array([cv2.resize(cv2.imread(fname), (32, 32), interpolation = cv2.INTER_AREA) for fname in testfile])
    X_test.dump('./Data/testx.npy')

    y_test = np.array(pd.read_csv('GTSRB/GT-final_test.csv', sep=";")['ClassId'])
    y_test.dump('./Data/testy.npy')

# load data from pickle
X_test = np.load('./Data/testx.npy', allow_pickle=True)
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

# convert the images to grayscale
X_train_gry = np.sum(X_train/3, axis=3, keepdims=True)
X_validation_gry = np.sum(X_validation/3, axis=3, keepdims=True)
X_test_gry = np.sum(X_test/3, axis=3, keepdims=True) 

# Normalize data
X_train_normalized_gry = (X_train_gry-128)/128
X_validation_normalized_gry = (X_validation_gry-128)/128
X_test_normalized_gry = (X_test_gry-128)/128

# update the train, val and test data with normalized gray images
X_train = X_train_normalized_gry
X_validation = X_validation_normalized_gry
X_test = X_test_normalized_gry

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_val = tf.keras.utils.to_categorical(y_validation, NUM_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)
#@title ### Step 5 (New): Build Model for Training from Scratch


# 1. Load the MobileNetV1 base with RANDOM weights
base_model = tf.keras.applications.MobileNet(
    input_shape=(IMG_WIDTH, IMG_HEIGHT, 1),
    alpha=0.25,
    include_top=False,
    weights=None  # KEY CHANGE: Do NOT load pre-trained weights
)

# 2. Add our classification head
# The entire model will be trainable from the start.
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = tf.keras.Model(inputs=base_model.input, outputs=x)

print("Model built to be TRAINED FROM SCRATCH.")
model.summary()
model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

if(PRETRAIN):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=50,
        validation_data=(X_validation, y_val),
        callbacks=[early_stopping]
    )
    model.save_weights('h5/traffic-trained.weights.h5')

model.load_weights('h5/traffic-trained.weights.h5')

_, baseline_model_accuracy = model.evaluate(
    X_test, y_test, verbose=0)

print(baseline_model_accuracy)
tesX = []
if(PRETRAIN):
    for i in range(43):
        tesX.append([])

    for i in range(len(X_test)):
        loss, acc = model.evaluate(X_test[i:i+1], y_test[i:i+1], verbose=0)
        if(acc == 1.):
            tesX[y_test[i:i+1][0].argmax()].append(i)

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
            if (res == i):
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
