import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import glob
import pickle
import numpy as np
import pandas as pd

from src import tensorfi2 as tfi
import time, sys
'''
TRAINING_FILE_LOCATION = '../GTSRB_Dataset/Final_Training/Images/'
TEST_FILE_LOCATION = "../GTSRB_Dataset/Final_Test/Images/"
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
'''
# load data from pickle
trainx = np.load('./Data/trainx.npy', allow_pickle=True)
trainy = np.load('./Data/trainy.npy', allow_pickle=True)
'''
 #get path for test images
testfile = pd.read_csv(TEST_FILE_LOCATION +'GT-final_test.test.csv', sep=";")['Filename'].apply(lambda x: TEST_FILE_LOCATION + x).tolist()
print(testfile)

X_test = np.array([cv2.resize(cv2.imread(fname), (32, 32), interpolation = cv2.INTER_AREA) for fname in testfile])
X_test.dump('./testx.npy')

y_test = np.array(pd.read_csv('../GTSRB_Dataset/GT-final_test.csv', sep=";")['ClassId'])
y_test.dump('./testy.npy')
'''
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
# print("There are {} training examples ".format(n_train))
# print("There are {} validation examples".format(n_validation))
# print("There are {} testing examples".format(n_test))
# print("Image data shape is {}".format(image_shape))
# print("There are {} classes".format(n_classes))

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
from tensorflow import keras
from keras import datasets, layers, models

import tensorflow_model_optimization as tfmot
import tensorflow as tf
quantize_bit_size = 8
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

#training batch_size=128, epochs=10
conv = model.fit(X_train, y_train, batch_size=128, epochs=10,
       validation_data=(X_validation, y_validation))


class CustomQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    def get_weights_and_quantizers(self, layer):
        return [(layer.kernel, tfmot.quantization.keras.quantizers.LastValueQuantizer(
            num_bits=quantize_bit_size, symmetric=True, narrow_range=False, per_axis=False))]

    def get_activations_and_quantizers(self, layer):
        return [(layer.activation, tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
            num_bits=quantize_bit_size, symmetric=True, narrow_range=False, per_axis=False))]

    def set_quantize_weights(self, layer, quantize_weights):
        layer.kernel = quantize_weights[0]

    def set_quantize_activations(self, layer, quantize_activations):
        layer.activation = quantize_activations[0]

    def get_output_quantizers(self, layer):
        return [tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
            num_bits=quantize_bit_size, symmetric=True, narrow_range=False, per_axis=False)]
        

    def get_config(self):
        return {}

#quantize Convolutional and Dense Layers
def annotate_for_quantization(model):
    annotated_layers = []
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
            annotated_layers.append(
                tfmot.quantization.keras.quantize_annotate_layer(
                    layer, quantize_config=CustomQuantizeConfig()
                )
            )
        else:
            annotated_layers.append(layer)
    return tf.keras.Sequential(annotated_layers)

q_aware_model_annotated = annotate_for_quantization(model)

with tfmot.quantization.keras.quantize_scope(
    {'CustomQuantizeConfig':CustomQuantizeConfig,}):
    q_aware_model = tfmot.quantization.keras.quantize_apply(q_aware_model_annotated)

q_aware_model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])




train_images_subset = X_train[0:1000] # out of 60000
train_labels_subset = y_train[0:1000] 
#QAT on model
q_aware_model.fit(train_images_subset, train_labels_subset,
                  batch_size=500, epochs=10, validation_split=0.1) 

_, baseline_model_accuracy = model.evaluate(
    X_test, y_test, verbose=0)

_, q_aware_model_accuracy = q_aware_model.evaluate(
   X_test, y_test, verbose=1)

#check accuracy for original and quantization aware model
print('Baseline test accuracy:', baseline_model_accuracy)
print('Quant test accuracy:', q_aware_model_accuracy)




q_aware_model.save_weights('h5/traffic-trained-quantized-16bit.weights.h5')

q_aware_model.load_weights('h5/traffic-trained-quantized-16bit.weights.h5')


tesX = []
for i in range(43):
	tesX.append([])

for i in range(len(X_test)):
	loss, acc = q_aware_model.evaluate(X_test[i:i+1], y_test[i:i+1], verbose=0)
	if(acc == 1.):
		tesX[int(y_test[i:i+1])].append(i)

with open("Data/tesX_quant_16bit.txt", "wb") as fp:
	pickle.dump(tesX, fp)


with open("Data/tesX_quant_16bit.txt", "rb") as fp:
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
            res = tfi.inject(model=q_aware_model, bit_size=quantize_bit_size, x_test=X_test[tesXi[j:j+1]], confFile=conf, numFaults=numFaults, parallel_num=int(sys.argv[4]), parallel_total=int(sys.argv[5]))
            if (res.res == i):
                count = count + 1.
        countX[i] = countX[i] + count

for i in range(43):
    countX[i] = countX[i]/numFaults

f.write(str(countX))
f.write("\n")
f.write("Time for %d injections: %f seconds" % (numFaults, time.time() - start))
f.close()
