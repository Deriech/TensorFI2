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

testfile = pd.read_csv(TEST_FILE_LOCATION +'GT-final_test.test.csv', sep=";")['Filename'].apply(lambda x: TEST_FILE_LOCATION + x).tolist()
X_test = np.array([cv2.resize(cv2.imread(fname), (32, 32), interpolation = cv2.INTER_AREA) for fname in testfile])

trainx.dump('./Data/trainx.npy')
trainy.dump('./Data/trainy.npy')
X_test.dump('./Data/testx.npy')

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
X_train_processed = np.array([clahe.apply(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)) for image in trainx])
X_test_processed = np.array([clahe.apply(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)) for image in X_test])
y_test = np.array(pd.read_csv('GTSRB/GT-final_test.csv', sep=";")['ClassId'])

X_train_processed.dump('./Data/trainx_processed.npy')
X_test_processed.dump('./Data/testx_processed.npy')

