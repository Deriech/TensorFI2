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
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
 #go throgh all others labels and store images into np array
for label in range(0, 43):
    label_file = "{:05}".format(label)
    filelist = glob.glob(TRAINING_FILE_LOCATION+str(label_file)+'/*.ppm')
    x, y = get_image_label_resize(label, filelist)
    g = cv2.cvtColor(x[0], cv2.COLOR_RGB2GRAY)
    g_he = clahe.apply(g)
    plt.subplot(121), plt.imshow(g_he, cmap='gray'), plt.axis("off"), plt.title("Grayscale")
    plt.subplot(122), plt.hist(g_he.ravel(),color='k'), plt.title("Gray Histogram")
    plt.savefig("./histogram_equalized/label_"+str(label))
