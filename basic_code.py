import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from keras import preprocessing
from tensorflow import keras
import glob
import cv2
import os
import json
import h5py
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split

def convert_img(image_path, target_size=(128,128)):
    img_arr_img = []
    image = cv2.imread(image_path)
    image = cv2.resize(image, dsize=target_size)#interpolation=cv2.INTER_CUBIC) #/255.0

    img_arr_img.append(image)
    return np.asarray(img_arr_img)

reconstructed_model = keras.models.load_model("../AI-diseases-prediction-master/pretrained models/cnn_malaria.h5")

image_path = '../AI-diseases-prediction-master/cell_images/Uninfected/C1_thinF_IMG_20150604_104722_cell_73.png'
#image_path = '../AI-diseases-prediction-master/cell_images/Parasitized/C33P1thinF_IMG_20150619_115808a_cell_205.png'
Xtest = convert_img(image_path)
image_result = reconstructed_model.predict(Xtest)
if(image_result > 0.5):
    print("Infected")
else:
    print("Not infected")
