from tempfile import TemporaryDirectory
from flask.globals import request
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
from flask import Flask
from flask import request, redirect,render_template
from werkzeug.utils import secure_filename
import tempfile
tempdirectory = tempfile.gettempdir()

def convert_img(image_path, target_size=(128,128)):
    img_arr_img = []
    image = cv2.imread(image_path)
    image = cv2.resize(image, dsize=target_size)#interpolation=cv2.INTER_CUBIC) #/255.0

    img_arr_img.append(image)
    return np.asarray(img_arr_img)


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    errors = []
    results = ''
    methodSelected = request.form.get('methodSelected')
    image_path = ''
    if request.method == "POST":
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join(tempdirectory, filename))
        image_path = os.path.join(tempdirectory, filename)
            # os.remove(filename)
            # file=open(os.path.join(tempdirectory, filename), encoding="utf-8")
            # textImport = file.read()
        
        
        if methodSelected == 'malaria':
            reconstructed_model = keras.models.load_model("../AI-diseases-prediction-master/pretrained models/cnn_malaria.h5")
            #image_path = '../AI-diseases-prediction-master/cell_images/Uninfected/C1_thinF_IMG_20150604_104722_cell_73.png'
            #image_path = '../AI-diseases-prediction-master/cell_images/Parasitized/C33P1thinF_IMG_20150619_115808a_cell_205.png'
            Xtest = convert_img(image_path)
            image_result = reconstructed_model.predict(Xtest)
            if(image_result > 0.5):
                results = "Your report says you have diagnosed with Malaria. Hey, No need to worry because, Malaria is a preventable and treatable disease. The primary objective of treatment is to ensure complete cure, that is the rapid and full elimination of the Plasmodium parasite from the patient’s blood, in order to prevent progression of uncomplicated malaria to severe disease or death, and to chronic infection that leads to malaria-related anaemia. From a public health perspective, treatment is meant to reduce transmission of the infection to others, by reducing the infectious reservoir and by preventing the emergence and spread of resistance to antimalarial medicines. Follow this Treatment table given in the link to get rid off malaria quickly.👨‍⚕️         https://www.cdc.gov/malaria/resources/pdf/Malaria_Treatment_Table.pdf "
            else:
                results = "Not infected"
                

        if methodSelected == 'xray':
            print("Still under development")
                #############################################################################
            ###
            
            ###
      

    return render_template('index.html', results = results)


