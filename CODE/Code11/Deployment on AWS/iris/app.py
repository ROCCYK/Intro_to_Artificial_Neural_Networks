# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 11:45:03 2021

@author: noopa
"""


import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template
from collections.abc import Mapping

app=Flask(__name__)
pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = classifier.predict(final_features)
    output = prediction[0]

    
    return render_template('index.html', prediction_text='The flower belong to species {}'.format(output))
    
#Run Locally
#if __name__=='__main__':
#    app.run(debug=True)
if __name__=='__main__':
    app.run(host = '0.0.0.0', port=8080)