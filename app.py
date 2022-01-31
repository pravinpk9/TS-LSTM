#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import tensorflow
from pickle import load

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify


# In[9]:


# Load saved models
MODEL_PATH = 'pred_model.h5'
SCALER_PATH = 'scale_model.pkl'

model = tensorflow.keras.models.load_model(MODEL_PATH)
scaler = load(open(SCALER_PATH, 'rb'))

#print('Loading all saved models completed')


# In[10]:


# Define a flask app
app = Flask(__name__)

@app.route('/', methods=['GET'])
def model_predict():
    return "1"
    val1 = request.args['values']
    empty_array = np.array([])
    for i in val1.split(','):
        empty_array= np.append(empty_array, np.array([float(i)]))
    
    pred = np.array([empty_array])
    pred = np.reshape(pred, (pred.shape[0],1, pred.shape[1]))
    res = model.predict(pred)
    return jsonify(prediction = str(scaler.inverse_transform(res)))

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)


# In[ ]:




