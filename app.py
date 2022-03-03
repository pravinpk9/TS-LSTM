#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow
from pickle import load

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify


# In[2]:


# Load saved models
MODEL_PATH = 'pred_model.h5'
SCALER_PATH = 'scale_model.pkl'

model = tensorflow.keras.models.load_model(MODEL_PATH)
scaler = load(open(SCALER_PATH, 'rb'))

#print('Loading all saved models completed')


# In[43]:


# Define a flask app
app = Flask(__name__)

@app.route('/', methods=['GET'])
def model_predict():
    #return "1"
    val1 = request.args['values']
    days = request.args['days']
    
    print(val1)
    print(days)
#     empty_array = np.array([])
#     for i in val1.split(','):
#         print(scaler.transform(np.array([float(i)])))
#         j = scaler.transform(np.array([float(i)]))
#         empty_array= np.append(empty_array, np.array([float(j)]))
    
#     pred = np.array([empty_array])
#     pred = np.reshape(pred, (pred.shape[0],1, pred.shape[1]))
#     res = model.predict(pred)
#     print(res)
#     print(res[0])
#     for x in res[0]:
#         print(x)
        
        
    pred_result = []
    empty_array = np.array([])
    for i in val1.split(','):
        print(scaler.transform(np.array([[float(i)]])))
        j = scaler.transform(np.array([[float(i)]]))
        empty_array= np.append(empty_array, j)

    for index in range(int(days)):
        print(empty_array[-4:])
        pred = np.array([empty_array[-4:]])
        pred = np.reshape(pred, (pred.shape[0], 1, pred.shape[1]))

        pred_values = model.predict(pred)
        print(pred_values)
        empty_array= np.append(empty_array, pred_values)
        for e in pred_values[0]:
            pred_result.append(e)
        print('---------')
    
    return str(pred_result)
    return jsonify(prediction = str(scaler.inverse_transform([pred_result])))

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)


# In[ ]:




