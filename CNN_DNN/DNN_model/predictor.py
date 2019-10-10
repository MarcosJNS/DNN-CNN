# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 10:50:03 2019

@author: marcos
"""




from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib
import collections
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.python.saved_model import tag_constants



model_dir = os.path.join("c:\\","Users\marcos\Documents\DNN_model\model_info")

BATCH_SIZE=10

ACTIONS = ['NaN', 'Place', 'Remove']

def predict_input_fn(features, labels, batch_size):
    features = dict(features)
    inputs = (features, labels) if labels is not None else features
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.batch(batch_size)
    return dataset

y_name = '100' 
COLUMN_NAMES=[]
for i in range(101):
  
  if i<101:     
     COLUMN_NAMES.append(str(i))
  else :
     COLUMN_NAMES.append('action')

TRAINING = "Action_training.csv"     
     
train = pd.read_csv(TRAINING, names=COLUMN_NAMES, header=0)
train_x, train_y = train, train.pop(y_name)     
print('heyyy',train_x)  
# Specify that all features have real-value data
#  feature_columns = [tf.contrib.layers.real_valued_column("", dimension=101)]
feature_columns = [tf.feature_column.numeric_column(key=key)
                   for key in train_x.keys()]  
  
classifier =tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                              hidden_units=[101,202,202,404,404,202,202,101],model_dir=model_dir,
                                              n_classes=3)
  
  

  # Build 3 layer DNN
  
  

        
csv_main=os.path.join("predict.csv")
f_track=open(csv_main, 'r')
predict_sheet=pd.read_csv(f_track, header=None)         
    
    
predict_x  = {}       
for i in range(100):
        main_row=predict_sheet.values[:,i]
    
        write_dict={str(i) : main_row}
        predict_x.update(write_dict)
print(main_row)    
predictions =classifier.predict(
        input_fn=lambda: predict_input_fn(predict_x, labels=None,batch_size=BATCH_SIZE))
    
for prediction in zip(predictions):
        class_id = prediction[0]
        class_ID=class_id["classes"]
        probability = class_id["probabilities"]
        print(probability)
        class_ID=str(class_ID)
        print(class_ID)
        