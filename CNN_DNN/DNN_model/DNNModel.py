# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 12:06:45 2019

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

#iris=load_iris()
#
#x,y=iris.data,iris.target
#x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.5,test_size=0.5,random_state=123)



## Data sets
BATCH_SIZE = 10
TRAINING = "Action_training.csv"
TEST = "Action_test.csv"
ACTIONS = ['NaN', 'Place', 'Remove']
model_dir='model_info'

def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset

def eval_input_fn(features, labels, batch_size):
    features = dict(features)
    inputs = (features, labels) if labels is not None else features
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.batch(batch_size)
    return dataset



def main():
  global acc, model_dir, BATCH_SIZE
   #oad datasets.

 
  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=TRAINING,
      target_dtype=np.int,
      features_dtype=np.float32)

  test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=TEST,
      target_dtype=np.int,
      features_dtype=np.float32)
  
  COLUMN_NAMES=[]
  for i in range(101):
  
      if i<101:     
         COLUMN_NAMES.append(str(i))
      else :
         COLUMN_NAMES.append('action')
         
  y_name = '100'       
  
  train = pd.read_csv(TRAINING, names=COLUMN_NAMES, header=0)
  train_x, train_y = train, train.pop(y_name)
 
    
  test = pd.read_csv(TEST, names=COLUMN_NAMES, header=0)
  test_x, test_y = test, test.pop(y_name)
  print('heyyyy',train_x)  
  
  
# Specify that all features have real-value data
#  feature_columns = [tf.contrib.layers.real_valued_column("", dimension=101)]
  feature_columns = [tf.feature_column.numeric_column(key=key)
                   for key in train_x.keys()]
  #model_dir='model_info'
  # Build 3 layer DNN
  
  
  classifier =tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                              hidden_units=[101,202,202,404,404,202,202,101],model_dir=model_dir,
                                              n_classes=3)
  
  
  
  # Define the training inputs
  def get_train_inputs():
    x = tf.constant(training_set.data)
    print(x)
    y = tf.constant(training_set.target)
    print(y)
    return x, y

  # Fit model.
  

  #classifier.fit(input_fn=get_train_inputs, steps=2000)

  # Define the test inputs
  def get_test_inputs():
    x = tf.constant(test_set.data)
    y = tf.constant(test_set.target)

    return x, y


  classifier.train(
    input_fn=lambda: train_input_fn(train_x, train_y, batch_size=BATCH_SIZE),
    steps=1000)
  
  
#  for prediction, expect in zip(predictions, expected):
#    class_id = prediction['class_ids'][0]
#    probability = prediction['probabilities'][class_id]
#    print('Prediction is "{}" ({:.1f}%), expected "{}"'.format(
#        ACTIONS[class_id], 100 * probability, expect))
  
  # Evaluate accuracy.
  
  
  
  eval_result = classifier.evaluate(
    input_fn=lambda: eval_input_fn(test_x, test_y, batch_size=BATCH_SIZE))
  print('Test set accuracy: {accuracy:0.3f}'.format(**eval_result))
  
  feature_columns = [tf.feature_column.numeric_column(key=key)
                   for key in train_x.keys()]
  feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
  serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
  export_dir = classifier.export_savedmodel('model_info', serving_input_receiver_fn)
  print('Exported to {}'.format(export_dir))
  print('Test set accuracy: {accuracy:0.3f}'.format(**eval_result))


 







    
  def predict_input_fn(features, labels, batch_size):
        features = dict(features)
        inputs = (features, labels) if labels is not None else features
        dataset = tf.data.Dataset.from_tensor_slices(inputs)
        dataset = dataset.batch(batch_size)
        return dataset
    


      
      
    
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
        
#        for i in class_ID.split():
#            
#            if i.isdigit():
#                
#                hey=int(i) 
#        print('heyyyyy',hey) 
#
#        print('Prediction is "{}" ({:.1f}%), expected "{}"'.format(
#            ACTIONS[class_ID]))
#      
















#  # Classify new flower
#  def new_samples():
#    return np.array([[6.4, 2.7, 5.6, 2.1]], dtype=np.float32)
#
#  predictions = list(classifier.predict(input_fn=new_samples))
#
#  print("Predicted class: {}\n".format(predictions))


if __name__ == "__main__":
    
#    
#    for i in range(300):
        
        main()

