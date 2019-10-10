# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 14:19:12 2019

@author: marcos
"""
 
import os
import urllib
import collections
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.python.saved_model import tag_constants












TRAINING = "predict.csv"
TRAINING='Action_training.csv'



COLUMN_NAMES=[]
for i in range(101):
  
      if i<101:     
         COLUMN_NAMES.append(str(i))
      else :
         COLUMN_NAMES.append('action')
         

   
y_name = '100'       
  
train = pd.read_csv(TRAINING, names=COLUMN_NAMES, header=0)
print(train)  
train_x, train_y = train, train.pop(y_name)
print(train_x[1])
print(train[:])