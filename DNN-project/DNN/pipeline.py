
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
import pandas as pd

from scipy.interpolate import interp1d
import DNNAnalysis_Lib
import sys

sys.path.append("C:\Program Files (x86)\IronPython 2.7\Lib")


#Actions we are going to train and predict.
ACTIONS_pan = ['NaN', 'Place', 'Remove','Saltear']
#ACTIONS_hand=['NaN','Stir','RemoveCover']
ACTIONS_hand=['NaN','Stir']






def train(gather_data,predict_train,obj,N,actions,Steps):
    if gather_data == True: 
        DNNAnalysis_Lib.train_test_teasers_split(obj,N,actions,Steps)    
    DNNAnalysis_Lib.train(obj,N,actions,predict_train)
    

def predict(obj,N,actions,Steps):

        DNNAnalysis_Lib.pred_norm(obj,N,Steps)  
        DNNAnalysis_Lib.predict(obj,N,actions)
        
def predict2train(obj,N,actions):        
        DNNAnalysis_Lib.just_split(obj,N,actions)
        
def query(arg,obj,gather_data,predict_train,N,actions,Steps):
    
    if arg =='train': train(gather_data,predict_train,obj,N,actions,Steps),
    
    if arg =='predict': predict(obj,N,actions,Steps),
    
    if arg =='predict2train': predict2train(obj,N,actions)
        
    else :
        print(arg, 'query not found')
    
    
    
    
    
def pipeline(obj,arg,gather_data,predict_train):    
    
    
    if obj == 'pan' :
        actions=ACTIONS_pan
        N=30
        Steps=2
        obj='pan'
    
    if obj == 'hand' :
        actions=ACTIONS_hand
        N=30
        Steps=2
        obj='hand' 
        
        
    query(arg,obj,gather_data,predict_train,N,actions,Steps)   
    