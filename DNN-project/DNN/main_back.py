
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib
import collections
import numpy as np
import tensorflow as tf
import pandas as pd
import numpy as np
import csv
import os
from scipy.interpolate import interp1d
import cv2
import DNNAnalysis_Lib
import sys

sys.path.append("C:\Program Files (x86)\IronPython 2.7\Lib")


#Actions we are going to train and predict.
ACTIONS_pan = ['NaN', 'Place', 'Remove','Saltear']
#ACTIONS_hand=['NaN','Stir','RemoveCover']
ACTIONS_hand=['NaN','Stir']




object_action= 'pan'
if object_action == 'pan' :
    actions=ACTIONS_pan
    N=30
    Steps=2
    obj='pan'
    
if object_action == 'hand' :
    actions=ACTIONS_hand
    N=30
    Steps=2
    obj='hand'    
    



    
#Frame span to infer actions. 

#When analyzing a video the frames it advances in each iteration. An iteration is trying to get an action in 30 frames, 
#the first one will be from 0 to 30, then from 2 to 32 and so forth.


#Normalization of the tracking from 'classified' folder for training and val.
#DNNAnalysis_Lib.train_test_teasers_split(obj,N,actions,2)

#Normalization of prediction info for prediction or for training with multiple actions (read info in folders).
DNNAnalysis_Lib.pred_norm(obj,N,Steps)

#Split of prediction data (already cathegorized).
#DNNAnalysis_Lib.just_split(obj,N,actions)

#Train, if predict=True, instead of using Action_train and val, it will use Predict_train and val.
#DNNAnalysis_Lib.train(obj,N,actions,train_with_predict=False)


#Prediction, creates a new csv file with prediction.
DNNAnalysis_Lib.predict(obj,N,actions)
