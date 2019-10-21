
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
actions=ACTIONS_pan

#Frame span to infer actions. 
N=30   

#When analyzing a video the frames it advances in each iteration. An iteration is trying to get an action in 30 frames, 
#the first one will be from 0 to 30, then from 2 to 32 and so forth.
Steps=2

#Normalization of the tracking from 'classified' folder for training and val.
#DNNAnalysis_Lib.train_test_teasers_split('pan',N,actions)

#Normalization of prediction info for prediction or for training with multiple actions (read info in folders).
DNNAnalysis_Lib.pred_norm('pan',N,Steps)

#Split of prediction data (already cathegorized).
#DNNAnalysis_Lib.just_split('pan',N,actions)

#Train, if predict=True, instead of using Action_train and val, it will use Predict_train and val.
#DNNAnalysis_Lib.train('pan',N,actions,train_with_predict=True)


#Prediction, creates a new csv file with prediction.
DNNAnalysis_Lib.predict('pan',N,actions)
