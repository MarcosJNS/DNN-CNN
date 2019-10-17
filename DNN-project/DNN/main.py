# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:26:57 2019

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
import numpy as np
import csv
import os
from scipy.interpolate import interp1d
import cv2
import DNNAnalysis_Lib
import sys

sys.path.append("C:\Program Files (x86)\IronPython 2.7\Lib")

ACTIONS_pan = ['NaN', 'Place', 'Remove']
actions=ACTIONS_pan
N=30
Steps=2

DNNAnalysis_Lib.train_test_teasers_split('pan',N,actions)

#DNNAnalysis_Lib.pred_norm('pan',N,Steps)

#DNNAnalysis_Lib.just_split('pan',N,actions)

DNNAnalysis_Lib.train('pan',N,actions,train_with_predict=False)

DNNAnalysis_Lib.predict('pan',N,actions)
