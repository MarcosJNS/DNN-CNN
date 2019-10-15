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
import DNNAnalysis
import sys

sys.path.append("C:\Program Files (x86)\IronPython 2.7\Lib")

ACTIONS_pan = ['NaN', 'Place', 'Remove']
actions=ACTIONS_pan



#DNNAnalysis.train_test_teasers_split('pan',30,actions)

#DNNAnalysis.pred_norm('pan',30,2)


#DNNAnalysis.just_split('pan',30,actions)

DNNAnalysis.train('pan',30,actions,train_with_predict=True)
#
#DNNAnalysis.predict('pan',30,actions)