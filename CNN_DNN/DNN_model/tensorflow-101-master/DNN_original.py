# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 11:53:34 2019

@author: marcos
"""

# DNNClassifier on CSV input dataset.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib
import cv2
import numpy as np
import tensorflow as tf

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

def main():
  # Load datasets.
  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TRAINING,
      target_dtype=np.int,
      features_dtype=np.float32)
  print(training_set)

  test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TEST,
      target_dtype=np.int,
      features_dtype=np.float32)

  # Specify that all features have real-value data
  feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

  # Build 3 layer DNN
  classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                              hidden_units=[5,10,5],
                                              n_classes=3)
  # Define the training inputs
  def get_train_inputs():
    x = tf.constant(training_set.data)
    print(x)
    y = tf.constant(training_set.target)
    print(y)
    return x, y

  # Fit model.
  classifier.fit(input_fn=get_train_inputs, steps=2000)

  # Define the test inputs
  def get_test_inputs():
    x = tf.constant(test_set.data)

    y = tf.constant(test_set.target)

    return x, y

  # Evaluate accuracy.
  accuracy_score = classifier.evaluate(input_fn=get_test_inputs,
                                       steps=1)["accuracy"]

  print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

  # Classify new flower
  def new_samples():
    return np.array([[6.4, 2.7, 5.6, 2.1]], dtype=np.float32)

  predictions = list(classifier.predict(input_fn=new_samples))

  print("Predicted class: {}\n".format(predictions))

if __name__ == "__main__":
    main()
