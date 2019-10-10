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
import numpy as np
import csv
import os
from scipy.interpolate import interp1d
import cv2

## Data sets
BATCH_SIZE = 10
#TRAINING = "Action_training.csv"
#TEST = "Action_test.csv"
ACTIONS = ['NaN', 'Place', 'Remove']
#model_dir='model_info'
D_size=50


def centroid_action( mask_pan,video_i, object_track, img2,tracking_vector) :
 
    
    #if sarten_colocada(vitro_mask, mask_pan) == True:


    contours, _ = cv2.findContours(mask_pan ,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    cnts = max(contour_sizes, key=lambda x: x[0])[1]
    
        # compute the center of the contour
    M = cv2.moments(cnts)
    if M["m00"]>0:
       
        c0X = int(M["m10"] / M["m00"])
        c0Y = int(M["m01"] / M["m00"])
        xS1 = c0X
        yS1 = c0Y
        coord=xS1,yS1
        print(yS1)
        # draw the contour and center of the shape on the image
        cv2.drawContours(img2, [cnts], -1, (0, 255, 0), 2)
        cv2.circle(img2, (c0X, c0Y), 7, (255, 255, 255), -1)
        cv2.putText(img2, "center", (c0X - 20, c0Y - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        tracking_vector.append(coord)
        print(tracking_vector)
        with open(object_track + "tracking" + str(video_i) + ".csv","a", newline='') as trackingF:
            tracking_info = csv.writer(trackingF, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            tracking_info.writerow(coord)


def normalize_vec(vector,D_size):

   if len(vector)>D_size:
                erase_pos=len(vector)-D_size
                erase_list=[]
                for i in range(len(vector)): erase_list.append(vector[i])
                
                selection=[]
                selection_vec=( np.random.choice(len(vector), size=erase_pos, replace=False))
                for i in range(len(selection_vec)): selection.append(selection_vec[i])
                selection.sort(reverse=True) 
   
                for i in range(len(selection)):

                    del erase_list[selection[i]]
                    
                vector=erase_list
                return vector

   if len(vector)<D_size:
                
                add_list=[]
                for i in range(len(vector)): add_list.append(vector[i])
                
                while (len(add_list)< D_size): 
                    
                    GAP=0
                    j=0
                    while j+1<len(add_list):
             
                        gap=round(abs(add_list[j]-add_list[j+1]))     
                         
                        if gap>GAP:
                            GAP=gap
                            pos=j
                            
                        j=j+1 
                        
        

                    interp =round((add_list[pos]+ add_list[pos+1])/2)
                    prev=add_list[pos]
                    add_list[pos]=interp
                    for i in range(len(add_list)-pos+1): 
                       
                        
                        if len(add_list)==pos+i:
                            add_list.append(prev)
                        
                        else:
                            
                            prev_i=add_list[pos+i]
                            add_list[pos+i]=prev
                            prev=prev_i
                           
#                    if count==4:
#                            print(add_list)
#                          
#                            print(GAP) 
#                            print(pos)
             
                return add_list            
   if len(vector) == D_size:
       return vector


def train_test_teasers_split(directory):
    
    global Sheets_det
    try:
        os.remove('tracking_sheet.csv')
    except OSError:
        pass
    for root,dirs,files in os.walk(directory):
        for file in files:
           if file.endswith(".csv"):
                csv_f=os.path.join(directory,file)
                f=open(csv_f, 'r')
                #f=r'C:\Users\marcos\Documents\DNN_model\Actions_dataset\NaN_1.csv'
                tracking_sheet=pd.read_csv(f, header=None) 
                f_name=os.path.basename(csv_f)
                X=tracking_sheet[0]
                Y=tracking_sheet[1]
                Sheets_det+=1
                
                X=normalize_vec(X,D_size)
                Y=normalize_vec(Y,D_size)
#                print(len(X))
#                print(len(Y))
    #            z=np.polyfit(X,Y,6)
    #            p = np.poly1d(z)
    #            xp = np.linspace(min(X), max(X), 100)
    #            yp = p(xp)
                if f_name.find('NaN')  !=-1:
                    action=0
                if f_name.find('Place')  !=-1:
                    action=1 
                    
                if f_name.find('Remove') !=-1:
                    action=2
    #                
    #            row=np.sqrt(xp**2+yp**2)
                row=np.append(X,Y)   
                row=np.append(row,action)
                
                with open("tracking_sheet.csv","a", newline='') as trackingF:
                            tracking_info = csv.writer(trackingF, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            tracking_info.writerow(row)
    
                f.close()
                
    



    try:
        os.remove('Action_test.csv')
    except OSError:
        pass
    try:
        os.remove('Action_train.csv')
    except OSError:
        pass
    
    test_var=round(0.2*Sheets_det)
    
#    directory_main=os.path.join("c:\\","Users\marcos\Documents\Asistente\codigo_python\Asistente\Action_tracking")
#    csv_main=os.path.join(directory_main,"tracking_sheet.csv")
#    f_track=open(csv_main, 'r')
    f_track=open("tracking_sheet.csv", 'r')
    tracking_sheet=pd.read_csv(f_track, header=None) 
                  
    test_values=np.random.choice(Sheets_det, size=test_var, replace=False)
    
    
    for i in range(Sheets_det):
        
        main_row=tracking_sheet.values[i,:]
        if i in  test_values:
    
            
            with open("Action_test.csv","a", newline='') as Test_F:
                tracking_info = csv.writer(Test_F, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                tracking_info.writerow(main_row)
            
        else :
    
             with open("Action_train.csv","a", newline='') as Train_F:
                 tracking_info = csv.writer(Train_F, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                 tracking_info.writerow(main_row)                       


def pred_norm(directory,D_size):
    
    global Sheets_det
    try:
        os.remove('predict.csv')
    except OSError:
        pass

    for root,dirs,files in os.walk(directory):
 
        for file in files:
            
           if file.endswith(".csv"):
                
                row=[]               
                Init=0
                End=D_size
                end_of_file = False
                iter=0
                
                csv_f=os.path.join(directory,file)
                f=open(csv_f, 'r')
                #f=r'C:\Users\marcos\Documents\DNN_model\Actions_dataset\NaN_1.csv'
                tracking_sheet=pd.read_csv(f, header=None) 
                
                X=tracking_sheet[0] 
                Y=tracking_sheet[1]
                print('Puntos_encontrados',len(tracking_sheet))
                for i in range(len(tracking_sheet)):
   

                        while end_of_file == False:
                            Xs=[]
                            Ys=[]
                            Sheets_det+=1
                            Init+=5
                            End+=5
                            Xs.append(X[Init:End])
                            Ys.append(Y[Init:End])
                            iter+=1
                            if (len(tracking_sheet)-End) < 5 :
                                end_of_file = True
                           
                           
    
            
                            row=np.append(Xs,Ys)   
                           
                        
                            with open('predict.csv',"a", newline='') as trackingF:
                                    tracking_info = csv.writer(trackingF, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                                    tracking_info.writerow(row)
                            f.close()        

                print('lines',iter)
    



def train(model_dir):
  
  global  BATCH_SIZE, ACTIONS  
    
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


  TRAINING="Action_train.csv"
  TEST="Action_test.csv"
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


 







def predict(model_dir,TRAINING,PREDICT):
 
    
    global  BATCH_SIZE 
    
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
    
#    TRAINING = "Action_training.csv"     
         
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
      
      
    
            
    csv_main=os.path.join(PREDICT)
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
            