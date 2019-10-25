
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import collections
import numpy as np
import tensorflow as tf
import pandas as pd
import csv
import cv2
import errno
import sys


sys.path.append("C:\Program Files (x86)\IronPython 2.7\Lib")
import random
ROOT_DIR=os.path.abspath("../")
BATCH_SIZE = 10
ACTIONS_pan = ['NaN', 'Place', 'Remove']





def centroid_action( mask_pan,video_i, object_track, img2,tracking_vector) :

    
    foldername = os.path.join(ROOT_DIR,"DNN\\data_"+object_track+"\\raw_data_"+object_track+"\\")    
    folderdrop = os.path.join(ROOT_DIR,"DNN\\data_"+object_track+"\\classified\\")  
    
    if not os.path.exists(os.path.dirname(foldername)):
        try:
            os.makedirs(os.path.dirname(foldername))
        except OSError as exc: 
            if exc.errno != errno.EEXIST:
                raise
                
                
    if not os.path.exists(os.path.dirname(folderdrop)):
        try:
            os.makedirs(os.path.dirname(folderdrop))
        except OSError as exc: 
            if exc.errno != errno.EEXIST:
                raise
  
    contours, _ = cv2.findContours(mask_pan ,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    cnts = max(contour_sizes, key=lambda x: x[0])[1]
   
    M = cv2.moments(cnts)
    
    if M["m00"]>0:
       
        c0X = int(M["m10"] / M["m00"])
        c0Y = int(M["m01"] / M["m00"])
        xS1 = c0X
        yS1 = c0Y
        coord=xS1,yS1
        cv2.drawContours(img2, [cnts], -1, (0, 255, 0), 2)
        cv2.circle(img2, (c0X, c0Y), 7, (255, 255, 255), -1)
        cv2.putText(img2, "center", (c0X - 20, c0Y - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        tracking_vector.append(coord)
        
        

        with open(foldername + '\\' + object_track + "tracking" + str(video_i) + ".csv","a", newline='') as trackingF:
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
                           
             
                return add_list            
   if len(vector) == D_size:
       return vector


#def train_test_teasers_split(train_object,D_size,ACTIONS):
#    
#    
#    foldername = os.path.join(ROOT_DIR,"DNN\\data_"+train_object+"\\classified\\")
#    folderdrop = os.path.join(ROOT_DIR,"DNN\\data_"+train_object+"\\train_ready\\")    
#    if not os.path.exists(os.path.dirname(foldername)):
#      try:
#        os.makedirs(os.path.dirname(foldername))
#      except OSError as exc: 
#        if exc.errno != errno.EEXIST:
#            raise
#    
#    if not os.path.exists(os.path.dirname(folderdrop)):
#      try:
#        os.makedirs(os.path.dirname(folderdrop))
#      except OSError as exc: 
#        if exc.errno != errno.EEXIST:
#            raise
#    
#    Sheets_det=0
#    
#    try:
#        os.remove(foldername+"tracking_sheet_"+ train_object +".csv")
#    except OSError:
#        pass
#    for root,dirs,files in os.walk(foldername):
#        
#        for file in files:
#           if file.endswith(".csv"):
#                csv_f=os.path.join(foldername,file)
#                try:
#                        f=open(csv_f, 'r')
#                except FileNotFoundError:
#                            pass
#                tracking_sheet=pd.read_csv(f, header=None, error_bad_lines=False) 
#                f_name=os.path.basename(csv_f)
#                X=tracking_sheet[0]
#                Y=tracking_sheet[1]
#                
#                
#                X=normalize_vec(X,D_size)
#                Y=normalize_vec(Y,D_size)
#                
#
#                ok=False
#                if f_name.find('NaN')  !=-1:
#                    action=0
#                    ok=True
#                if f_name.find('Place')  !=-1:
#                    action=1 
#                    ok=True
#                if f_name.find('Remove') !=-1:
#                    action=2
#                    ok=True
#                if f_name.find('Saute') !=-1:
#                    action=3   
#                    ok=True
#                if f_name.find('Romove_Cover') !=-1:
#                    action=1   
#                    ok=True
#                if f_name.find('Stir') !=-1:
#                    action=2   
#                    ok=True
#                   
#                row=np.append(X,Y)   
#                
#                row=np.append(row,action)
#                if ok==True:
#                    Sheets_det+=1
#                    with open(foldername+"tracking_sheet_"+ train_object +".csv","a", newline='') as trackingF:
#                                tracking_info = csv.writer(trackingF, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#                                tracking_info.writerow(row)
#    
#                f.close()
#                
#    
#
#
#
#    try:
#        os.remove(folderdrop + 'Action_test_'+ train_object +'.csv')
#    except OSError:
#        pass
#    try:
#        os.remove(folderdrop + 'Action_train_'+ train_object +'.csv')
#    except OSError:
#        pass
#    
#    test_var=round(0.2*Sheets_det)
#    
#    info_row_train=[]
#    info_row_train.append(Sheets_det-test_var)
#    info_row_train.append(2*D_size)
#    for i in range(len(ACTIONS)): 
#    
#        info_row_train.append(ACTIONS[i])
#    
#    info_row_test=[]
#    info_row_test.append(test_var)
#    info_row_test.append(2*D_size)
#    for i in range(len(ACTIONS)): 
#    
#        info_row_test.append(ACTIONS[i])
#        
#        
#    
#    f_track=open(foldername+"tracking_sheet_"+ train_object +".csv", 'r')
#    tracking_sheet=pd.read_csv(f_track, header=None, error_bad_lines=False) 
#                  
#    test_values=np.random.choice(Sheets_det, size=test_var, replace=False)
#    
#    
#    with open(folderdrop+"Action_test_"+ train_object +".csv","a", newline='') as Test_F:
#                tracking_info = csv.writer(Test_F, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#                tracking_info.writerow(info_row_test)
#    
#   
#    with open(folderdrop+"Action_train_"+ train_object +".csv","a", newline='') as Train_F:
#                 tracking_info = csv.writer(Train_F, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#                 tracking_info.writerow(info_row_train) 
#    
#    for i in range(Sheets_det):
#        
#        main_row=tracking_sheet.values[i,:]
#        if i in  test_values:
#    
#            
#            with open(folderdrop+"Action_test_"+ train_object +".csv","a", newline='') as Test_F:
#                tracking_info = csv.writer(Test_F, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#                tracking_info.writerow(main_row)
#            
#        else :
#    
#             with open(folderdrop+"Action_train_"+ train_object +".csv","a", newline='') as Train_F:
#                 tracking_info = csv.writer(Train_F, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#                 tracking_info.writerow(main_row)                       


def train_test_teasers_split(train_object,D_size,ACTIONS,step):
    
    
    foldername = os.path.join(ROOT_DIR,"DNN\\data_"+train_object+"\\classified\\")
    folderdrop = os.path.join(ROOT_DIR,"DNN\\data_"+train_object+"\\train_ready\\")    
    if not os.path.exists(os.path.dirname(foldername)):
      try:
        os.makedirs(os.path.dirname(foldername))
      except OSError as exc: 
        if exc.errno != errno.EEXIST:
            raise
    
    if not os.path.exists(os.path.dirname(folderdrop)):
      try:
        os.makedirs(os.path.dirname(folderdrop))
      except OSError as exc: 
        if exc.errno != errno.EEXIST:
            raise
    
    Sheets_det=0
    
    try:
        os.remove(foldername+"tracking_sheet_"+ train_object +".csv")
    except OSError:
        pass
        for root,dirs,files in os.walk(foldername):
 
         for file in files:
            
           if file.endswith(".csv"):
                
           
                
                row=[]               
                Init=0
                End=D_size
                end_of_file = False
                iter=0
                
                csv_f=os.path.join(foldername,file)
                try:
                        f=open(csv_f, 'r')
             
                        tracking_sheet=pd.read_csv(f, header=None) 
                        
                        X=tracking_sheet[0] 
                        Y=tracking_sheet[1]
                        print('Points found : ',len(tracking_sheet))

                        f_name=os.path.basename(csv_f)
                        for i in range(len(tracking_sheet)):
           
        
                                while end_of_file == False:
                                    Xs=[]
                                    Ys=[]



                                    Xs.append(X[Init:End])
                                    Ys.append(Y[Init:End])
                                    iter+=1

                                    if (len(tracking_sheet)-End) < 2 :
                                        end_of_file = True
                                   
                                    
                                    row=np.append(Xs,Ys)   
                                    Init+=step
                                    End+=step
                                    ok=False
                                    if f_name.find('NaN')  !=-1:
                                        action=0
                                        ok=True
                                    if f_name.find('Place')  !=-1:
                                        action=1 
                                        ok=True
                                    if f_name.find('Remove') !=-1:
                                        action=2
                                        ok=True
                                    if f_name.find('Saute') !=-1:
                                        action=3   
                                        ok=True
                                    if f_name.find('Stir') !=-1:
                                        action=1   
                                        ok=True
                                    if f_name.find('Romove_Cover') !=-1:
                                        action=2   
                                        ok=True
                                                                              
                                    
                                    row=np.append(row,action)
                                    if ok==True:
                                        Sheets_det+=1
                                        with open(foldername+"tracking_sheet_"+ train_object +".csv","a", newline='') as trackingF:
                                                    tracking_info = csv.writer(trackingF, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                                                    tracking_info.writerow(row)                                        

                except FileNotFoundError:
                                   pass
                f.close()
                
    



    try:
        os.remove(folderdrop + 'Action_test_'+ train_object +'.csv')
    except OSError:
        pass
    try:
        os.remove(folderdrop + 'Action_train_'+ train_object +'.csv')
    except OSError:
        pass
    
    test_var=round(0.2*Sheets_det)
    
    info_row_train=[]
    info_row_train.append(Sheets_det-test_var)
    info_row_train.append(2*D_size)
    for i in range(len(ACTIONS)): 
    
        info_row_train.append(ACTIONS[i])
    
    info_row_test=[]
    info_row_test.append(test_var)
    info_row_test.append(2*D_size)
    for i in range(len(ACTIONS)): 
    
        info_row_test.append(ACTIONS[i])
        
        
    
    f_track=open(foldername+"tracking_sheet_"+ train_object +".csv", 'r')
    tracking_sheet=pd.read_csv(f_track, header=None, error_bad_lines=False) 
                  
    test_values=np.random.choice(Sheets_det, size=test_var, replace=False)
    
    
    with open(folderdrop+"Action_test_"+ train_object +".csv","a", newline='') as Test_F:
                tracking_info = csv.writer(Test_F, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                tracking_info.writerow(info_row_test)
    
   
    with open(folderdrop+"Action_train_"+ train_object +".csv","a", newline='') as Train_F:
                 tracking_info = csv.writer(Train_F, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                 tracking_info.writerow(info_row_train) 
    
    for i in range(Sheets_det):
        
        main_row=tracking_sheet.values[i,:]
        if i in  test_values:
    
            
            with open(folderdrop+"Action_test_"+ train_object +".csv","a", newline='') as Test_F:
                tracking_info = csv.writer(Test_F, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                tracking_info.writerow(main_row)
            
        else :
    
             with open(folderdrop+"Action_train_"+ train_object +".csv","a", newline='') as Train_F:
                 tracking_info = csv.writer(Train_F, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                 tracking_info.writerow(main_row)    
                 
                 
def just_split(train_object,D_size,ACTIONS)    :
        
    foldername = os.path.join(ROOT_DIR,"DNN\\data_"+train_object+"\\train_test_split\\")
    folderdrop = os.path.join(ROOT_DIR,"DNN\\data_"+train_object+"\\train_test_split\\splitted\\")   
    
    
    if not os.path.exists(os.path.dirname(foldername)):
      try:
        os.makedirs(os.path.dirname(foldername))
      except OSError as exc: 
        if exc.errno != errno.EEXIST:
            raise
    
    if not os.path.exists(os.path.dirname(folderdrop)):
      try:
        os.makedirs(os.path.dirname(folderdrop))
      except OSError as exc: 
        if exc.errno != errno.EEXIST:
            raise
       

    for root,dirs,files in os.walk(foldername):
        for file in files:
           if file.endswith(".csv"):
                csv_f=os.path.join(foldername,file)
           
                try:
                        f=open(csv_f, 'r')
                  
                        
                        tracking_sheet=pd.read_csv(f, header=None, error_bad_lines=False) 
        
                        try:
                            os.remove(folderdrop + 'Predict_test_'+ train_object +'.csv')
                        except OSError:
                            pass
                        try:
                            os.remove(folderdrop + 'Predict_train_'+ train_object +'.csv')
                        except OSError:
                            pass
                        
                        Sheets_det=len(tracking_sheet)
                        
                        
                        test_var=round(0.2*Sheets_det)
                        
                        info_row_train=[]
                        info_row_train.append(Sheets_det-test_var)
                        info_row_train.append(2*D_size)
                        for i in range(len(ACTIONS)): 
                        
                            info_row_train.append(ACTIONS[i])
                        
                        info_row_test=[]
                        info_row_test.append(test_var)
                        info_row_test.append(2*D_size)
                        for i in range(len(ACTIONS)): 
                        
                            info_row_test.append(ACTIONS[i])
                     
                                      
                        test_values=np.random.choice(Sheets_det, size=test_var, replace=False)
                        
                        
                        with open(folderdrop+"Predict_test_"+ train_object +".csv","a", newline='') as Test_F:
                                    tracking_info = csv.writer(Test_F, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                                    tracking_info.writerow(info_row_test)
                        
                       
                        with open(folderdrop+"Predict_train_"+ train_object +".csv","a", newline='') as Train_F:
                                     tracking_info = csv.writer(Train_F, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                                     tracking_info.writerow(info_row_train) 
                        
                        for i in range(Sheets_det):
                            
                            main_row=tracking_sheet.values[i,:]
                            if i in  test_values:
                        
                                
                                with open(folderdrop+"Predict_test_"+ train_object +".csv","a", newline='') as Test_F:
                                    tracking_info = csv.writer(Test_F, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                                    tracking_info.writerow(main_row)
                                
                            else :
                        
                                 with open(folderdrop+"Predict_train_"+ train_object +".csv","a", newline='') as Train_F:
                                     tracking_info = csv.writer(Train_F, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                                     tracking_info.writerow(main_row)      
                except FileNotFoundError:
                            pass

def pred_norm(pred_object,D_size,step):
    
    
    foldername = os.path.join(ROOT_DIR,"DNN\\data_"+pred_object+"\\data_prediction\\")    
    if not os.path.exists(os.path.dirname(foldername)):
      try:
        os.makedirs(os.path.dirname(foldername))
      except OSError as exc: 
        if exc.errno != errno.EEXIST:
            raise
    try :
        os.remove(foldername+'predict_'+pred_object+'.csv')
    except FileNotFoundError:
        pass 

    for root,dirs,files in os.walk(foldername):
 
        for file in files:
            
           if file.endswith(".csv"):
                
                row=[]               
                Init=0
                End=D_size
                end_of_file = False
                iter=0
                
                csv_f=os.path.join(foldername,file)
                try:
                        f=open(csv_f, 'r')
             
                        tracking_sheet=pd.read_csv(f, header=None) 
                        
                        X=tracking_sheet[0] 
                        Y=tracking_sheet[1]
                        print('Points found : ',len(tracking_sheet))
                        for i in range(len(tracking_sheet)):
           
        
                                while end_of_file == False:
                                    Xs=[]
                                    Ys=[]



                                    Xs.append(X[Init:End])
                                    Ys.append(Y[Init:End])
                                    iter+=1

                                    if (len(tracking_sheet)-End) < 2 :
                                        end_of_file = True
                                   
       
                                    row=np.append(Xs,Ys)   
                                    Init+=step
                                    End+=step
                                
                                    with open(foldername+'predict_'+pred_object+'.csv',"a", newline='') as trackingF:
                                            tracking_info = csv.writer(trackingF, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                                            tracking_info.writerow(row)
                                    f.close()        

                except FileNotFoundError:
                                   pass
                
                
    



def train(train_object,D_size,ACTIONS,train_with_predict):
  
  global  BATCH_SIZE 

  foldername = os.path.join(ROOT_DIR,"DNN\\data_"+train_object+"\\model_"+train_object+"\\")
  model_dir = os.path.join(ROOT_DIR,"DNN\\data_"+train_object+"\\model_"+train_object+"\\model_info\\")
  if not os.path.exists(os.path.dirname(foldername)):
      try:
        os.makedirs(os.path.dirname(foldername))
      except OSError as exc: 
        if exc.errno != errno.EEXIST:
            raise
    
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

  if train_with_predict==False:  
      TRAINING=os.path.join(ROOT_DIR,"DNN\\data_"+train_object+"\\train_ready\\Action_train_"+ train_object +".csv")
      TEST=os.path.join(ROOT_DIR,"DNN\\data_"+train_object+"\\train_ready\\Action_test_"+ train_object +".csv")
  if train_with_predict==True: 
      TRAINING=os.path.join(ROOT_DIR,"DNN\\data_"+train_object+"\\train_ready\\Predict_train_"+ train_object +".csv")
      TEST=os.path.join(ROOT_DIR,"DNN\\data_"+train_object+"\\train_ready\\Predict_test_"+ train_object +".csv")    


  
  COLUMN_NAMES=[]
  for i in range(D_size*2+1):
  
      if i<D_size*2+1:     
         COLUMN_NAMES.append(str(i))
      else :
         COLUMN_NAMES.append('action')
         
  y_name = str(D_size*2)     
  
  train = pd.read_csv(TRAINING, names=COLUMN_NAMES, header=0)
  train_x, train_y = train, train.pop(y_name)

    
  test = pd.read_csv(TEST, names=COLUMN_NAMES, header=0)
  test_x, test_y = test, test.pop(y_name)


  feature_columns = [tf.feature_column.numeric_column(key=key)
                   for key in train_x.keys()]

  
  
  classifier =tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                              hidden_units=[61,122,244,488,488,244,122,61],model_dir=model_dir,
                                              n_classes=len(ACTIONS))
  
  
  
  classifier.train(
    input_fn=lambda: train_input_fn(train_x.astype(int), train_y.astype(int), batch_size=BATCH_SIZE),
    steps=1000)
  
  
  
  
  eval_result = classifier.evaluate(
    input_fn=lambda: eval_input_fn(test_x.astype(int), test_y.astype(int), batch_size=BATCH_SIZE))
  print('Test set accuracy: {accuracy:0.3f}'.format(**eval_result))
  
  feature_columns = [tf.feature_column.numeric_column(key=key)
                   for key in train_x.keys()]
  feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
  serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
  export_dir = classifier.export_savedmodel('model_info', serving_input_receiver_fn)
  print('Exported to {}'.format(export_dir))
  print('Test set accuracy: {accuracy:0.3f}'.format(**eval_result))


 





def predict(pred_object,D_size,ACTIONS):
 
    
    global  BATCH_SIZE 
    
    model_dir = os.path.join(ROOT_DIR,"DNN\\data_"+pred_object+"\\model_"+pred_object+"\\model_info\\")
    model_dir = os.path.join(ROOT_DIR,"DNN\\data_"+pred_object+"\\model_"+pred_object+"\\model_info\\")
    foldername = os.path.join(ROOT_DIR,"DNN\\data_"+pred_object+"\\data_prediction\\predict_"+pred_object+".csv")   
    TRAINING=os.path.join(ROOT_DIR,"DNN\\data_"+pred_object+"\\train_ready\\Action_train_"+ pred_object +".csv")
    
    def predict_input_fn(features, labels, batch_size):
        features = dict(features)
        inputs = (features, labels) if labels is not None else features
        dataset = tf.data.Dataset.from_tensor_slices(inputs)
        dataset = dataset.batch(batch_size)
        return dataset
    
    COLUMN_NAMES=[]
    for i in range(D_size*2+1):
  
      if i<D_size*2+1:     
         COLUMN_NAMES.append(str(i))
      else :
         COLUMN_NAMES.append('action')
         
    y_name = str(D_size*2)   
    


    train = pd.read_csv(TRAINING, names=COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)     

    feature_columns = [tf.feature_column.numeric_column(key=key)
                       for key in train_x.keys()]  
      
    classifier =tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                                  hidden_units=[61,122,244,488,488,244,122,61],model_dir=model_dir,
                                                  n_classes=len(ACTIONS))
      
      
  
      
    f_track=open(foldername, 'r')
    predict_sheet=pd.read_csv(f_track, header=None)         
        
        
    predict_x  = {}       
    for i in range(D_size*2):
            main_row=predict_sheet.values[:,i]
        
            write_dict={str(i) : main_row}
            predict_x.update(write_dict)
    print(main_row)    
    predictions =classifier.predict(
            input_fn=lambda: predict_input_fn(predict_x, labels=None,batch_size=BATCH_SIZE))
    i=0    
    
    
    try:
        os.remove(pred_object + "_prediction_results.csv")
    except OSError:
        pass
    
    for prediction in zip(predictions):
            class_id = prediction[0]
            class_ID=class_id["classes"]
            probability = class_id["probabilities"]
            print(probability)
            class_ID=str(class_ID)
            print(class_ID)
            i+=1
            
            with open(pred_object + "_prediction_results.csv","a", newline='') as trackingF:
                tracking_info = csv.writer(trackingF, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                tracking_info.writerow(class_ID)
