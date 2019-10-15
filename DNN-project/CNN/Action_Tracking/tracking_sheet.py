# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 10:48:49 2019

@author: marcos
"""

import pandas as pd
import numpy as np
import csv
import os
from scipy.interpolate import interp1d
import pandas as pd

directory = os.path.join("c:\\","Users\marcos\Documents\DNN_model\Actions_dataset")


def normalize_vec(vector):
   D_size=50
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
                       





def main():
    
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
                
                X=normalize_vec(X)
                Y=normalize_vec(Y)
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
                
    
    
Sheets_det=0
                
main()


try:
    os.remove('Action_test.csv')
except OSError:
    pass
try:
    os.remove('Action_train.csv')
except OSError:
    pass

test_var=round(0.2*Sheets_det)

directory_main=os.path.join("c:\\","Users\marcos\Documents\Asistente\codigo_python\Asistente\Action_tracking")
csv_main=os.path.join(directory_main,"tracking_sheet.csv")
f_track=open(csv_main, 'r')
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

