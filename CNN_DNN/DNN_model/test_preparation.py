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

directory = os.path.join("c:\\","Users\marcos\Documents\DNN_model\Actions_dataset\Test")
D_size=50

def normalize_vec(vector):
   global D_size
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
    
    global Sheets_det,D_size
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
                           
                        
                            with open("predict.csv","a", newline='') as trackingF:
                                    tracking_info = csv.writer(trackingF, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                                    tracking_info.writerow(row)
                            f.close()        

                print(iter)
    
    
Sheets_det=0
                
main()



