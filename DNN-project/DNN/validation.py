# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 15:19:37 2019

@author: marcos
"""

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
from pylab import savefig
import numpy as np
import os
from sklearn.metrics import precision_recall_curve, f1_score,average_precision_score, log_loss

#from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt


ROOT_DIR=os.path.abspath("../")

#objects='Hand_VAL'
objects='Pan_VAL'

    
Matrix_data=os.path.join(ROOT_DIR,'DNN\\'+objects+'.csv')
f=open(Matrix_data, 'r')
    
tracking_sheet=pd.read_csv(f)
val_pred=tracking_sheet['Pred']
val_true=tracking_sheet['GT']
val_pred=val_pred.values.tolist()
val_true=val_true.values.tolist() 



y_true = val_true
y_pred = val_pred
confusion_matrix=confusion_matrix(y_true, y_pred)
#confusion_matrix= confusion_matrix / confusion_matrix.astype(np.float).sum(axis=0)
cf = sn.heatmap(confusion_matrix, annot=True)
axes=cf.axes

axes.set_ylim(0,4)
figure=cf.get_figure()
figure.savefig(objects+'.png',dpi=700)

#print(f1_score(y_true, y_pred, average='micro'))
TP = np.diag(confusion_matrix)
FP = np.sum(confusion_matrix, axis=0) - TP
FN = np.sum(confusion_matrix, axis=1) - TP

num_classes = 4
TN = []
for i in range(num_classes):
    temp = np.delete(confusion_matrix, i, 0)    # delete ith row
    temp = np.delete(temp, i, 1)  # delete ith column
    TN.append(sum(sum(temp)))
TN


precision = TP/(TP+FP)
recall = TP/(TP+FN)
#print(precision)
#print(recall)

Prob_data=os.path.join(ROOT_DIR,'DNN\\'+objects+'_Prob.csv')
f_prob=open(Prob_data, 'r')
tracking_sheet_prob=pd.read_csv(f_prob,header=None)
y_prob=tracking_sheet_prob.values

val_true=tracking_sheet['GT']
y_true=val_true.values

print(log_loss(y_true, y_prob))
#disp = plot_precision_recall_curve(classifier, X_test, y_test)
#disp.ax_.set_title('2-class Precision-Recall curve: '
#                   'AP={0:0.2f}'.format(average_precision))