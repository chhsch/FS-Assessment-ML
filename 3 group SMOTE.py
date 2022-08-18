#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 13:00:52 2021

@author: hathaway
"""

"""
Created on Tue Nov 30 13:14:55 2021

@author: hathaway
"""

import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing, metrics
from sklearn import datasets, svm, metrics
from sklearn.svm import SVC
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from numpy import mean
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from collections import Counter
#讀取資料，將ID設定成index
df= pd.read_excel('/Users/hathaway/Desktop/new data/normalize(舊data)corr.xlsx')
df.set_index("ID" , inplace=True)
#依照不同級距做分數的切割
#df.loc[(df.Qn<15),["Qn_group"]]=1
#df.loc[(df.Qn>15),["Qn_group"]]=2
#df.loc[(df.Qn==15),["Qn_group"]]=1
#ID=[15,18,19,20,23,24,25,27,28,29,30,31,34,35,37,39,40,41,42,43,44,45,48]
ID=[15,18,19,20,23,24,251,252,27,28,29,30,311,312,34,35,37,39,40,41,42,43,44,45,48]

#為了做cross validation,輪流將個別受試者的資料挑出，當test集
def testdataforX(i):
    test=df.loc[ID[i]]
    #print(test)
    if test.ndim==1:
        x_test= test.drop(['Qn_group'],axis=0)
        x_test= x_test.drop(['Qn'],axis=0)
    else:
        x_test= test.drop(['Qn_group'],axis=1)
        x_test= x_test.drop(['Qn'],axis=1)
    print(x_test)
    df_clean=df.drop(index=[ID[i]])
    return df_clean,x_test,test

#做undersampling 以及挑出training data
def traindata(df_clean):
    
    smo = SMOTE(k_neighbors=5,random_state = 1969)
    #smo = ADASYN(random_state=5)
    #smo=SMOTE(kind='regular',k_neighbors=2)


    #count_class_2, count_class_1= df_clean.Qn_group.value_counts()
    #不知道為何count_class_1, count_class_2是反的
    #class_1 = df_clean[df_clean['Qn_group'] == 1]
    #class_2 = df_clean[df_clean['Qn_group'] == 2]
    #class_3 = df_clean[df_clean['Qn_group']== 3]
    #count_class_1, count_class_2, count_class_3 = df_clean.Qn_group.value_counts()
    #print(count_class_1,count_class_2, count_class_3)
    
    #class_1_under = class_1.sample(count_class_2)
    #class_1_over = class_1.sample(count_class_2,replace=True)
    #class_2_under = class_2.sample(count_class_2)
    #train_under = pd.concat([class_1_under,class_2_under, class_3], axis=0)
    #train_under = pd.concat([class_1_over,class_2], axis=0)
    
    x=df_clean.drop(['Qn_group'],axis=1)
    x=x.drop(['Qn'] ,axis=1)
    y= df_clean['Qn_group']
    
    x_train_array,y_train_array = smo.fit_resample(x, y)
    print(Counter(x_train_array))
    print(x_train_array)
    print(Counter(y_train_array))
    print(y_train_array)
    return x_train_array,y_train_array

#挑出y_test且輸出成表格
def testdataforY(i,y_test_table_zero):
    test=df.loc[ID[i]]
    if test.ndim==1:
        y_test= int(test['Qn_group'])
        y_testValues=y_test
        
    else:
        y_test1= test['Qn_group']
        y_test=y_test1.to_frame()
        y_testValues=y_test.values
    y_testValues_table=np.vstack((y_test_table_zero,y_testValues))
    y_test_table_zero=y_testValues_table
    return y_test_table_zero

#模型訓練
def modeltrain(x_train_array,y_train_array,x_test,test):
    forest = ensemble.RandomForestClassifier(n_estimators = 10,random_state=0, n_jobs=-1)
    forest_fit = forest.fit(x_train_array,y_train_array)
    boost = ensemble.AdaBoostClassifier(n_estimators = 100)
    boost_fit = boost.fit(x_train_array,y_train_array)
    C=4
    clf=svm.SVC(kernel='rbf',C=C,gamma=4)
    #clf=svm.SVC(kernel='poly', degree=3, C=C)
    clf=svm.SVC(kernel='linear', C=C)
    clf.fit(x_train_array,y_train_array)
    xgbc=XGBClassifier()
    xgbc_fit=xgbc.fit(x_train_array,y_train_array)
    if test.ndim==1:
        x_test=x_test.values.reshape(1,30)
    else:
        x_test=x_test.values
    y_pred1=clf.predict(x_test) 
    y_pred=pd.DataFrame(y_pred1)
    return y_pred

#將y_pred堆疊
def PredicYtable(y_pred,y_predic_table_zero1):
    y_predicValues=y_pred.values
    y_predicValues_table=np.vstack((y_predic_table_zero1,y_predicValues))
    
    y_predic_table_zero1=y_predicValues_table
    print(y_predic_table_zero1)
    return y_predic_table_zero1

#將實際值與預測值合併，目前還不會用到
def tabletogether(y_test_table_zero,y_predic_table_zero1):
    y_predic_table_all= np.zeros((1,2))
    y_predic_table_all=np.hstack((y_test_table_zero,y_predic_table_zero1))
    return y_predic_table_all

#計算f1 score
def result(y_test_table_zero,y_predic_table_zero1,y_f1_table_zero,y_precision_table_zero,y_recall_table_zero):
    #conf_mat=confusion_matrix(y_predic_table_zero1,y_test_table_zero)
    #accuracy = (conf_mat[0, 0] + conf_mat[1, 1])/conf_mat.sum()
    #print(conf_mat)
    #print("Accuracy: {:.2f}%".format(accuracy*100))
    f1=f1_score(y_test_table_zero, y_predic_table_zero1, average='micro')
    precision=precision_score(y_test_table_zero,y_predic_table_zero1,average='micro')
    recall=recall_score(y_test_table_zero,y_predic_table_zero1,average='micro')
    y_f1_table_zero=np.vstack((y_f1_table_zero,f1))
    y_precision_table_zero=np.vstack((y_precision_table_zero,precision))
    y_recall_table_zero=np.vstack((y_recall_table_zero,recall))
    #y_con_table=np.vstack((y_con_table,conf_mat))
    #print("{:.2f}%" .format(f1*100))
    return y_f1_table_zero,y_precision_table_zero,y_recall_table_zero

class ClfMetrics:
    """
    This class calculates some of the metrics of classifier including accuracy, precision, recall, f1 according to confusion matrix.
    Args:
        y_true (ndarray): 1d-array for true target vector.
        y_pred (ndarray): 1d-array for predicted target vector.
    """
    def __init__(self, y_true, y_pred):
        self._y_true = y_true
        self._y_pred = y_pred
    def confusion_matrix(self):
        """
        This function returns the confusion matrix given true/predicted target vectors.
        """
        
        cm=confusion_matrix(y_test_table_zero,y_predic_table_zero1)
        
        self._tp = cm[0, 0]
        print(self._tp)
        self._tn = cm[1, 1]
        print(self._tn)
        self._fp = cm[1, 0]
        print(self._fp)
        self._fn = cm[0, 1]
        print(self._fn)
        return cm
    def accuracy_score(self):
        """
        This function returns the accuracy score given true/predicted target vectors.
        """
        cm = self.confusion_matrix()
        accuracy = (self._tn + self._tp) / np.sum(cm)
        return accuracy
    def precision_score(self):
        """
        This function returns the precision score given true/predicted target vectors.
        """
        precision = self._tp / (self._tp + self._fp)
        return precision
    def recall_score(self):
        """
        This function returns the recall score given true/predicted target vectors.
        """
        recall = self._tp / (self._tp + self._fn)
        return recall
    def f1_score(self, beta=1):
        """
        This function returns the f1 score given true/predicted target vectors.
        Args:
            beta (int, float): Can be used to generalize from f1 score to f score.
        """
        precision = self.precision_score()
        recall = self.recall_score()
        f1 = (1 + beta**2)*precision*recall / ((beta**2 * precision) + recall)
        return f1
def main():
    y_predic_table_zero1= np.zeros((1,)) 
    y_test_table_zero= np.zeros((1,))
    y_f1_table_zero= np.zeros((1,))
    y_precision_table_zero= np.zeros((1,))
    y_recall_table_zero= np.zeros((1,))
    #print("report:\n",classification_report(y_test_table_zero,y_predic_table_zero1))
    #y_con_table= np.zeros((3,3))
    for i in range(0,25):
        df_clean,x_test,test=testdataforX(i)
        x_train_array,y_train_array=traindata(df_clean)
        y_pred=modeltrain(x_train_array,y_train_array,x_test,test)
        y_test_table_zero=testdataforY(i,y_test_table_zero)
        y_predic_table_zero1=PredicYtable(y_pred,y_predic_table_zero1)
            #y_predic_table_all=tabletogether(y_test_table_zero,y_predic_table_zero1)
            #y_f1_table_zero,y_precision_table_zero,y_recall_table_zero=result(y_test_table_zero,y_predic_table_zero1,y_f1_table_zero)
    y_predic_table_zero1=np.delete(y_predic_table_zero1,[0],axis=0) 
    y_test_table_zero=np.delete(y_test_table_zero,[0],axis=0)
    y_f1_table_zero=np.delete(y_f1_table_zero,[0],axis=0)
    #y_predic_table_all=np.delete(y_predic_table_all,[0],axis=0)
    return y_predic_table_zero1,y_test_table_zero

y_predic_table_zero1,y_test_table_zero=main()   
y_test_table_zeroAA=np.ravel(y_test_table_zero)
y_predic_table_zero1AA=np.ravel(y_predic_table_zero1)

clf_metrics = ClfMetrics(y_test_table_zeroAA,y_predic_table_zero1AA)

clf_metrics2=clf_metrics.confusion_matrix()
print(clf_metrics2)
clf_metrics3=clf_metrics2.T

acc=(clf_metrics2[0,0]+clf_metrics2[1,1]+clf_metrics2[2,2])/360
print('accuracy: %.4f' %acc)
precision1=clf_metrics2[0,0]/sum(clf_metrics3[0])
if np.isnan(precision1):
            precision1= np.nan_to_num(precision1)
precision2=clf_metrics2[1,1]/sum(clf_metrics3[1])
if np.isnan(precision2):
            precision2= np.nan_to_num(precision2)
precision3=clf_metrics2[2,2]/sum(clf_metrics3[2])
if np.isnan(precision3):
            precision3= np.nan_to_num(precision3)
precisionmean=(precision1+precision2+precision3)/3
print('precision: %.4f' %precisionmean)
recall1=clf_metrics2[0,0]/sum(clf_metrics2[0])
recall2=clf_metrics2[1,1]/sum(clf_metrics2[1])
recall3=clf_metrics2[2,2]/sum(clf_metrics2[2])
recallmean=(recall1+recall2+recall3)/3
print('recall: %.4f' %recallmean)

f1=2*precisionmean*recallmean/ (precisionmean+ recallmean)
if np.isnan(f1):
            f1= np.nan_to_num(f1)
print('f1: %.4f' %f1)

#acc=clf_metrics.accuracy_score()
#precision=clf_metrics.precision_score()
#recall=clf_metrics.recall_score()
#f1=clf_metrics.f1_score()

