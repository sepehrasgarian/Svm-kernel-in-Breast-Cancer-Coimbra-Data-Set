# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 18:28:58 2018

@author: sepehr
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 14:46:31 2018

@author: sepehr
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
dataset=pd.read_csv('dataR2.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
print(Y)
from sklearn.cross_validation import train_test_split
X_train1,X_test,Y_train1,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0) 
X_train,X_arzyabi,Y_train,Y_arzyabi=train_test_split(X_train1,Y_train1,test_size=0.15,random_state=0 )
from sklearn.svm  import SVC
svclass = SVC(kernel='rbf',gamma=0.01)#kernel method"""             
svclass.fit(X_train,Y_train)
predictions = svclass.predict(X_train) 
"""
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(Y_test, predictions)) 
"""
count = 0

"""for x in range (0,83):
    if predictions[x] !=Y_train[x]:
        print("hello")
        count=count + 1 
print(count)
print("error",(count/(83))*100)"""
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print( "errorrr:",1-metrics.accuracy_score( Y_train,predictions))