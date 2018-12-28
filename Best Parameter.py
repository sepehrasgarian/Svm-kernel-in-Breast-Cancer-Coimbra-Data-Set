# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 17:44:06 2018

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
X_train1,X_test,Y_train1,Y_test=train_test_split(X,Y,test_size=0.15,random_state=0) 
X_train,X_arzyabi,Y_train,Y_arzyabi=train_test_split(X_train1,Y_train1,test_size=0.15,random_state=0 )
from sklearn.svm  import SVC
svclass = SVC()#kernel method"""             
svclass.fit(X_train,Y_train)
predictions = svclass.predict(X_arzyabi) 
#Import scikit-learn metrics module for accuracy calculation
# Model Accuracy: how often is the classifier correct?
param_grid = {'C': [0.1,1, 10, 100, 1000],'gamma': [1,0.1,0.01,0.001,0.0001] ,'coef0': [1,2,3,4,5,6,7,8,9,10],'kernel': ['rbf']} 
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(X_arzyabi,Y_arzyabi)
grid.best_params_
print(grid.best_params_,"saaaa")




