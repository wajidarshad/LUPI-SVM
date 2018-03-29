# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 16:00:11 2018

@author: Wajid Abbasi
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
from classifiers import *

def get_data_with_noise(X,noise_level):# This funtion adds random noise to data. Just to create toy input feature space. No use otherwise
    return X+noise_level*(2*np.random.normal(size=[X.shape[0],X.shape[1]])-1)

if __name__=="__main__":
    samples=1000
    n_features=10
    
    X, Y = make_classification(n_samples=samples,n_features=n_features, n_redundant=0, n_informative=n_features,
                                 n_clusters_per_class=1)# make a toy dataset for classfication using SKlearn
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)# Splited data into train and test
    
    clf=linclassLUPI(epochs=1000, Lambda=0.1,Lambda_star=0.01,Lambda_s=0.001)# Define LUPI-SVM Classifier
    
    LUPI_train_data = list(zip(get_data_with_noise(X_train,0.2),X_train,y_train))
    """
    Zip() convert two feature spaces and labels in useable format to our implementation.
    Please remember, first feature space to zip() would always be considered as input feature space.
    
    Here, Input_space: get_data_with_noise(X_train,0.2) ; Privileged_space:X_train and labels:y_train

    get_data_with_noise() is just to make difference between input and Privileged feature spaces for toy dataset. 
    In real you have your own features for both spaces.
    
    """
    clf.fit(LUPI_train_data) # for training we need both feature spaces
    print("Predicted Scores:",clf.predict_score(get_data_with_noise(X_test,0.2)))# for testing we need only input feature space
    clf.save('trained_clf.txt')
