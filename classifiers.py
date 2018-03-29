# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 00:59:45 2017

@author: Wajid Arshad Abbasi

This module contains the class definitions for the Stochastic subgradient descent based large margin classifiers for Learning Using Privileged Information (LUPI)

"""


import random
import numpy as np
import matplotlib.pyplot as plt

    
class ClassifierBase:
    """
    This is the base class for LUPI
    """
    
    def __init__(self,**kwargs):
    
        if 'epochs' in kwargs:
            self.epochs=kwargs['epochs']
        else:
            self.epochs=100
        if 'Lambda' in kwargs:
            self.Lambda=kwargs['Lambda']
        else:
            self.Lambda=0.01
        if 'Lambda_star' in kwargs:
            self.Lambda_star=kwargs['Lambda_star']
        else:
            self.Lambda_star=0.01
        if 'Lambda_s' in kwargs:
            self.Lambda_s=kwargs['Lambda_s']
        else:
            self.Lambda_s=0.001
        self.w=None
        self.w_star=None
        self.Name=None
        
    def fit(self,bags,**kwargs):
        pass
        
        
    def predict_score(self,test_example):
        w=self.w
        pred_score=test_example.dot(w.T)
        return pred_score
    def save(self,ofname):
        with open(ofname,'w') as fout:
            fout.write(self.toString())
    def load(self,ifname):
        with open(ifname) as fin:
           self.fromString(fin.read())         
    def toString(self):
        import json
        s='#Name='+str(self.__class__)
        s+='#w='+str(json.dumps(self.w.tolist()))
        s+='#w_star='+str(json.dumps(self.w_star.tolist()))
        s+='#Epochs='+str(self.epochs)  
        s+='#Lambda='+str(self.Lambda)
        s+='#Lambda_star='+str(self.Lambda_star)
        s+='#Lambda_s='+str(self.Lambda_s)
        return s
        
    def fromString(self,s):    
        import json
        for token in s.split('#'):
            if token.find('w=')>=0 or token.find('W=')>=0:
                self.w=np.array(json.loads(token.split('=')[1]))
            if token.find('w_star=')>=0 or token.find('W_star=')>=0:
                self.w_star=np.array(json.loads(token.split('=')[1]))
            elif token.find('Epochs=')>=0:
                self.epochs=float(token.split('=')[1]) 
            elif token.find('Lambda_star=')>=0:
                self.Lambda_star=float(token.split('=')[1])
            elif token.find('Lambda=')>=0:
                self.Lambda=float(token.split('=')[1])
            elif token.find('Lambda_s=')>=0:
                self.Lambda_s=float(token.split('=')[1])

#############################################################################################

class linclassLUPI(ClassifierBase):   
    """
    This class defines the stochastic gradient descent based linear large margin classifier for LUPI.

    Parent Class: ClassifierBase
    
    Properties:
    epochs: No. of epochs to be run for optimization
    Lambda, Lambda_satr and Lambda_s: The Regularization Hyperparameters
    
    Methods:
    train(dataset)
    predict(example)
    load(filename)
    save(filename)
    
    USAGE
    Class definition:
    clf=linclassLUPI() # create a classifier object with default arguments epochs=100, Lambda=0.01, Lambda_star=0.01, Lambda_s=0.001
    clf=linclassLUPI(epochs=100, Lambda=0.01,Lambda_star=0.1,Lambda_s=0.001) # create a classifier object with customized arguments
    
    Training:
    clf.fit(clf.train([[[x1],[X1*],y1],[x2],[X2*],y2],[x3],[X3*],y3],....[Xn],[Xn*],yn]])) where X:Input Feature Space, X*: Privileged Feature Space and y: Labels
    
    Predict:
    clf.predict_score([[X_test1],[X_test2]]) X_test: test examples only input feature space
    
    Load Classifier:
    clf.load(filename)
    
    Save Classifier:
    clf.save(filename)
    """
    
    def fit(self, dataset,**kwargs):
        
        siz1=np.shape(dataset[0][0])[0]
        siz2=np.shape(dataset[0][1])[0]
        w=np.array(np.zeros(siz1))  
        w_star=np.array(np.zeros(siz2))
        T=(len(dataset))*self.epochs
        for t in range(T):
            mue=1.0/(self.Lambda*(t+1))
            mue_star=1.0/(self.Lambda_star*(t+1))
            update_w=False
            update_w_star=False
            if (t)%self.epochs==0:
                np.random.shuffle(dataset)
            instance_chosen=dataset[(t-1)%len(dataset)]
            if 1-instance_chosen[2]*(instance_chosen[0].dot(w.T))-instance_chosen[2]*(instance_chosen[1].dot(w_star.T))>0 and 1-instance_chosen[2]*(instance_chosen[0].dot(w.T))>0:
                update_w=True
            if -instance_chosen[2]*(instance_chosen[1].dot(w_star.T))>0 or 1-instance_chosen[2]*(instance_chosen[0].dot(w.T))-instance_chosen[2]*(instance_chosen[1].dot(w_star.T))>0:
                update_w_star=True
            if update_w:
                w=((1-(1.0/(t+1)))*w)+(mue*(instance_chosen[2]*instance_chosen[0]))
            else:
                w=((1-(1.0/(t+1)))*w)
            if update_w_star:
                w_star=((1-(1.0/(t+1)))*w_star)-(mue_star*self.Lambda_s*(instance_chosen[2]*instance_chosen[1]))+(mue_star*(instance_chosen[2]*instance_chosen[1]))
            else:
                w_star=((1-(1.0/(t+1)))*w_star)-(mue_star*self.Lambda_s*(instance_chosen[2]*instance_chosen[1]))
        self.w=w
        self.w_star=w_star

#####################################################################################################################################


