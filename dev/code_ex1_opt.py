#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 14:11:49 2019

@author: abhilash
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('ex1data2.txt',delimiter = ',')
X = data[:,:2]
y = data[:,2]
m = y.shape[0]
#feature normalization
def feat_norm(X):
    mu = np.mean(X)
    sig = np.std(X)
    t = np.ones((X.shape[0],1))
    X_norm = (X - (t*mu))/(t*sig)
    return X_norm
X = np.concatenate((np.ones((m,1)),X.T),axis = 1)
#computing cost fxn
def comp_cost_mult(X,y,theta):
    m = y.shape[0]
    J = (1/(2*m)) *(X.dot(theta) - y).T * (X.dot(theta) - y)
    return J
#Gradient Descent
alpha = 0.01
num_iters = 1000
theta = np.zeros((3,1))
def grad_desc_mult(X,y,theta,alpha,num_iters):
    m = y.shape[0]
    J_his = np.zeros((num_iters,1))
    for i in (0,num_iters):
        theta = theta - alpha * (1/m) * (((X.dot(theta)) - y).T * X).T
        J_his[i,0] = comp_cost_mult(X,y,theta)
    return theta,J_his
theta,J_his = grad_desc_mult(X,y,theta,alpha,num_iters)
print(theta)