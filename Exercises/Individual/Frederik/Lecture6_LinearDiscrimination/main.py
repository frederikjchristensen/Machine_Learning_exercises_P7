import math
import pandas as pd
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from numpy import linalg as LA
from random import *

######################
##Training Gaussians##
######################

# Load dataset
data = loadmat("mnist_all.mat")
train1 = data['train1']
train2 = data['train2']
train3 = data['train3']
train4 = data['train4']
train5 = data['train5']
train6 = data['train6']
train7 = data['train7']
train8 = data['train8']
train9 = data['train9']
train0 = data['train0']
train = np.concatenate([train0, train1, train2, train3, train4, train5, train6, train7, train8, train9])
print(np.shape(train))

# Compute means for all classes
m0 = np.mean(train0, axis=0)
m1 = np.mean(train1, axis=0)
m2 = np.mean(train2, axis=0)
m3 = np.mean(train3, axis=0)
m4 = np.mean(train4, axis=0)
m5 = np.mean(train5, axis=0)
m6 = np.mean(train6, axis=0)
m7 = np.mean(train7, axis=0)
m8 = np.mean(train8, axis=0)
m9 = np.mean(train9, axis=0)
mean_vectors = []
mean_vectors.append(m0)
mean_vectors.append(m1)
mean_vectors.append(m2)
mean_vectors.append(m3)
mean_vectors.append(m4)
mean_vectors.append(m5)
mean_vectors.append(m6)
mean_vectors.append(m7)
mean_vectors.append(m8)
mean_vectors.append(m9)
print("Mean Shapes: ", np.shape(m0),np.shape(m1),np.shape(m2),np.shape(m3),np.shape(m4),np.shape(m5),np.shape(m6),np.shape(m7),np.shape(m8),np.shape(m9))
print("Mean_vectors shape: ", np.shape(mean_vectors))

classes_means = (m0+m1+m2+m3+m4+m5+m6+m7+m8+m9)/10
S_B = 0
for i in range (0,len(mean_vectors)):
    S_B += (mean_vectors[i]-classes_means)*(mean_vectors[i]*classes_means).T

S_W = 0
for i in range (0,len(mean_vectors)):
    S_W += (train[i]-mean_vectors[i])*(train[i]-mean_vectors[i]).T

print("S_B shape: ", np.shape(S_B))
print("S_W shape: ", np.shape(S_W))

SUSHI = np.linalg.inv(S_W)*S_B
print("Sushi shape: ", np.shape(SUSHI))



