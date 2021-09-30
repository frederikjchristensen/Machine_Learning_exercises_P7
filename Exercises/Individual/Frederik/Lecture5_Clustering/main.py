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
train5 = data['train5']
train6 = data['train6']
train8 = data['train8']
train = np.concatenate([train5, train6, train8])

# Compute covariance for all classes
cov = np.cov(train.T)

# Compute means for all classes
mean = np.mean(train, axis=0)

# Compute eigenvectors for the three classes
V, W = LA.eig(cov)

# Principle axis:
W_2d = np.array([W[:,0], W[:,1]])

# Compute formula
z = (W_2d @ (train-mean).T).T
plt.scatter(z[:,0], z[:,1], color='blue')

test5 = data['test5']
test6 = data['test6']
test8 = data['test8']

zt5 = (W_2d @ (test5-mean).T).T
zt6 = (W_2d @ (test6-mean).T).T
zt8 = (W_2d @ (test8-mean).T).T

mat2D = loadmat("2D568class.mat")
train2D = np.concatenate([mat2D["trn5_2dim"],mat2D["trn6_2dim"],mat2D["trn8_2dim"]])

# K-means
m1 = train2D[randrange(0,len(train2D))]
m2 = train2D[randrange(0,len(train2D))]
m3 = train2D[randrange(0,len(train2D))]
ms = [m1,m2,m3]
print(ms)

# K-means algorithm
b_ti = np.zeros((len(train2D),3))
i = 0
for x in train:
    dist1 = math.dist(x,m1)
    dist2 = math.dist(x,m2)
    dist3 = math.dist(x,m3)
    distances = [dist1, dist2, dist3]
    max_d = np.argmax(distances)
    b_ti[i][max_d] = 1
    i = i + 1


##################################################################################################
################################### Gaussian Mixture modelling ###################################
##################################################################################################

mat = loadmat("2D568class.mat")
# The concatenated data is unlabelled.
trn5_2D = mat["trn5_2dim"]
trn6_2D = mat["trn6_2dim"]
trn8_2D = mat["trn8_2dim"]

data = np.concatenate([trn5_2D,trn6_2D,trn8_2D])

# Find reference vectors and K-means
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3)
gmm.fit(data)

#predictions from gmm
labels = gmm.predict(data)
frame = pd.DataFrame(data)
frame['cluster'] = labels
frame.columns = ['Weight', 'Height', 'cluster']

color=['blue','green','cyan', 'black']
for k in range(0,3):
    data = frame[frame["cluster"]==k]
    plt.scatter(data["Weight"],data["Height"],c=color[k])
plt.show()
# E-Step



# M-Step




