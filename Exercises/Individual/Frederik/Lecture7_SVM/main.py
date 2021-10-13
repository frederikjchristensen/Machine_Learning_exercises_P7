from sklearn import *
import math
import pandas as pd
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from numpy import linalg as LA
from random import *

# Load dataset
data = loadmat("mnist_all.mat")
# Train data and labels
train0 = data['train0']
train1 = data['train1']
train = np.concatenate([train0, train1])

print(np.shape(train),"Train shape\n",np.shape(train0),"train0 shape\n",np.shape(train1),"train1 shape\n")

label = np.concatenate([np.zeros((5923)),np.ones((6742))])
print(np.shape(label),"Label shape\n")

# Load test
test = [data["test0"],data["test1"]]
test_labels = np.concatenate([np.zeros(len(data["test0"])),np.ones(len(data["test1"]))])
print("Test labels:",np.shape(test_labels))
# Define our classifier using the SVM (Sklearn) library
clf = svm.SVC()
clf.fit(train, label)
SVM_Prediction = clf.predict(test)

target_classes = [i+np.ones(len(test[i])) for i in range(2)]
print("Target_classes: ", np.shape(target_classes))

accuracy = [np.sum(SVM_Prediction[test_labels == i] == i) / len(target_classes[i])*100 for i in range(2)]
print("Accuracy: ", accuracy)
