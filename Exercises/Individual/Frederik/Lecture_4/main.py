import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np

mat = loadmat("mnist_all.mat")
train5 = mat["train5"]
train6 = mat["train6"]
train8 = mat["train8"]
print(loadmat("mnist_all.mat").keys())

data = np.concatenate([train5, train6, train8])

print("Train5: ",np.shape(train5))
print("Train6: ",np.shape(train6))
print("Train8: ",np.shape(train8))
print("Combined: ",np.shape(data))

#(1) from the 10 - class database, choose three classes (5, 6 and 8) and then reduce dimension to 2;

# Compute covariance matrix
cov_matrix = np.cov(data.T)
print("Cov_matrix: ", np.shape(cov_matrix))
# Compute mean for data sample
mean = np.mean(data, axis=0)
print("Mean shape: ", np.shape(mean))
# Compute eigenvectors for the data sample.

eig_val, eig_vector = np.linalg.eig(cov_matrix)

# Principle axis:
W_2d = np.array([eig_vector[:,0],eig_vector[:,1]])

# Compute formula
z = (W_2d @ (data-mean).T).T
plt.scatter(z[:,0],z[:,1], color="blue")


