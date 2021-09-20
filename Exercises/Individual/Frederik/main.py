import numpy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

#(a) classify instances in tst_xy, and use the corresponding label file tst_xy_class to calculate the accuracy;

#(b) classify instances in tst_xy_126 by assuming a uniform prior over the space of hypotheses,
# and use the corresponding label file tst_xy_126_class to calculate the accuracy;

#(c) classify instances in tst_xy_126 by assuming a prior probability of 0.9 for Class x and 0.1 for Class y,
# and use the corresponding label file tst_xy_126_class to calculate the accuracy; compare the results with those of (b).


def load_np(df):
    numpy = np.asarray(df.values)
    return numpy

def plot_points(numpy):
    x, y = numpy.T
    return x, y

def params(numpy): # Gets the mean, std, var and cov of a dataset.
    mean = np.mean(numpy, axis=0)
    std = np.std(numpy)
    var = np.var(numpy)
    cov = np.cov(numpy.T)
    return mean, std, var, cov

df = pd.read_csv('trn_x.txt', sep='  ', names=['col1', 'col2'])
df_1 = pd.read_csv('trn_y.txt', sep='  ', names=['col1', 'col2'])

x_train = load_np(df)
y_train = load_np(df_1)

x_1, x_2 = plot_points(x_train)
y_1, y_2 = plot_points(y_train)


plt.scatter(x_1, x_2, edgecolors='blue')
plt.scatter(y_1, y_2, edgecolors='green')
plt.show()

# (a) classify instances in tst_xy, and use the corresponding label file tst_xy_class to calculate the accuracy;

# First we must define the Multivariate gaussian/normal distribution
# To do so we need the parameters, mean, variance, standard deviation and covariance.
def classify_tst_xy ():
    # Get parameters for the distribution. Just need mean and covariance.
    x_mean, x_std, x_var, x_cov = params(x_train)
    y_mean, y_std, y_var, y_cov = params(y_train)
    # Define the multivariate_normal distribution.
    x_dist = multivariate_normal(mean=x_mean, cov=x_cov)
    y_dist = multivariate_normal(mean=y_mean, cov=y_cov)

    # Load the test data xy.
    tst_xy = pd.read_csv('tst_xy.txt', sep='  ', names=['col1', 'col2'])
    xy_test = load_np(tst_xy)
    class_xy = numpy.empty(len(xy_test), dtype = int)
    # Now for each and every entry in the data, you will compare the probability density of it belonging to x or y
    # If it belongs in class x (C1) we save value 1 in our class vector. If it belongs to y (C2) we save 2.
    for i in range(0,len(xy_test)):
        if x_dist.pdf(xy_test[i]) > y_dist.pdf(xy_test[i]):
            class_xy[i] = 1
        else:
            class_xy[i] = 2


    return class_xy

class_xy = classify_tst_xy()
label_xy = load_np(pd.read_csv('tst_xy_class.txt'))
count = 0
print(label_xy)
for i in range(0,len(label_xy)):
    if int(class_xy[i]) == int(label_xy[i]):
        count += 1
accuracy = count/2028
print("Accuracy: ", accuracy)