import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
boston_dataset = load_boston()
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target

"""
CRIM: Per capita crime rate by town
ZN: Proportion of residential land zoned for lots over 25,000 sq. ft
INDUS: Proportion of non-retail business acres per town
CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
NOX: Nitric oxide concentration (parts per 10 million)
RM: Average number of rooms per dwelling
AGE: Proportion of owner-occupied units built prior to 1940
DIS: Weighted distances to five Boston employment centers
RAD: Index of accessibility to radial highways
TAX: Full-value property tax rate per $10,000
PTRATIO: Pupil-teacher ratio by town
B: 1000(Bk — 0.63)², where Bk is the proportion of [people of African American descent] by town
LSTAT: Percentage of lower status of the population
MEDV: Median value of owner-occupied homes in $1000s """

X = np.zeros((506, 14))

for row in range(506):
    for col in range(14):
        if col == 0:
            X[row][col] = 1
        else:
            X[row][col] = boston.loc[row][col-1]

theta = np.zeros((14,1))
Y = np.zeros((506, 1))
alpha = 0.001


for row in range(506):
    Y[row][0] = boston.loc[row][13]

def feature_scaling (X):
    for j in range(1,len(X[0])):
        sums = 0
        maximum = X[0][j]
        minimum = X[0][j]  
        for i in range(len(X)):
            sums = sums + X[i][j]
            if X[i][j] > maximum:
                maximum = X[i][j]
            if X[i][j] < minimum:
                minimum = X[i][j]
        avg = (float(sums))/len(X)
        diff = maximum - minimum
        for i in range(len(X)):
            X[i][j] = (X[i][j] - avg)/diff
    return X  
   
X = feature_scaling(X)

def train_test_split (X, Y):
    length_of_train = int(0.9 * len(X))
    return X[:length_of_train], Y[:length_of_train], X[length_of_train:], Y[length_of_train:]

X_train, Y_train, X_test, Y_test = train_test_split (X, Y)

def linear_regression (X, Y, theta, alpha):
    convergence = False
    m = len(X)
    n = len(X[0]) - 1
    while( not convergence):
        error = X.dot(theta) - Y
        cost = np.transpose(error).dot(error)/(2 * m)
        update_theta = np.zeros((n+1,1))
        for j in range(n+1):
            sum_cost = 0
            for i in range(m):
                sum_cost = sum_cost + (error[i][0] * X[i][j])
            update_theta[j][0] = theta[j][0] - (alpha * sum_cost) / m 
        
        update_error = X.dot(update_theta) - Y
        update_cost = (np.transpose(update_error).dot(update_error))/(2 * m)
        print(round(update_cost[0][0] - cost[0][0], 3))
        
        if (round(update_cost[0][0] - cost[0][0], 3)) == 0:
            convergence = True
            return theta
        theta = update_theta

theta = linear_regression(X_train, Y_train, theta, alpha)

Y_pred = X_test.dot(theta)

SSE = (np.transpose(Y_test - Y_pred)).dot(Y_test - Y_pred)[0][0] / (2 * len(Y_test))    