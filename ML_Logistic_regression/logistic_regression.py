import pandas as pd
import numpy as np

data = pd.read_csv(open('data.csv'))

X = np.zeros((569, 31))
Y = np.zeros((569, 1))

for i in range(569):
    for j in range(31):
        if j == 0:
            X[i][j] = 1
        else:
            X[i][j] = data.loc[i][j + 1]

for i in range(569):
    if data.loc[i][1] == 'B':
        Y[i][0] = 0
    else:
        Y[i][0] = 1
        
theta = np.zeros((31,1))
alpha = 1
lamda = 1

min_val = X.min(axis = 0)
max_val = X.max(axis = 0)
X = (X - X.mean(axis = 0)) / (max_val - min_val)

for i in range(569):
    X[i][0] = 1
    
length_of_train = int(0.8 * len(X))
length_of_test = len(X) - length_of_train

X_train = X[:length_of_train]
X_test = X[length_of_train:]
Y_train = Y[:length_of_train]
Y_test = Y[length_of_train:]

def sigmoid(theta, X):
    hypothesis = np.zeros((len(X), 1))
    sums = X.dot(theta)
    for i in range(len(sums)):
        hypothesis[i][0] = 1/(1 + np.exp(0 - sums[i][0]))
    return hypothesis

def cost_calculate(hypothesis , Y, theta):
    costs = 0
    theta_square = np.transpose(theta).dot(theta)
    for i in range(len(hypothesis)):
        costs += (lamda * (theta_square[0][0] - theta[0][0] * theta[0][0]) - Y[i][0] * np.log(hypothesis[i][0]) - (1 - Y[i][0]) * np.log(1 - hypothesis[i][0]))/(2 * len(Y)) 
    return costs    
    
def logistic_regression(X_train, Y_train, theta, alpha):
    epsilon = 0.0001
    convergence = False
    
    while (not convergence):
        hypo = sigmoid(theta, X_train)
        cost = cost_calculate(hypo, Y_train, theta)
        update_theta = np.zeros((31, 1))
        for j in range(len(theta)):
            sums = 0
            for i in range(len(X_train)):
                sums += (hypo[i][0] - Y_train[i][0]) * X_train[i][j]
            update_theta[j][0] = theta[j][0] * (1 - alpha * lamda/len(X_train)) - (alpha * sums)/len(X_train)
            
        update_hypo = sigmoid(update_theta, X_train)
        update_cost = cost_calculate(update_hypo, Y_train, update_theta)

        print(abs(update_cost - cost))
        
        if abs(update_cost - cost) < epsilon:
            convergence = True
            return theta
        
        theta = update_theta

theta = logistic_regression(X_train, Y_train, theta, alpha)
pred = np.zeros((len(Y_test), 1))
hypo = X_test.dot(theta)

for i in range(len(pred)):
    if hypo[i][0] >= 0.5:
        pred[i][0] = 1

ctr = 0
for i in range(len(pred)):
    if pred[i][0] == Y_test[i][0]:
        ctr += 1
print (ctr/len(pred))
