import pandas as pd
import numpy as np

x = pd.read_csv(open('X.csv'))
y = pd.read_csv(open('y.csv'))

X = x.to_numpy()
Y = y.to_numpy()

def train_test_split(X, Y):

    X_train = np.zeros((3999, 400))
    Y_train_orig = np.zeros((3999, 1))
    X_test = np.zeros((1000, 400))
    Y_test_orig = np.zeros((1000, 1))
    
    for i in range(X.shape[0]):
        if i % 5 == 0:
            X_test[int(i/5)] = X[i]
            Y_test_orig[int(i/5)] = Y[i]
        else:
            X_train[i - int(i/5) - 1] = X[i]
            Y_train_orig[i - int(i/5) - 1] = Y[i]
            
    Y_train = np.zeros((3999, 10))
    Y_test = np.zeros((1000, 10))
    
    for i in range(Y_train.shape[0]):
        j = int(Y_train_orig[i][0] % 10)
        Y_train[i][j] = 1
    
    for i in range(Y_test.shape[0]):
        j = int(Y_test_orig[i][0] % 10)
        Y_test[i][j] = 1
        
    return X_train.T, Y_train.T, X_test.T, Y_test.T

X_train, Y_train, X_test, Y_test = train_test_split(X, Y)

def sigmoid(Z):
    
    return 1/(1 + np.exp(-Z))

def relu(Z):
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i][j] = max(Z[i][j], Z[i][j])
            
    return Z

def tanhyp(Z):
    
    return np.tanh(Z)

def initialize_parameters(layer_dims):
    parameters = {}
    for l in range(1, len(layer_dims)):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

def forward_activation(A_prev, W, b, activation):
    Z = np.dot(W, A_prev) + b
    if activation == 'relu':
        return Z, relu(Z)
    
    elif activation == 'sigmoid':
        return Z, sigmoid(Z)
    
    elif activation == 'tanhyp':
        return Z, tanhyp(Z)
    
def forward_propagation(X, parameters):
    caches = {}
    A = X
    caches['A0'] = X
    L = int(len(parameters)/2)
    for l in range(1, L):
        A_prev = A
        Z, A = forward_activation(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], 'sigmoid')
        caches['A' + str(l)] = A
        caches['Z' + str(l)] = Z
    
    Z, A = forward_activation(caches['A' + str(L - 1)], parameters['W' + str(L)], parameters['b' + str(L)], 'sigmoid')
    caches['A' + str(L)] = A
    caches['Z' + str(L)] = Z
    
    return caches

def compute_cost(AL, Y, parameters):
    m = Y.shape[1]
    cost = -(np.sum(np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), 1 - Y)))/m

    return cost

def backward_activation(dA, Z, A, A_prev, W, activation, m):
    dZ = np.zeros((Z.shape[0], Z.shape[1]))
    if activation == 'sigmoid':
        dZ = np.multiply(dA, np.multiply(A, 1 - A))
    elif activation == 'relu':
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                if Z[i][j] >= 0:
                    dZ[i][j] = 1
                else:
                    dZ[i][j] = 0
    elif activation == 'tanhyp':
        dZ = 1 - np.power(A, 2) 
                    
    dW = np.dot(dZ, A_prev.T)/m
    db = np.sum(dZ, axis = 1, keepdims = True)/m
    
    dA_prev = np.dot(W.T, dZ)
    
    return dW, db, dA_prev

def backward_propagation(parameters, caches, Y):
    grads = {}
    L = int(len(parameters)/2)
    AL = caches['A' + str(L)]
    m = AL.shape[1]
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    grads['dW' + str(L)], grads['db' + str(L)], grads['dA' + str(L - 1)] = backward_activation(dAL, caches['Z' + str(L)], caches['A' + str(L)], caches['A' + str(L - 1)], parameters['W' + str(L)], 'sigmoid', m)
    
    for l in reversed(range(L - 1)):
        grads['dW' + str(l + 1)], grads['db' + str(l + 1)], grads['dA' + str(l)] = backward_activation(grads['dA' + str(l + 1)], caches['Z' + str(l + 1)], caches['A' + str(l + 1)], caches['A' + str(l)], parameters['W' + str(l + 1)], 'sigmoid', m)
        
    return grads
              
def update_parameters(parameters, grads, learning_rate):
    L = int(len(parameters)/2)        
    for l in range(1, L + 1):
        parameters['W' + str(l)] = parameters['W' + str(l)] - learning_rate * grads['dW' + str(l)]
        parameters['b' + str(l)] = parameters['b' + str(l)] - learning_rate * grads['db' + str(l)]
    
    return parameters
        
def nn_model(X, Y, learning_rate):
    layer_dims = []
    layer_dims.append(X.shape[0])
    layer_dims.append(int(X.shape[0] * 1.5))
    layer_dims.append(Y.shape[0])
    L = len(layer_dims) - 1
    
    parameters = initialize_parameters(layer_dims)
    
    for i in range(3000):
        caches = forward_propagation(X, parameters)
        
        cost = compute_cost(caches['A' + str(L)], Y, parameters)
        if (i % 50 == 0):
            print('cost after iteration ' + str(i) + ' is = ', cost)
        
        grads = backward_propagation(parameters, caches, Y)
        
        parameters = update_parameters(parameters, grads, learning_rate)
        
    return parameters
    
parameters = nn_model(X_train, Y_train, 1)

def predict(parameters, X):
    Z1 = np.dot(parameters['W1'], X) + parameters['b1']
    A1 = sigmoid(Z1)
    Z2 = np.dot(parameters['W2'], A1) + parameters['b2']
    A2 = sigmoid(Z2)
    
    pred = np.zeros((A2.shape[0], A2.shape[1]))
    
    for j in range(A2.shape[1]):
        max_val = A2[0][j]
        index = 0
        for i in range(A2.shape[0]):
            if A2[i][j] > max_val:
                max_val = A2[i][j]
                index = i
        pred[index][j] = 1
    
    return pred
    
pred = predict(parameters, X_test)

def accuracy(pred, Y):
    ctr = 0
    for j in range(Y.shape[1]):
        for i in range(Y.shape[0]):
            if Y[i][j] == 1 and pred[i][j] == 1:
                ctr += 1
                
    return ctr/Y.shape[1]

acc = accuracy(pred, Y_test) 

print (acc)    