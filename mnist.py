import mnist_reader
from activationFunction import sigmoid, dSigmoid, ReLU, dReLU, softmax
import numpy as np
X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
m = X_train.shape[0]


def initialize_parameters(layer_dims): 
    parameters = {}
    L = len(layer_dims)
    for i in range(1, L):
        parameters['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1]) * 0.01
        parameters['b' + str(i)] = np.zeros(shape = (layer_dims[i],1))
    return parameters

def linear_forward_propagation(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache


def forward_propagation(A_prev, W,b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward_propagation(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    if activation == "ReLU":
         Z, linear_cache = linear_forward_propagation(A_prev, W, b)
         A, activation_cache = ReLU(Z)
    if activation == "softmax":
        Z, linear_cache = linear_forward_propagation(A_prev, W, b)
        A, activation_cache = softmax(Z)
    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_linear_forward(X, parameters):
    A = X
    L = len(parameters)
    caches = []
    for i in range(1, L):
        A_prev = A
        A, cache = forward_propagation(A_prev, parameters['W' + str(i)], parameters['b' + str(i)], "ReLU")
        caches.append(cache)
    AL, cache = forward_propagation(A_prev, parameters['W' + str(L)], parameters['b' + str(L)], "softmax" )
    caches.append(cache)
    return AL, caches


def cost_function(AL, y) : 
   cost = -np.sum(y * np.log(AL))
   return cost 

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


def backward_propagation(dA, cache, activation):
    linear_cache , activation_cache = cache,
    if activation == "ReLU" :
        dZ = dReLU(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    if activation == "sigmoid" :
        dZ = dSigmoid(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    if activation == "softmax":
       
    return dA_prev, dW, db


def L_model_backward_propagation(caches, AL, y): # tao ra dAL truoc 
    




    

# def main() :
#     parameters = initialize_parameters([5,4,1])
#     print(parameters)

# if (__name__ == "__main__"):
#     main()
   
