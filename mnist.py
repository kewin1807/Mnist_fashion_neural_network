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

def cost_function(AL, y) :
   cost = -1 / m *np.sum(np.multiply(np.log(AL), Y) + np.multiply(1-Y, np.log(1-AL)))
   return cost 

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    return dA_prev, dW, db



    

# def main() :
#     parameters = initialize_parameters([5,4,1])
#     print(parameters)

# if (__name__ == "__main__"):
#     main()
   
