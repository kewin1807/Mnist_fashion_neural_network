import numpy as np 
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

def dSigmoid(dA, cache):
    Z = cache
    dZ = dA * sigmoid(Z) * (1 - sigmoid(Z))
    assert (dZ.shape == Z.shape)
    return dZ

def ReLU(Z):
    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)
    cache = Z
    return A, cache
def dReLU(dA, cache): 
    Z = cache
    Z[Z<=0] = 0
    Z[Z>0] = 1
    return dA * Z
def softmax(Z):       
    e_Z = np.exp(Z - np.max(Z, axis = 0, keepdims = True))
    A = e_Z / e_Z.sum(axis = 0)
    return A, Z
def dSoftmax(dA, cache):
    z = cache
    s , z = softmax(z)
    dZ = dA * s * (1 - s)
    return dZ

