import numpy as np 
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

def dSigmoid(dA, cache):
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    assert (dZ.shape == Z.shape)
    return dZ

def ReLU(Z):
    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)
    cache = Z
    return A, cache
def dReLU(dA, cache): 
    Z = cache
    dZ = np.array(dA, copy=True) 
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ

def softmax(Z):       
    e_Z = np.exp(Z - np.max(Z,  axis=0))
    A = e_Z / e_Z.sum(axis = 0)
    return A, Z

