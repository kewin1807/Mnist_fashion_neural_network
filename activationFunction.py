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
    e = np.exp(Z - np.max(Z))  
    if e.ndim == 1:
        A =  e / np.sum(e, axis=0)
    else:  
        A = e / np.array([np.sum(e, axis=1)]).T  
    cache = Z 
    return A, cache
def dSoftmax(dA, cache):
    X, W, b = cache
    dZ = X.dot(dA.T)
    return dZ