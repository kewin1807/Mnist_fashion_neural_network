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

def relu(Z):
    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)
    cache = Z
    return A, cache
def dReLU(dA, cache): # đạo hàm của hàm ReLU
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ
def softmax(Z):       # activation when output là multiclass 
    e = numpy.exp(Z - numpy.max(Z))  # ngăn không bị tràn số 
    A = Z
    if e.ndim == 1:
        A =  e / numpy.sum(e, axis=0)
    else:  
        A = e / numpy.array([numpy.sum(e, axis=1)]).T  # ndim = 2
    cache = Z 
    return A, cache
def dSoftmax(dA, cache):
    X, W, b = cache

    #  A = softmax((W.T.dot(X)))
    # E = A - Y
    dZ = X.dot(dA.T)
    return dZ