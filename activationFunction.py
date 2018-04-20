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
# def dSoftmax(dA, cache):
#     Z = cache
#     A, cache1 = softmax(Z)
#     jacobian_m = np.zeros((Z.shape[0], Z.shape[0]))
#     for i in range (Z.shape[0]) :
#         for j in range (Z.shape[0]) :
#             if i == j :
#                jacobian_m[i][j] =A[i] * (1 - A[i])
#             else :
#                 jacobian_m[i][j] = -1 * A[i] * A[j]
#     return dA * jacobian_m