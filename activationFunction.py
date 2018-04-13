import numpy as np 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dSigmoid(x) : #đạo hàm của sigmoid 
   return sigmoid(x) * (1 - sigmoid(x))

def ReLU(x) : 
    return x * (x > 0)

def softmax(x):       # activation when output là multiclass 
    e = numpy.exp(x - numpy.max(x))  # ngăn không bị tràn số 
    if e.ndim == 1:
        return e / numpy.sum(e, axis=0)
    else:  
        return e / numpy.array([numpy.sum(e, axis=1)]).T  # ndim = 2