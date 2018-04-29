import numpy as np
def dictionary_to_vector(parameters):
    """
    Roll all our parameters dictionary into a single vector satisfying our specific required shape.
    """
    keys = []
    count = 0
    getKeys = []
    for key,value in parameters.items():
        getKeys.append(key)
    for key in getKeys:
        
        # flatten parameter
        new_vector = np.reshape(parameters[key], (-1,1))
        keys = keys + [key]*new_vector.shape[0]
        
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta

def vector_to_dictionary(theta, param):
    """
    Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
    """
    parameters = {}
    paramSize = 0
    for i in range (len(param) // 2) :
        paramWSize = param["W" + str(i+1)].shape[0] * param["W" + str(i+1)].shape[1]
        paramBSize = param["b" + str(i+1)].shape[0]
        W = paramSize + paramWSize
        b = W + paramBSize
        parameters["W" + str(i+1)] = theta[paramSize:W].reshape(param["W" + str(i+1)].shape)
        parameters["b" + str(i+1)] = theta[W : b].reshape(param["b" + str(i+1)].shape)
        paramSize = b
    return parameters

def gradients_to_vector(parameters, grads):
    """
    Roll all our gradients dictionary into a single vector satisfying our specific required shape.
    """
    keys = []
    count = 0
    getKeys = []
    
    for key,value in parameters.items():
        getKeys.append(key)
    for key in getKeys:

        # flatten parameter
        new_vector = np.reshape(grads["d" + key], (-1,1))
        
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1
    return theta