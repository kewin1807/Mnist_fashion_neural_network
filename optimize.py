
import numpy as np
import math
def random_minibatch(X, y, minibatch_size = 64):
    m = X.shape[1]
    mini_batches = []
    y = y.reshape((1, m))
    permutation = list(np.random.permutation(m))
    shuffle_X = X[:, permutation]
    shuffle_Y = y[:, permutation].reshape((1, m))
    num_minibatch = int(math.floor(m / minibatch_size))
    for i in range(num_minibatch):
        mini_batch_X = shuffle_X[:, i * minibatch_size : (i+1) * minibatch_size]
        mini_batch_Y = shuffle_Y[:, i * minibatch_size : (i+1)* minibatch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % minibatch_size != 0:
        mini_batch_X = shuffle_X[:, num_minibatch * minibatch_size : m]
        mini_batch_Y = shuffle_Y[:, num_minibatch * minibatch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

# momentum
def initial_velocity(parameters):
    V = {}
    L = len(parameters) // 2
    for i in range(L):
        V["dW" + str(i+1)] = np.zeros((parameters["W"+ str(i+1)].shape[0], parameters["W" + str(i+1)].shape[1]))
        V["db"+ str(i+1)] = np.zeros((parameters["b"+ str(i+1)].shape[0], parameters["b" + str(i+1)].shape[1]))
    return V

def update_parameter_momentum(parameters, grads, V, beta , learning_rate):
    L = len(parameters) // 2
    for i in range(L):
        V["dW" + str(i+1)] = beta * V["dW" + str(i+1)] + (1 - beta)* grads["dW"+ str(i+1)]
        V["db"+ str(i+1)] = beta * V["db" + str(i+1)] + (1 - beta) * grads["db" + str(i+1)]

        parameters["W" + str(i+1)] = parameters["W"+ str(i+1)] - learning_rate * V["dW" + str(i+1)]
        parameters["b" + str(i+1)] = parameters["b" + str(i+1)] - learning_rate * V["db" + str(i+1)]
    return parameters, V

#adam 
def initial_Adam(parameters):
    L = len(parameters) // 2
    V = {}
    S = {}
    for i in range(L):
        V["dW" + str(i+1)] = np.zeros((parameters["W"+ str(i+1)].shape[0], parameters["W" + str(i+1)].shape[1]))
        V["db"+ str(i+1)] = np.zeros((parameters["b"+ str(i+1)].shape[0], parameters["b" + str(i+1)].shape[1]))
        S["dW" + str(i+1)] = np.zeros((parameters["W"+ str(i+1)].shape[0], parameters["W" + str(i+1)].shape[1]))
        S["db"+ str(i+1)] = np.zeros((parameters["b"+ str(i+1)].shape[0], parameters["b" + str(i+1)].shape[1]))
    return V, S


# GRADED FUNCTION: update_parameters_with_adam# GRADED 

def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate,
                                beta1=0.9, beta2=0.999, epsilon=1e-8):
   
    L = len(parameters) // 2            
    v_corrected = {}
    s_corrected = {}
    
  
    for l in range(L):
        
        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads['dW' + str(l + 1)]
        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads['db' + str(l + 1)]

        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - np.power(beta1, t))
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - np.power(beta1, t))
        
        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * np.power(grads['dW' + str(l + 1)], 2)
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.power(grads['db' + str(l + 1)], 2)
        
        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - np.power(beta2, t))
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - np.power(beta2, t))
        
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v_corrected["dW" + str(l + 1)] / np.sqrt(s["dW" + str(l + 1)] + epsilon)
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v_corrected["db" + str(l + 1)] / np.sqrt(s["db" + str(l + 1)] + epsilon)
      
    return parameters, v, s



