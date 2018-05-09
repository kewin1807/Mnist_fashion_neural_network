import mnist_reader
from activationFunction import sigmoid, dSigmoid, ReLU, dReLU, softmax
from convertFunction import dictionary_to_vector, vector_to_dictionary, gradients_to_vector
import numpy as np

from scipy import sparse 
import matplotlib.pyplot as plt
X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
X_train = np.transpose(X_train[:5000])
y_train = np.transpose(y_train[:5000])
X_test = np.transpose(X_test[:500])
y_test = np.transpose(y_test[:500])

# x_norm_train = np.linalg.norm(X_train, axis=0)
# x_norm_test = np.linalg.norm(X_test, axis=0)
# X_train = X_train/x_norm_train
# X_test = X_test/x_norm_test

#normalization data 
X_train = (X_train - np.mean(X_train)) / np.std(X_train)
X_test = (X_test - np.mean(X_test)) / np.std(X_test)

def initialize_parameters(layer_dims): 
    parameters = {}
    L = len(layer_dims)
    for i in range(1, L):
        parameters['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1])*0.01
        parameters['b' + str(i)] = np.zeros(shape = (layer_dims[i],1))*0.01
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
    L = len(parameters) // 2
    caches = []
    for i in range(1, L):
        A_prev = A 
        A, cache = forward_propagation(A_prev, parameters['W' + str(i)], parameters['b' + str(i)], "ReLU")
        caches.append(cache)
       
    AL, cache = forward_propagation(A, parameters['W' + str(L)], parameters['b' + str(L)], "softmax")
    caches.append(cache)
   
    return AL, caches

def transform_one_hot(y, num_labels):
   Y = sparse.coo_matrix((np.ones_like(y), 
        (y, np.arange(len(y)))), shape = (num_labels, len(y))).toarray()
   return Y 

def cost_function(AL, y, lamda, parameters): 
   m = X_train.shape[1]
   L = len(parameters) // 2
   k = 0
   for i in range(L):
       k += np.sum(parameters["W" + str(i+1)] ** 2)
   Y = transform_one_hot(y, 10)
   cost = -1. / m *(np.sum(np.multiply(np.log(AL), Y))) + lamda / (2 * m) * k
   return cost

def linear_backward(dZ, cache, lamda):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) / m + lamda / m * W
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def L_model_backward_propagation(caches, AL, y, lamda):
    L =len(caches)
    grads = {}
    Y = transform_one_hot(y, 10)
    dZL = AL - Y
    linear_cache, activation_cache = caches[L-1]
    grads['dA' + str(L-1)], grads['dW' + str(L)], grads['db' + str(L)] = linear_backward(dZL, linear_cache, lamda)
    L-=1
    while (L > 0):
        linear_cache, activation_cache = caches[L-1]
        dZ = dReLU(grads['dA' + str(L)] , activation_cache)
        grads['dA' + str(L-1)], grads['dW' + str(L)], grads['db' + str(L)] = linear_backward(dZ, linear_cache, lamda)
        L-=1
    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for i in range(L):
        parameters["W" + str(i+1)] = parameters["W" + str(i+1)] - learning_rate * grads["dW" + str(i+1)]
        parameters["b" + str(i+1)] = parameters["b" + str(i+1)] - learning_rate * grads["db" + str(i+1)]
    return parameters

def loop(lamda) :
    num_iterations = 3000
    cost = []
    parameters = initialize_parameters([X_train.shape[0],128,10])
    
    for i in range(num_iterations):
        AL, caches = L_model_linear_forward(X_train, parameters)
        cost.append(cost_function(AL, y_train, lamda, parameters))
        if i % 100 == 0:
            print(cost_function(AL, y_train, lamda, parameters))
        # print(AL)
        grads = L_model_backward_propagation(caches, AL, y_train, lamda)
        parameters = update_parameters(parameters, grads, 0.25)
    return parameters, grads, cost
    
def getProbsAndPreds(X, parameters):
    probs, cache = L_model_linear_forward(X, parameters)
    preds = np.argmax(probs,axis=0)
    return probs,preds

def getAccuracy(X, Y, parameters):
    prob,preds = getProbsAndPreds(X, parameters)
    accuracy = sum(preds == Y)/(float(len(Y)))
    return accuracy


def gradient_checking(parameters, gradients, X, y ,epsilon = 1e-7 ) :
    parameter_values = dictionary_to_vector(parameters)
    grads = gradients_to_vector(parameters,gradients)
    m = parameter_values.shape[0]
    n = grads.shape[0]
    gradApproxiamte = np.zeros((m, 1))
    J_plus  = np.zeros((m, 1))
    J_minus = np.zeros((m, 1))
    
    for i in range (m) : 
        thetaPlus = np.copy(parameter_values)
        thetaPlus[i][0] += epsilon
        AL, caches = L_model_linear_forward(X, vector_to_dictionary(thetaPlus, parameters))
        J_plus[i] = cost_function(AL, y)


        thetaMinus = np.copy(parameter_values)
        thetaMinus[i][0] -= epsilon
        AL, caches = L_model_linear_forward(X, vector_to_dictionary(thetaMinus, parameters))
        J_minus[i] = cost_function(AL, y)
        
        gradApproxiamte[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)
    numerator = np.linalg.norm(grads - gradApproxiamte)                                     
    denominator = np.linalg.norm(grads) + np.linalg.norm(gradApproxiamte)                  
    difference = numerator / denominator                                              
    if difference > 1e-7:
        print("\033[93m" + "There is a mistake in the backward propagation ! difference = " + str(difference) + "\033[0m")
    else:
        print("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
    
    return difference
    
    
        
        
         
      
    

if (__name__ == "__main__"):
    # parameters = initialize_parameters([X_train.shape[0],256, 128, 10]) 
    # AL, caches = L_model_linear_forward(X_train, parameters)
    
    # m,n = softmax(AL)
    # print (m, AL.shape)
    parameter1s, grads, cost = loop(0.001)
    train_accuracy = getAccuracy(X_train, y_train, parameter1s)
    print("Train_accuracy: " + str(train_accuracy))
    test_accuracy = getAccuracy(X_test, y_test, parameter1s)
    print("test_accuracy: " + str(test_accuracy))
    plt.plot(cost)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate = 0.01")
    plt.show()

    


    
    
    
    
   
  