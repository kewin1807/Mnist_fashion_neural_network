import mnist_reader
from activationFunction import sigmoid, dSigmoid, ReLU, dReLU, softmax, dSoftmax
from convertFunction import dictionary_to_vector, vector_to_dictionary, gradients_to_vector
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
X_train = np.transpose(X_train[:5000])
y_train = np.transpose(y_train[:5000])
X_test = np.transpose(X_test[:1000])
y_test = np.transpose(y_test[:1000])

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

def transform_one_hot(Y):
    m = Y.shape[0]
    OHX = scipy.sparse.csr_matrix((np.ones(m), (Y, np.array(range(m)))))
    OHX = np.array(OHX.todense())
    return OHX

def cost_function(AL, y) : 
   m = X_train.shape[1]
   y = transform_one_hot(y)
   cost = -(np.sum(y * np.log(AL)) ) / m
   return cost

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


def backward_propagation(dA, cache, activation):
    linear_cache , activation_cache = cache
    if activation == "ReLU" :
        dZ = dReLU(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    if activation == "sigmoid" :
        dZ = dSigmoid(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    if activation == "softmax":
        dZ = dSoftmax(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def L_model_backward_propagation(caches, AL, y): # tao ra dAL truoc 
    L =len(caches)
    grads = {}
    y = transform_one_hot(y)
    dAL = np.subtract(AL,y)
    current_cache = caches[-1]

    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] =  backward_propagation(dAL, current_cache, activation = "softmax")
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp =  backward_propagation(grads["dA" + str(l + 2)], current_cache, activation = "ReLU")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for i in range(L):
        parameters["W" + str(i+1)] = parameters["W" + str(i+1)] - learning_rate * grads["dW" + str(i+1)]
        parameters["b" + str(i+1)] = parameters["b" + str(i+1)] - learning_rate * grads["db" + str(i+1)]
    return parameters



    

def loop() :
    
    num_iterations = 2000
    # print (y_train)
    cost = []
    parameters = initialize_parameters([X_train.shape[0],256, 150 , 10]) 
    for i in range(num_iterations):
        AL, caches = L_model_linear_forward(X_train, parameters)
        cost.append(cost_function(AL, y_train))
        if  i % 100 == 0:
            print(cost_function(AL, y_train))
        grads = L_model_backward_propagation(caches, AL, y_train)
        parameters = update_parameters(parameters, grads, 0.09)
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
    
    parameter1s, grads, cost = loop()
    
    train_accuracy = getAccuracy(X_train, y_train, parameter1s)
    print("Train_accuracy: " + str(train_accuracy))
    test_accuracy = getAccuracy(X_test, y_test, parameter1s)
    print("test_accuracy: " + str(test_accuracy))
    
    plt.plot(cost)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate = 0.01")
    plt.show()


    
    
    
    
   
  
   
