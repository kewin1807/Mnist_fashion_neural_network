import mnist_reader
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
X_train = X_train[:5000]/255
y_train = y_train[:5000].reshape(5000, 1)
X_test = X_test[:1000]/255
y_test = y_test[:1000].reshape(1000, 1)

X_train = np.insert(X_train, 0, 1, axis=1)
X_test = np.insert(X_test, 0, 1, axis=1)

yy = y_train
y_train = 1. * np.equal(
    np.tile(y_train, (1, 10)),
    np.tile(np.arange(10), (X_train.shape[0], 1)),
)

def sigmoid(x):
    return 1 / (np.exp(-x) + 1)

def initialize_theta(num_labels, m):
    
    parameter = np.random.randn(num_labels, m) * 0.01
    return parameter

def update_params(parameter, learning_rate, grads):
    parameter= parameter - learning_rate * grads
    return parameter
def propagate(w, X, Y, lamda):
    m = X.shape[0]
    A = sigmoid(np.dot(X, np.transpose(w)))           
    cost = -1./m * (np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))) +  lamda / (2. * m) * np.sum(w**2)                               
    dw = 1. / m * np.dot(np.transpose(A-Y), X) + (lamda / m) * w

    # assert(dw.shape == w.shape)
    cost = np.squeeze(cost)
    # assert(cost.shape == ())
   
    grads = dw     
    return grads, cost
def loop():
    num_iterations = 1000
    
    costs = []
    parameters = initialize_theta(10, X_train.shape[1]) 
    for i in range(num_iterations):
        grads, cost = propagate(parameters, X_train, y_train, lamda=0.)
        costs.append(cost)
        if  i % 100 == 0:
            print(cost)
        parameters = update_params(parameters, grads, 0.3)
    return parameters, grads, costs
    
def getProbsAndPreds(X, parameters):
    probs = sigmoid(np.dot(X, np.transpose(parameters)))
    preds = np.argmax(probs,axis=1)
    preds = np.reshape(preds, (preds.shape[0], 1))
    return probs,preds

def getAccuracy(X, Y, parameters):
    prob,preds = getProbsAndPreds(X, parameters)
    accuracy = 1. * np.sum(np.equal(Y, preds))/(Y.shape[0])
    return accuracy
if (__name__ == "__main__"):
    
    parameter1s, grads, costs= loop()

    train_accuracy = getAccuracy(X_train, yy, parameter1s)
    print("Train_accuracy: " + str(train_accuracy))
    test_accuracy = getAccuracy(X_test, y_test, parameter1s)
    print("test_accuracy: " + str(test_accuracy))
    
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate = 0.01")
    plt.show()
    

