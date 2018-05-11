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
    
    