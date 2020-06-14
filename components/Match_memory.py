import numpy as np


# Add the threshold
def activation_func(np_data, theta):
#active function, input is a np_array"
    #A = 1/(1+np.exp(-Z / annealing))
    for i in range(np_data.shape[1]):
        if np_data[:,i] >= theta :
            np_data[:,i] = 1
            continue
        else:   
            np_data[:,i] = 0
    return np_data  


# Find the pattern back
def match_memory(weight, data, memory, theta=0.5):
    # match
    weighted_data = np.dot(weight, np.matrix(data).T).T
    activate_data = activation_func(weighted_data, theta)  

    J = np.zeros(np.shape(memory)[0])
    # Calculation of error
    for k in range(memory.shape[0]):
        J[k] = np.linalg.norm(activate_data - memory[k])

    return activate_data, J
