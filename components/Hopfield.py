import numpy as np
from components.Match_memory import match_memory


class HopfieldNetwork:

    def __init__(self, memory, tha, eps, optimizer): 
        self.eps = eps
        self.theta = tha
        self.memory = memory
        self.optimizer = optimizer

        self.weight = self.initial_weight()

    # Initialize weights
    def initial_weight (self):
        weight = 0
        weight = np.dot(np.matrix(self.memory[0,:]).T, np.matrix(self.memory[0])) 
        for i in range(self.memory.shape[0]):
            weight = weight + np.dot(np.matrix(self.memory[i,:]).T, np.matrix(self.memory[i]))
        np.fill_diagonal(weight,0)
        weight = weight/self.memory.shape[0]

        return weight


    # Use simultaneous perturbation to get a better result
    def add_perturbation(self, test, iter):
        # Operation of Hopfield and calculation of loss
        data, _ = match_memory(self.weight, test, self.memory)
        J_min = np.min(_)
        print(0, '   ', J_min, data)
        weight_best = self.weight

        # Iteration for finding a closer pattern
        for i in range(iter):  #max number of iteration is 10  

            # Update the weights according to each pattern
            weight_best = self.optimizer.step(weight_best, data, self.memory)

            # Again for implementing Hopfield by using new weights
            data, J = match_memory(weight_best, data, self.memory)
            J_min = np.min(J)

            print(i+1, '   ', J_min, data)

            if  J_min == 0:
                break

            

