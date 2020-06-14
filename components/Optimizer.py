import numpy as np
from components.Match_memory import match_memory


class SPSA:
    """
    An optimizer class that implements Simultaneous Perturbation Stochastic Approximation (SPSA)
    """
    def __init__(self, a, c, A, alpha, gamma):
        # Initialize gain parameters and decay factors
        self.a = a
        self.c = c
        self.A = A
        self.alpha = alpha
        self.gamma = gamma

        self.min_vals = -4.0
        self.max_vals = 4.0

        # counters
        self.t = 0


    def step(self, current_estimate, data, memory):
        """
        :param current_estimate: This is the current estimate of the parameter vector
        :return: returns the updated estimate of the vector
        """

        # get the current values for gain sequences
        a_t = self.a / (self.t + 1 + self.A)**self.alpha
        c_t = self.c / (self.t + 1)**self.gamma

        # Generate ten random sets for different delta
        # Store each set in dictionary
        delta_set = {}
        loss_cal = []
        
        for i in range(10):      
            # get the random perturbation vector from bernoulli distribution
            # it has to be symmetric around zero
            # But normal distribution does not work which makes the perturbations close to zero
            # Also, uniform distribution should not be used since they are not around zero    

            delta = np.random.randint(0,2, current_estimate.shape) * 2 - 1
            delta_set['randset' + str(i)] = delta

            _, loss_plus = match_memory(current_estimate + delta * c_t, data, memory)
            _, loss_minus = match_memory(current_estimate - delta * c_t, data, memory)
            loss_cal.append(np.min(loss_plus - loss_minus))
            
        # Select the optimal set of delta
        index_best = np.argmin(loss_cal)
        delta_best = delta_set['randset' + str(index_best)]   

        # compute the estimate of the gradient
        g_t = np.min(loss_cal)  / (2.0 * delta_best * c_t)
        current_estimate = current_estimate - a_t * g_t
 
        # Ignore results that are outside the boundaries
        if (self.min_vals is not None) and (self.max_vals is not None):      
            current_estimate = np.minimum ( current_estimate, self.max_vals )
            current_estimate = np.maximum ( current_estimate, self.min_vals )

        # increment the counter
        self.t += 1

        return current_estimate