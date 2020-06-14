import numpy as np
from components.Optimizer import SPSA
from components.Hopfield import HopfieldNetwork

'''
memory0 = [1, 1, 0, 0]              
memory1 = [0, 1, 1, 0]               
memory2 = [0, 1, 0, 1]
memory = np.array([memory0,memory1,memory2])
'''

patterns = np.array([[0,0,0,0],
                    [0,0,0,1],
                    [0,0,1,0],
                    [0,0,1,1],
                    [0,1,0,0],
                    [0,1,0,1],
                    [0,1,1,0],
                    [0,1,1,1],
                    [1,0,0,0],
                    [1,0,0,1],
                    [1,0,1,0],
                    [1,0,1,1],
                    [1,1,0,0],
                    [1,1,0,1],
                    [1,1,1,0]])


#create the test data
test = np.array([1,0,1,1])


#visualize for the memory data
for i in range(np.shape(patterns)[0]):
    print('Memories for ' + repr(i+1) +'th Data point: ', patterns[i])
print('Testing data: ', test)


max_iter = 100
optimizer = SPSA(a=100.0, c=2.0, A=max_iter/10.0, alpha=1.0, gamma=0.167)


print('\nStarting the test!!!')
hopfield = HopfieldNetwork(patterns, tha=0.5, eps=1, optimizer=optimizer)
hopfield.add_perturbation(test, 10)


'''
annealing = int(input('Annealing or not? Please use level 1 ~ 12   '))

try:
    if annealing > 12 and annealing < 1:
        raise NameError('Annealing level: ' + str(annealing))

except NameError:
    print('Input an unsolvalbe annealing level! Please use level 1 ~ 12')
'''
