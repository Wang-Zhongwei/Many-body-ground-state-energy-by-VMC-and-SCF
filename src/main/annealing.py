import random

import numpy as np


def simulated_annealing(initial_state, cost_function, neighborhood_function, temperature_function, max_iterations, **kwargs):
    state = initial_state
    cost = cost_function(state, **kwargs)[0]
    accept = 0
    for i in range(max_iterations):
        temperature = temperature_function(i)
        next_state = neighborhood_function(state)
        next_cost = cost_function(next_state, **kwargs)[0]
        delta_cost = next_cost - cost
        if delta_cost > 0 and random.random() > np.exp(-delta_cost / temperature): 
            continue

        state = next_state
        cost = next_cost
        accept += 1
        print('Accepted count %d: (alpha, beta) = %s, cost = %f' % (accept, state, cost))
    
    print(f'Minimized cost {cost} with acceptance rate {accept/max_iterations}')
    return state



