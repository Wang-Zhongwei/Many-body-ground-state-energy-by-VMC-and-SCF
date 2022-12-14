import random

import numpy as np


def simulated_annealing(initial_state, cost_function, neighborhood_function, temperature_function, max_iterations, **kwargs):
    """_summary_

    Args:
        initial_state (array-like): initial parameter set
        cost_function (function): cost function to be minimized
        neighborhood_function (function): map current state to a random next state, where step length dependent on iteration number
        temperature_function (function): gradually reduced over time in order to maintain the ground state
        max_iterations (int): max interaction of the monte carlo numbers

    Returns:
        array: history of optimized (state, cost)
    """
    state = initial_state
    cost = cost_function(state, **kwargs)[0]
    history = [state, cost]
    accept = 0
    for i in range(max_iterations):
        temperature = temperature_function(i)
        next_state = neighborhood_function(state)
        next_cost = cost_function(next_state, **kwargs)[0]
        delta_cost = next_cost - cost
        if delta_cost > 0 and random.random() > np.exp(-delta_cost / temperature): 
            continue
        
        history.append(state = next_state, cost = next_cost)
        accept += 1
        print('Accepted count %d: (alpha, beta) = %s, cost = %f' % (accept, state, cost))
    
    print(f'Minimized cost {cost} with acceptance rate {accept/max_iterations}')
    return history



