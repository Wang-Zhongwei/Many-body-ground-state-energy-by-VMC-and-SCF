import sys
import numpy as np


import random

def simulated_annealing(initial_state, cost_function, neighborhood_function, temperature_function, max_iterations):
    state = initial_state
    cost = cost_function(state)
    for i in range(max_iterations):
        temperature = temperature_function(i)
        next_state = neighborhood_function(state)
        next_cost = cost_function(next_state)
        delta_cost = next_cost - cost
        if delta_cost > 0 and random.random() > np.exp(-delta_cost / temperature): 
            continue
        state = next_state
        cost = next_cost
    return state



