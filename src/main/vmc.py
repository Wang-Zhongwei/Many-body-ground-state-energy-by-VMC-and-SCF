# Adapted from morten hjorth-jensen https://github.com/CompPhysics/ComputationalPhysics 
# author: Zhongwei Wang

import math
import sys
from random import random
from functools import partial

import numpy as np
from annealing import *

# Read name of output file from command line
if len(sys.argv) != 2:
    print('Usage: python he.py <out_file>')
    sys.exit(1)
outfileName = sys.argv[1]

# Trial wave function
def wave_function(r: np.ndarray, alpha, beta):
    argument = 0.0

    for i in range(NUM_OF_PARTICLES):
        r_single_particle = 0.0
        for j in range(DIMENSION):
            r_single_particle += r[i, j]**2
        argument += math.sqrt(r_single_particle)

    return math.exp(-argument*alpha)

# Local energy (numerical derivative)
# the argument wf is the wave function value at r (so we don't need to calculate it again)


def local_energy(r, wave_function):
    # Kinetic energy
    r_plus = r.copy()
    r_minus = r.copy()

    e_kinetic = 0.0

    wf = wave_function(r)
    for i in range(NUM_OF_PARTICLES):
        for j in range(DIMENSION):
            r_plus[i, j] = r[i, j] + h
            r_minus[i, j] = r[i, j] - h
            wf_minus = wave_function(r_minus)
            wf_plus = wave_function(r_plus)
            e_kinetic -= wf_minus+wf_plus-2*wf
            r_plus[i, j] = r[i, j]
            r_minus[i, j] = r[i, j]

    e_kinetic = .5*h2*e_kinetic/wf

    # Potential energy
    e_potential = 0.0

    # Electron-proton contribution
    for i in range(NUM_OF_PARTICLES):
        r_single_particle = 0.0
        for j in range(DIMENSION):
            r_single_particle += r[i, j]**2
        e_potential -= CHARGE/math.sqrt(r_single_particle)

    # Electron-electron contribution
    for i1 in range(NUM_OF_PARTICLES-1):
        for i2 in range(i1+1, NUM_OF_PARTICLES):
            r_12 = 0.0
            for j in range(DIMENSION):
                r_12 += (r[i1, j] - r[i2, j])**2
            e_potential += 1/math.sqrt(r_12)

    return e_potential + e_kinetic

def evaluate_energy(wf_params, num_cycles: int, thermalization: int):
    # Bind wave function to wf_params
    alpha, beta = wf_params
    concrete_wave_func = partial(wave_function, alpha=alpha, beta=beta)

    # Initialization
    r = ELEC_STEP_LENGTH * np.random.uniform(-1, 1, (NUM_OF_PARTICLES, DIMENSION))
    wf = concrete_wave_func(r)
    energy = energy2 = 0.0
    accept = 0

    for cycle in range(num_cycles + thermalization):
        # Trial position
        r_trial = r + ELEC_STEP_LENGTH * np.random.uniform(-1, 1, (NUM_OF_PARTICLES, DIMENSION))
        wf_trial = concrete_wave_func(r_trial)

        # Metropolis test to see whether we accept the move
        if np.random.rand() < wf_trial**2 / wf**2:
            r = r_trial
            wf = wf_trial
            accept += 1

        # Add to stats when done with thermalization
        if cycle >= thermalization:
            delta_e = local_energy(r, concrete_wave_func)
            energy += delta_e
            energy2 += delta_e**2

    # Calc average energy, error and acceptance rate
    energy /= num_cycles
    energy2 /= num_cycles
    error = np.sqrt(energy2 - energy**2)
    acceptance = accept / (num_cycles + thermalization)

    return energy, error ,acceptance


# Here starts the main program
NUM_OF_PARTICLES = 2
CHARGE = 2
DIMENSION = 3
MAX_VARIATIONS = 10
THERMALIZATION = 10
NUM_OF_CYCLES = 100000
ELEC_STEP_LENGTH = 1.0
PARAM_STEP_LENGTH = 0.1

alpha = 0.5 * CHARGE  # variational parameter

# Step length for numerical differentiation and its inverse squared
h = .001
h2 = 1/(h**2)

initial_state = np.array([1.5, 0]) # initial state of (alpha, beta) 
neighbor_func = lambda v: v + np.random.uniform(-1, 1, 2) * PARAM_STEP_LENGTH
simulated_annealing(initial_state, evaluate_energy, neighbor_func, temperature_function=lambda i: 0.05/(i+1), max_iterations=20, num_cycles=NUM_OF_CYCLES, thermalization=THERMALIZATION)