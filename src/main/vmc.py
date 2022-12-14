# Adapted from morten hjorth-jensen https://github.com/CompPhysics/ComputationalPhysics 
# author: Zhongwei Wang

import logging
import math
import sys
from functools import partial
from random import random

import numpy as np
from annealing import *

logging.basicConfig(level=logging.INFO)

# Read name of output file from command line
if len(sys.argv) != 2:
    print('Usage: python he.py <out_file>')
    sys.exit(1)
outfileName = sys.argv[1]

r_abs = lambda r: np.sqrt(np.sum(r**2, axis=-1))
# Trial wave function
def wave_function(r: np.ndarray, alpha: float, beta: float):

    # single body part
    wf = np.exp(-alpha * np.sum(r_abs(r)))

    # two body part
    for i in range(r.shape[0]):
        for j in range(i+1, r.shape[0]):
            r_ij = r_abs(r[i] - r[j])
            wf *= np.exp(r_ij / 2 / (1 + beta * r_ij))
    return wf


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
        r_single_particle = r_abs(r[i])
        e_potential -= CHARGE/r_single_particle

    # Electron-electron contribution
    for i1 in range(NUM_OF_PARTICLES-1):
        for i2 in range(i1+1, NUM_OF_PARTICLES):
            r_12 = r_abs(r[i1] - r[i2])
            e_potential += 1/r_12

    return e_potential + e_kinetic

def evaluate_energy(alpha, beta, num_cycles: int, thermalization: int):
    # Bind wave function to wf_params
    # alpha, beta = wf_params
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
    logging.info(f'(alpha, beta) = ({alpha}, {beta}); energy = {energy}, error = {error}, acceptance = {acceptance}')

    return energy, error ,acceptance


# Here starts the main program
NUM_OF_PARTICLES = 2
CHARGE = 2
DIMENSION = 3
MAX_VARIATIONS = 10
THERMALIZATION = 10
NUM_OF_CYCLES = 10000
ELEC_STEP_LENGTH = 1.0
PARAM_STEP_LENGTH = 0.1

alpha = 0.5 * CHARGE  # variational parameter

# Step length for numerical differentiation and its inverse squared
h = .001
h2 = 1/(h**2)

initial_state = np.array([1.5, 0]) # initial state of (alpha, beta) 
neighbor_func = lambda v: v + np.random.uniform(-1, 1, 2) * PARAM_STEP_LENGTH
# simulated_annealing(initial_state, evaluate_energy, neighbor_func, temperature_function=lambda i: 0.05/(i+1), max_iterations=20, num_cycles=NUM_OF_CYCLES, thermalization=THERMALIZATION)

# vectorize simulated_annealing 
evaluate_energy = np.vectorize(evaluate_energy, excluded=['num_cycles', 'thermalization'])
alpha_rgn = np.linspace(1.5, 2.0, 11)
beta_rgn = np.linspace(0, 2, 11)
alpha_mesh, beta_mesh = np.meshgrid(alpha_rgn, beta_rgn)


evaluate_energy(alpha_mesh, beta_mesh, NUM_OF_CYCLES, THERMALIZATION) # forced to change interface...