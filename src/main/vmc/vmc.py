# Adapted from morten hjorth-jensen https://github.com/CompPhysics/ComputationalPhysics 
# author: Zhongwei Wang

import math
import sys
from random import random

import numpy as np

# Read name of output file from command line
if len(sys.argv) != 2:
    print('Usage: python he.py <out_file>')
    sys.exit(1)
outfileName = sys.argv[1]

# Trial wave function
def wave_function(r: np.ndarray):
    argument = 0.0

    for i in range(NUM_OF_PARTICLES):
        r_single_particle = 0.0
        for j in range(DIMENSION):
            r_single_particle += r[i, j]**2
        argument += math.sqrt(r_single_particle)

    return math.exp(-argument*alpha)

# Local energy (numerical derivative)
# the argument wf is the wave function value at r (so we don't need to calculate it again)


def local_energy(r, wf):
    # Kinetic energy
    r_plus = r.copy()
    r_minus = r.copy()

    e_kinetic = 0.0

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

def evaluate_energy(initial_state: np.ndarray, wave_function, neighborhood_function, num_cycles: int, thermalization: int):
    # initialize state and wave function
    state = initial_state
    wf = wave_function(state)
    # Initialize energy and energy squared
    energy = energy2 = 0.0
    # Initialize acceptance counter
    accept = 0
    for cycle in range(num_cycles+thermalization):
        # Trial position
        next_state = neighborhood_function(state)
        next_wf = wave_function(next_state)
        # Metropolis test to see whether we accept the move
        if random() < next_wf**2 / wf**2:
            state = next_state
            wf = next_wf
            accept += 1
        # If we are done with thermalization, we add to the statistics
        if cycle >= thermalization:
            delta_e = local_energy(state, wf)
            energy += delta_e
            energy2 += delta_e**2
    # Return the average energy and its error and acceptance rate
    energy /= num_cycles
    energy2 /= num_cycles
    error = np.sqrt(energy2 - energy**2)
    return energy, error, accept/num_cycles


# Here starts the main program
NUM_OF_PARTICLES = 2
CHARGE = 2
DIMENSION = 3
MAX_VARIATIONS = 10
THERMALIZATION = 10
NUM_OF_CYCLES = 100000
STEP_LENGTH = 1.0

alpha = 0.5 * CHARGE  # variational parameter

# Step length for numerical differentiation and its inverse squared
h = .001
h2 = 1/(h**2)

r_old = np.zeros((NUM_OF_PARTICLES, DIMENSION), np.double)
r_new = np.zeros((NUM_OF_PARTICLES, DIMENSION), np.double)

# Loop over alpha values
outfile = open(outfileName, 'w')
for variate in range(MAX_VARIATIONS):

    alpha += .1
    energy = energy2 = 0.0
    accept = 0.0
    delta_e = 0.0

    # Initial position
    initial_state = STEP_LENGTH * np.random.uniform(-1, 1, (NUM_OF_PARTICLES, DIMENSION))

    energy, error, accept = evaluate_energy(initial_state, wave_function, lambda r: r + STEP_LENGTH * np.random.uniform(-1, 1, (NUM_OF_PARTICLES, DIMENSION)), NUM_OF_CYCLES, THERMALIZATION)

    # For a given alpha, write results to file
    outfile.write('%f %f %f %f\n' % (alpha, energy,
                    error, accept*1.0/(NUM_OF_CYCLES+THERMALIZATION)))
outfile.close()

print('\nDone. Results are in the file "%s", formatted as:\n\
alpha, <energy>, variance, error, acceptance ratio' % outfileName)
