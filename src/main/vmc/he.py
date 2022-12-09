# Adapted from morten hjorth-jensen https://github.com/CompPhysics/ComputationalPhysics 
# author: Zhongwei Wang

import math
import sys
from random import random

import numpy

# Read name of output file from command line
if len(sys.argv) != 2:
    print('Usage: python he.py <path/to/output>')
    sys.exit(1)
outfileName = sys.argv[1]

# Initialization function

def initialize():
    number_particles = eval(raw_input('Number of particles: '))
    charge = eval(raw_input('Charge of nucleus: '))
    dimension = eval(raw_input('Dimensionality: '))
    max_variations = eval(
        raw_input('Number of variational parameter values: '))
    thermalisation = eval(raw_input('Number of thermalization steps: '))
    number_cycles = eval(raw_input('Number of MC cycles: '))
    step_length = eval(raw_input('Step length: '))

    return number_particles, charge, dimension, max_variations, thermalisation, number_cycles, step_length

# Trial wave function


def wave_function(r):
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


# Here starts the main program

NUM_OF_PARTICLES = 2
CHARGE = 2
DIMENSION = 3
MAX_VARIATIONS = 10
THERMALIZATION = 10
NUM_OF_CYCLES = 100000
STEP_LENGTH = 1.0

outfile = open(outfileName, 'w')

alpha = 0.5 * CHARGE  # variational parameter

# Step length for numerical differentiation and its inverse squared
h = .001
h2 = 1/(h**2)

r_old = numpy.zeros((NUM_OF_PARTICLES, DIMENSION), numpy.double)
r_new = numpy.zeros((NUM_OF_PARTICLES, DIMENSION), numpy.double)

# Loop over alpha values
for variate in range(MAX_VARIATIONS):

    alpha += .1
    energy = energy2 = 0.0
    accept = 0.0
    delta_e = 0.0

    # Initial position
    for i in range(NUM_OF_PARTICLES):
        for j in range(DIMENSION):
            r_old[i, j] = STEP_LENGTH * (random() - .5)

    wfold = wave_function(r_old)

    # Loop over MC cycles
    for cycle in range(NUM_OF_CYCLES+THERMALIZATION):

        # Trial position
        for i in range(NUM_OF_PARTICLES):
            for j in range(DIMENSION):
                r_new[i, j] = r_old[i, j] + STEP_LENGTH * (random() - .5)

        wfnew = wave_function(r_new)

        # Metropolis test to see whether we accept the move
        if random() < wfnew**2 / wfold**2:
            r_old = r_new.copy()
            wfold = wfnew
            accept += 1

        # If we are done with thermalization, we add to the statistics
        if cycle >= THERMALIZATION:
            delta_e = local_energy(r_old, wfold)
            energy += delta_e
            energy2 += delta_e**2

    # We calculate mean, variance and error ...
    energy /= NUM_OF_CYCLES
    energy2 /= NUM_OF_CYCLES
    variance = energy2 - energy**2
    error = math.sqrt(variance/NUM_OF_CYCLES)

    # ...and write them to file
    outfile.write('%f %f %f %f %f\n' % (alpha, energy, variance,
                  error, accept*1.0/(NUM_OF_CYCLES+THERMALIZATION)))

outfile.close()

print('\nDone. Results are in the file "%s", formatted as:\n\
alpha, <energy>, variance, error, acceptance ratio' % outfileName)
