# Adapted from Dr Adam Luke Baskervile's code https://adambaskerville.github.io/posts/HartreeFockGuide/#the-hartree-fock-procedure
# author: Zhongwei Wang
import os
import subprocess
import sys

import numpy as np
from int import *
import utils


def tei(a, b, c, d, TEI):
    """Return value of two electron integral <ab|1/r12|cd> where r12 is the distance between two electrons

    Args:
        a (int): phi_a(r1) as in <ab|1/r12|cd>
        b (int): phi_b(r2)
        c (int): phi_c(r1)
        d (int): phi_d(r2)

    Returns:
        float: <ab|1/r12|cd> where r12 = |r1 - r2|
    """
    return TEI.get(utils.toIndex(a, b, c, d), 0)


def toFPrime(X: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Put Fock matrix in orthonormal AO basis
    """
    return np.dot(np.transpose(X), np.dot(F, X))


# Make density matrix and store old one to test for convergence
def makeDensity(C: np.ndarray, dim: int, Nelec: int) -> np.ndarray:
    """Return an updated density matrix given the new coefficient matrix C

    Args:
        C (np.ndarray): coefficient matrix
        D (np.ndarray): density matrix
        dim (int): dimension
        Nelec (int): number of electrons in the system

    Returns:
        np.ndarray: updated density matrix
    """
    den = np.zeros((dim, dim))
    for mu in range(dim):
        for nu in range(dim):
            for m in range(Nelec//2):
                den[mu, nu] += 2*C[mu, m]*C[nu, m]

    return den


def makeFock(H: np.ndarray, P: np.ndarray, dim: int, TEI) -> np.ndarray:
    """Generate Fock matrix including both single body and many body terms

    Args:
        H (array): single body term representing nucleus-electron attraction
        P (array): density matrix that generates many-body terms
        dim (int): number of basis functions

    Returns:
        array: Fock matrix
    """
    fock = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            fock[i, j] = H[i, j]
            for k in range(dim):
                for l in range(dim):
                    fock[i, j] += P[k, l] * \
                        (tei(i+1, j+1, k+1, l+1, TEI)-0.5*tei(i+1, k+1, j+1, l+1, TEI))

    return fock


def deltaD(D: np.ndarray, D_old: np.ndarray) -> float:
    """Calculate change in density matrix using Root Mean Square Deviation (RMSD).
    Note that det(D) == 1. So we don't need to divide by det(D) in the denominator.

    Args:
        D (np.ndarray): density matrix
        D_old (np.ndarray): old density matrix

    Returns:
        float: root mean square deviation of density matrix
    """
    delta = 0.0
    for i in range(dim):
        for j in range(dim):
            delta = delta + ((D[i, j] - D_old[i, j])**2)

    return np.sqrt(np.sum((D - D_old)**2))


def calcEnergy(D, Hcore, F):
    # dot wise multiplication
    return 0.5 * np.sum(D * (Hcore+F))


def scf(resource_dir: str, data_dir: str, num_elec: int, alphas: tuple):
    """Self consistent field (SCF) procedure where STO basis set is used.
    You can use closed shell assumption where num_electrons = 2 * len(alphas)

    Args:
        resource_dir (str): resource directory that contains Fock matrix elements
        data_dir (str): data directory where convergence 
        num_elec (int): number of electrons_
        alphas (tuple): alpha arguments in basis functions
    """


    dim = len(alphas)  # dim is the number of basis functions
    if dim < num_elec // 2:
        raise ValueError(
            f"Number of basis functions {dim} is less than half number of electrons {num_elec // 2}. System forbidden by Pauli Exclusion Principle!")

    # TODO: resolve type of atom by num_elec and append to data-dir accordingly

    S = Sij()
    S = utils.load_data(os.path.join(resource_dir, 's.dat'))
    T = utils.load_data(os.path.join(resource_dir, 't.dat'))
    V = utils.load_data(os.path.join(resource_dir, 'v.dat'))
    TEI = utils.load_data(os.path.join(resource_dir, 'g.dat'))

    # solve hf equations recursively
    H_core = T + V  # Form core Hamiltonian matrix as sum of one electron kinetic energy, T and potential energy, V matrices
    # Diagonalize basis using symmetric orthogonalization
    SVAL, SVEC = np.linalg.eigh(S)
    # Inverse square root of eigenvalues
    SVAL_minhalf = (np.diag(1/np.sqrt(SVAL)))
    X = np.dot(SVEC, np.dot(SVAL_minhalf, np.transpose(SVEC)))
    # P represents the density matrix, Initially set to zero.
    P_new = np.zeros((dim, dim))
    delta = 1  # Set placeholder value for delta
    cnt = 0  # Count how many SCF cycles are done, N(SCF)

    file = os.path.join(out_dir, 'Be.dat')
    # open file
    with open(file, 'w') as f:
        # write header
        f.write('Iteration\tEnergy\tDelta\n')
        while delta > 0.00001:
            cnt += 1                             # Add one to number of SCF cycles counter
            F = makeFock(H_core, P_new, dim, TEI)        # Calculate Fock matrix, F
            # Calculate transformed Fock matrix, F'
            F_prime = toFPrime(X, F)
            # Diagonalize F' matrix
            _, C_prime = np.linalg.eigh(F_prime)
            # 'Back transform' the coefficients into original basis using transformation matrix
            C = np.dot(X, C_prime)
            P_old = P_new
            P_new = makeDensity(C, dim, Nelec)  # Make density matrix
            # Test for convergence. If criteria is met exit loop and calculate properties of interest
            delta = deltaD(P_new, P_old)

            # write to file
            current_energy = calcEnergy(P_new, H_core, F)
            f.write('{}\t{:.6f}\t{:.6f}\n'.format(
                cnt, current_energy, delta))

        print("SCF procedure complete, TOTAL E(SCF) = {} hartrees".format(
            current_energy))
        print("All iteration data written to {}".format(file))

    # return energy
    return calcEnergy(P_new, H_core, F)


def update_resource(resource_dir: str, alphas: tuple):
    """Feed alphas to int.py and update resources in resource_dir

    Args:
        resource_dir (str): resource directory that contains Fock matrix elements
        alphas (tuple): alpha arguments in basis functions
    """

    # convert alphas to tuple of strings so that it can be passed to subprocess
    alphas = tuple(map(str, alphas))

    # fix: change to relative path later
    script = "src/main/hf/int.py"
    # how to recognize the path of int.py?
    proc = subprocess.Popen(['python', script, resource_dir, *alphas])
    proc.wait()
    print('Resources have been updated for alphas = ', alphas)


if __name__ == '__main__':

    # get from input resource directory and data directory
    if sys.argv[1] == '-h' or sys.argv[1] == '--help':
        print('Usage: python scf.py <path/to/source/folder> <path/to/data/folder>')
        sys.exit(0)

    if len(sys.argv) != 3:
        print('Usage: python scf.py <path/to/source/folder> <path/to/data/folder>')
        sys.exit(1)

    resource_dir = sys.argv[1]
    out_dir = sys.argv[2]

    # define dimension of the system or take from inputs
    Nelec = 4  # The number of electrons in our system
    dim = 2  # dim is the number of basis functions

    # initialize alphas
    alphas = np.array([3, 1])
    # update resources
    update_resource(resource_dir, alphas)
    # initialize number of trial steps
    steps = 100
    # initial step length
    init_len = 0.5
    # final step length aka prevision for alpha
    final_len = 0.005
    # penalty factor for energy increase, equivalent to temperature reciprocal
    penalty = 100

    # use exponential decay learning rate
    def step_length(cur_step, init_len, final_len):
        return init_len * np.exp(cur_step / steps * np.log(final_len / init_len))

    
    # Use simulated annealing to find optimized alphas
    current_energy = float('inf')
    accept = 0
    for i in range(steps):
        # get step length
        step_len = step_length(i, init_len, final_len)
        # get new alphas
        trial_alphas = alphas + 2 * \
            (np.random.rand(*alphas.shape) - 0.5) * step_len

        # update resources
        update_resource(resource_dir, trial_alphas)
        # run scf
        trial_energy = scf(resource_dir, out_dir, Nelec, trial_alphas)

        if (trial_energy > current_energy and
                np.random.rand() > np.exp(-penalty * (trial_energy - current_energy) / np.abs(current_energy))):
            # reject the trial
            continue

        # accept the trial
        accept += 1
        current_energy = trial_energy
        alphas = trial_alphas

        print(
            f'Current energy: {current_energy:.6f} at alphas: {alphas} at step {i}')

    print('Acceptance rate: ', accept / steps)
    print('Optimized alphas: ', alphas)
    print('Minimized energy: ', current_energy)

    # TODO: save annealing data and final result to file
