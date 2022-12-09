# Author: Zhongwei

# Inspiration: time consuming part is not how to write the code, but to design the interfaces

# calculate matrix and output to file
import numpy as np
import utils


def Sij(i: int, j: int, alphas: tuple):
    """1-indexed overlap matrix of basis functions
    S_ij := <phi_i|phi_j> 

    Args:
        i (int): row
        j (int): column
        alphas (tuple): alphas in the basis functions

    Returns:
        _float: S_ij
    """    
    if i == j: 
        return 1
    if i > j:
        return Sij(j, i, alphas=alphas)
    if i == 1:
        if j == 2:
            return 16 * np.sqrt(6 * alphas[i - 1]**3 * alphas[j - 1]**5) / (2 * alphas[i - 1] + alphas[j - 1])**4

def Tij(i: int, j: int, alphas: tuple):
    """1-indexed kinetic energy matrix of basis functions
    T_ij := <phi_i| -1/2 * del^2 |phi_j> 

    Args:
        i (int): row
        j (int): column
        alphas (tuple): alphas in the basis functions

    Returns:
        _float: T_ij
    """    
    if i > j:
        return Tij(j, i, alphas=alphas)
    if i == 1:
        if j == 1:
            return alphas[i - 1]**2 / 2
        if j == 2:
            return -8 * alphas[i - 1] * (alphas[i - 1] - alphas[j - 1]) * np.sqrt(2/3 * alphas[i - 1]**3 * alphas[j - 1]**5) / (2 * alphas[i - 1] + alphas[i - 1])**4
    if i == 2:
        if j == 2:
            return alphas[j - 1]**2 / 2

def Vij(i: int, j: int, alphas: float, Z: float = 4):
    """1-indexed potential energy matrix of basis functions
    V_ij := <phi_i| -Z/r |phi_j> 

    Args:
        i (int): row
        j (int): column
        alphas (tuple): alphas in the basis functions

    Returns:
        _float: V_ij
    """    
    if i > j:
        return Vij(j, i, alphas=alphas)
    if i == 1:
        if j == 1:
            return -Z * alphas[i - 1]
        if j == 2:
            return -8 * Z *np.sqrt(2/3 * alphas[i - 1]**3 * alphas[j - 1]**5) / (2 * alphas[i - 1] + alphas[j - 1])**3
    if i == 2:
        if j == 2:
            return -Z * alphas[j - 1] / 4

def tei(i: int, j: int, k: int, l: int, alphas: tuple):
    """1-indexed two-electron integrals
    g_ijkl := (phi_i^*(r1) phi_j(r1)| phi_k^*(r2) phi_l(r2)) 

    Returns:
        _float: g_ijkl
    """    
    index = utils.toIndex(i, j, k, l)
    if index == 5: # 1 1 1 1
        return 5/8 * alphas[0] 
    elif index == 12: # 2 1 1 1
        return -8 * np.sqrt(2/3 * alphas[0]**3 * alphas[1]**5) * (-1/(2*alphas[0] + alphas[1])**3 + (12*alphas[0] + alphas[1])/(6*alphas[0] + alphas[1])**4)
    elif index == 17: # 2 2 1 1 
        return 1/4 * alphas[1]**5 * (1/alphas[1]**4 - (6*alphas[0] + alphas[1]) / (2*alphas[0] + alphas[1])**4)
    elif index == 14: # 2 1 2 1
        return 176 * alphas[0]**3 * alphas[1]**5 / (2*alphas[0] + alphas[1])**7
    elif index == 19: # 2 2 2 1
        return -np.sqrt(alphas[0]**3 * alphas[1]**15 / 6) * (-144/alphas[1]**4 + 768*(84*alphas[0]**2 + 140*alphas[0]*alphas[1] + 61*alphas[1]**2) / (2*alphas[0] + 3*alphas[1])**6) / (6 * (2*alphas[0] + alphas[1])**4)
    elif index == 20: # 2 2 2 2
        return 93/512 * alphas[1]

# vectorize matrix calculation
Sij = np.vectorize(Sij, excluded=['alphas'])
Tij = np.vectorize(Tij, excluded=['alphas'])
Vij = np.vectorize(Vij, excluded=['alphas'])
tei = np.vectorize(tei, excluded=['alphas'])

# generate data
if __name__ == "__main__":
    import sys
    from functools import partial

    if sys.argv[1] == '-h' or sys.argv[1] == '--help':
        print("Usage: python3 int.py [output_dir] [alpha1] [alpha2] ...")
        exit(0)

    if len(sys.argv) < 3:
        print("Usage: python3 int.py [output_dir] [alpha1] [alpha2] ...")
        exit(1)

    output_dir = sys.argv[1]
    alphas = np.array(sys.argv[2:], dtype=float)
    NUM_OF_BASIS = len(alphas)

    # save alphas 
    utils.save_data(alphas, output_dir + "/alphas.dat")

    # generate 2d mesh
    rgn = np.arange(1, NUM_OF_BASIS+1)
    ii, jj = np.meshgrid(rgn, rgn)

    # generate 2d matrices
    S = Sij(ii, jj, alphas=alphas)
    T = Tij(ii, jj, alphas=alphas)
    V = Vij(ii, jj, alphas=alphas)

    # generate 4d mesh 
    ii, jj, kk, ll = np.meshgrid(rgn, rgn, rgn, rgn)
    # generate 4d matrices
    g = tei(ii, jj, kk, ll, alphas=alphas)

    utils.save_data(S, output_dir + "/S.dat")
    utils.save_data(T, output_dir + "/T.dat")
    utils.save_data(V, output_dir + "/V.dat")
    utils.save_data(g, output_dir + "/g.dat")
