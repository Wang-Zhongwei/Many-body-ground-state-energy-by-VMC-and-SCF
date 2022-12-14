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
            return 32 * np.sqrt(2* alphas[i - 1]**3 * alphas[j - 1]**3) * (alphas[i - 1] - alphas[j - 1])/ (2 * alphas[i - 1] + alphas[j - 1])**4

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
            return 4 * alphas[i - 1] * alphas[j - 1] * (4*alphas[i - 1] - alphas[j - 1]) * np.sqrt(2 * alphas[i - 1]**3 * alphas[j - 1]**3) / (2 * alphas[i - 1] + alphas[i - 1])**4
    if i == 2:
        if j == 2:
            return alphas[j - 1]**2 / 8

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
            return 4 * Z * (-2 * alphas[i - 1] + alphas[j - 1]) * np.sqrt(2 * alphas[i - 1]**3 * alphas[j - 1]**3) / (2 * alphas[i - 1] + alphas[j - 1])**3
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
        return -32 * alphas[0] * np.sqrt(2*alphas[0]**3 * alphas[1]**3) * (-264*alphas[0]**4 + 28*alphas[0]**3 * alphas[1] + 86*alphas[0]**2*alphas[1]**2 + 21*alphas[0]*alphas[1]**3 + alphas[1]**4) / (2*alphas[0] + alphas[1])**3 / (6*alphas[0] + alphas[1])**4
    elif index == 17: # 2 2 1 1 
        return  alphas[0] * alphas[1] * (8*alphas[0]**4 + 20*alphas[0]**3 * alphas[1] + 12*alphas[0]**2 * alphas[1]**2 + 10*alphas[0]*alphas[1]**3 + alphas[1]**4)/ (2*alphas[0] + alphas[1])**5
    elif index == 14: # 2 1 2 1
        return 16 * alphas[0]**3 * alphas[1]**3 * (20*alphas[0]**2 - 30*alphas[0]*alphas[1] + 13*alphas[1]**2)/ (2*alphas[0] + alphas[1])**7
    elif index == 19: # 2 2 2 1
        return 8 * np.sqrt(2*alphas[0]**3 * alphas[1]**3) * alphas[1] * (32*alphas[0]**6 + 240*alphas[0]**5*alphas[1] + 544 *alphas[0]**4 * alphas[1]**2+ 680*alphas[0]**3 * alphas[1]**3 - 102*alphas[0]**2 * alphas[1]**4 - 729*alphas[0]*alphas[1]**5 - 345*alphas[1]**6) / (2*alphas[0] + alphas[1])**3 / (2*alphas[0] + 3*alphas[1])**6
    elif index == 20: # 2 2 2 2
        return 77/512 * alphas[1]

# vectorize matrix calculation
Sij = np.vectorize(Sij, excluded=['alphas'])
Tij = np.vectorize(Tij, excluded=['alphas'])
Vij = np.vectorize(Vij, excluded=['alphas'])
tei = np.vectorize(tei, excluded=['alphas'])


def get2D(name: str, alphas: tuple):
    set = {'S', 'T', 'V'}
    if (name not in set):
        raise ValueError("name must be one of {}".format(set))

    num_of_basis = len(alphas)
    rgn = np.arange(1, num_of_basis + 1)
    ii, jj = np.meshgrid(rgn, rgn)
    if name == 'S':
        return Sij(ii, jj, alphas=alphas)
    elif name == 'T':
        return Tij(ii, jj, alphas=alphas)
    else:
        return Vij(ii, jj, alphas=alphas)
    
def get4D(name: str, alphas: tuple, use_Yoshimine: bool = False):
    set = {'G'}
    if (name not in set):
        raise ValueError("name must be one of {}".format(set))

    num_of_basis = len(alphas)
    rgn = np.arange(1, num_of_basis + 1)
    ii, jj, kk, ll = np.meshgrid(rgn, rgn, rgn, rgn)
    if name == 'G':
        ret = tei(ii, jj, kk, ll, alphas=alphas)
    
    
    if not use_Yoshimine:
        return ret
    set = {}
    for i in range(num_of_basis):
        for j in range(num_of_basis):
            for k in range(num_of_basis):
                for l in range(num_of_basis):
                    set[utils.toIndex(i + 1, j + 1, k + 1, l + 1)] = ret[i, j, k, l]
    return set 
       
    
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
    ALPHAS = np.array(sys.argv[2:], dtype=float)
    NUM_OF_BASIS = len(ALPHAS)

    S = get2D('S', ALPHAS)
    T = get2D('T', ALPHAS)
    V = get2D('V', ALPHAS)
    G = get4D('G', ALPHAS)

    # save data
    utils.save_data(ALPHAS, output_dir + "/alphas.dat")
    utils.save_data(S, output_dir + "/S.dat")
    utils.save_data(T, output_dir + "/T.dat")
    utils.save_data(V, output_dir + "/V.dat")
    utils.save_data(G, output_dir + "/G.dat")
