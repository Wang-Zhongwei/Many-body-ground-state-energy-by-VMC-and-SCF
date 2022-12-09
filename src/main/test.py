import numpy as np

NUM_OF_BASIS = 2
alphas = np.array([2, 3])
def Sij(i, j, alpha_i, alpha_j):
    return alpha_i * alpha_j * i * j


Sij = np.vectorize(Sij)
# generate mesh
ii, jj = np.meshgrid(np.arange(1, NUM_OF_BASIS+1), np.arange(1, NUM_OF_BASIS+1))
S = Sij(ii, jj, alphas[ii - 1], alphas[jj - 1])

print(S)