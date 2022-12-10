import numpy as np

def Sij(i, j, alpha):
    return i*j*alpha

Sij = np.vectorize(Sij)
xx, yy = np.meshgrid(np.arange(1, 4), np.arange(1, 4))
print(Sij(xx, yy, 2))

