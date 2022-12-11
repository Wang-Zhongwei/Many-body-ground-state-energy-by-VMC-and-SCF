
import numpy as np

def STO1(alpha: float, r: tuple):
    r_abs = np.sqrt(sum(x**2 for x in r))
    # normalization is unnecessary for VMC
    return np.exp(-alpha * r[0])

def STO2(alpha: float, r: tuple):
    r_abs = np.sqrt(sum(x**2 for x in r))
    return r * np.exp(-alpha * r[0] / 2)