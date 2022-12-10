import numpy as np
import utils


def symmetrize(Mat: np.ndarray):
    """Symmetrize a matrix given an upper or lower triangular one
    """
    return Mat + Mat.T - np.diag(Mat.diagonal())


def toIndex(a: int, b: int, c: int, d: int):
    """Return one compound index given four indices using Yoshimine sort
    """
    if a > b:
        ab = a*(a+1)/2 + b
    else:
        ab = b*(b+1)/2 + a

    if c > d:
        cd = c*(c+1)/2 + d
    else:
        cd = d*(d+1)/2 + c

    if ab > cd:
        abcd = ab*(ab+1)/2 + cd
    else:
        abcd = cd*(cd+1)/2 + ab

    return abcd


def save_data(data: np.ndarray, dest: str):
    """Save data in the dest

    Args:
        data (np.ndarray): data to be saved
        dest (str): destination of the data
    """
    if data.ndim < 3:
        np.savetxt(dest, data)
        return

    with open(dest, 'w') as f:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    for l in range(data.shape[3]):
                        f.write(f'{i+1}\t{j+1}\t{k+1}\t{l+1}\t{data[i, j, k, l]}\n')


def load_data(src: str):
    """Load data from the src

    Args:
        src (str): source of the data
    """
    if not src.endswith('g.dat'):
        return np.loadtxt(src)

    with open(src, 'r') as f:
        np.genfromtxt(src, dtype=None)
        return {toIndex(l[0], l[1], l[2], l[3]): l[4] for l in TEI}
