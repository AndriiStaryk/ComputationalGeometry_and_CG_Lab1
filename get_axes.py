import numpy as np

def get_axes(B, d):
    invB = np.linalg.inv(B)
    A = invB.T @ invB
    U, D, Vt = np.linalg.svd(A)
    axes = 1 / np.sqrt(D)
    return axes
