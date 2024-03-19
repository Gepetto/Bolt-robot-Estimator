import numpy as np


class utils():
    def __init__():
        pass

    def cross(a,b):
        # return a ^ b
        return np.array([b[1]*a[2] - b[2]*a[1],
                        b[2]*a[0] - b[0]*a[2],
                        b[0]*a[1] - b[1]*a[0]])

    def normalize(a):
        return a/np.linalg.norm(a)
    
    def MatrixFromVectors(L):
        # L tuple of numpy vectors

        A = np.stack(L).reshape((-1, 3))

        return A