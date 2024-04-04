import numpy as np

from Bolt_Utils import utils, Log
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation 

A = Rotation.from_euler('zyx', np.array([1, 2, 3])).as_matrix()
b = np.array([2, 3, 5])
print(A@b)

def rotation(EulerArray, ArrayToRotate) -> np.ndarray:
        R = Rotation.from_euler('zyx', EulerArray).as_matrix()
        return R@ArrayToRotate

print( rotation(np.array([1, 2, 3]), b) )

# quick gain model
def f(x, a, b):
    M = 100
    return (M/b - x)*a
'''
n= 20
X = np.linspace(0, 100, n)
Y = f(X, 0.6, 10)
Z = f(X, 0.15, 3)
Z = X + np.linspace(np.zeros(3), n*np.ones(3), n).T

print(Z)

plt.clf()
plt.plot(X, Y, label="Y")
plt.plot(X, Z[0], label="Z")
plt.grid()
plt.legend()
plt.show()

'''
