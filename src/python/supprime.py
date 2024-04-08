import numpy as np

from Bolt_Utils import utils, Log
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R


def rotation(EulerArray, ArrayToRotate) -> np.ndarray:
        Rot = R.from_euler('xyz', EulerArray).as_matrix()
        return Rot@ArrayToRotate




g0 = np.array([0, 0, 10])
g = np.array([1.5, 0.0001, 3])


gg0 = utils.cross(g, g0)
q0 = np.array( [np.linalg.norm(g) * np.linalg.norm(g0) + utils.scalar(g, g0)] )
quat = np.concatenate((gg0, q0), axis=0)
q = R.from_quat( quat )

g_out = q.apply(g0) # bonne orientation mais pas la bonne norme
g_out = g_out/np.linalg.norm(g_out) * np.linalg.norm(g)


print(g)
print(g_out)


grot = rotation(np.array([0.2, 0.2, 0]), g0)

print(grot)
print(np.linalg.norm(grot))







def rotation(EulerArray, ArrayToRotate) -> np.ndarray:
        R = Rotation.from_euler('zyx', EulerArray).as_matrix()
        return R@ArrayToRotate









'''
# quick gain model
def f(x, a, b):
    M = 100
    return (M/b - x)*a

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
