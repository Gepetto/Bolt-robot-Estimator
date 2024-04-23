import numpy as np

from Bolt_Utils import utils, Log
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R


def rotation(EulerArray, ArrayToRotate) -> np.ndarray:
        Rot = R.from_euler('xyz', EulerArray).as_matrix()
        return Rot@ArrayToRotate

def f(x):
    center = 1
    b0 = 5
    b = b0/center
    return 1/ (1 + np.exp(-b*x + b0))

RobotWeight = 10
acc_coeff = .4
N = 20

MinForce = 0
MaxForce = RobotWeight*9.81 * (1+acc_coeff)
upper = int(np.floor(N*(1+acc_coeff)))
lower = int(np.ceil(N*(1-acc_coeff/2)))

print(f'upper : {upper}, lower : {lower}')

PossibleForce_Left = np.linspace(MinForce, MaxForce, upper)
PossibleForce_Right = PossibleForce_Left.copy()
print(PossibleForce_Left)


k = 0
for i in range(upper):
    FL = PossibleForce_Left[i]
    ReasonnableMin = max(0, lower-i)
    ReasonnableMax = max(0, upper-i)
    for j in range( ReasonnableMin, ReasonnableMax):
        weight = PossibleForce_Left[i] + PossibleForce_Right[j]
        print(f'Indices :  {i} ; {j}     Valeur :  {np.floor(weight)}       Répartition :  {np.floor(PossibleForce_Left[i])}/{np.floor(PossibleForce_Right[j])}')
        k +=1
        
        
print(f'iteration totale : {k}')
print(f'iteration sans optim : {len(PossibleForce_Left)**2}')















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
        r = R.from_euler('zyx', EulerArray).as_matrix()
        return r@ArrayToRotate









'''
# quick gain model
def f(x, a, b):
    M = 100
    return (M/b - x)*a
'''
n= 100
X = np.linspace(-1, 3, n)
Y = f(X)
#Z = f(X, 0.15, 3)
#Z = X + np.linspace(np.zeros(3), n*np.ones(3), n).T


plt.figure(dpi=200)
plt.plot(X, Y, label="Y")
#plt.plot(X, Z[0], label="Z")
plt.grid()
plt.legend()
plt.show()










