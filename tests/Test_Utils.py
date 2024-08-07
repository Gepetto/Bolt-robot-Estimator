import numpy as np
from scipy.spatial.transform import Rotation as R
import time as t




from bolt_estimator.utils.Utils import Log, utils



q1 = np.array([0, 0, 0, 1])

q2 = utils.InvQuat(q1)
print("inverting unit quaternion :")
print(q2)
q3 = utils.QuatProduct(q1, q2)
print("product of unit and inverse unit quaternion :")
print(q3)
print("\n")




q4 = np.array([1, -2, 1, 3])
q5 = np.array([-1, 2, 3, 2])

print("left and right product by unit quaternion :")
print(utils.QuatProduct(q1, q5))
print(utils.QuatProduct(q5, q1))

print("left and right product by null quaternion :")
q0 = np.zeros((4,))
print(utils.QuatProduct(q0, q5))
print(utils.QuatProduct(q5, q0))
print("\n")





print("product of two quaternions by their inverse :")
print(utils.QuatProduct(q4, utils.InvQuat(q4)))
print(utils.QuatProduct(q5, utils.InvQuat(q5)))
# expect -9, -2, 11, 8
print("product of two quaternions, expect -9, -2, 11, 8")
print(utils.QuatProduct(q4, q5))
q = np.array([0, 1, 0, 1])
r = np.array([0.5, 0.5, 0.75, 1])
# expect 1.25, 1.5, 0.25, 0.5
print("product of two quaternions, expect 1.25, 1.5, 0.25, 0.5")
print(utils.QuatProduct(q, r))
# expect 0, 2, 0, 0
print("product of two quaternions, expect 0, 2, 0, 0")
print(utils.QuatProduct(q, q))
print("\n")





x = np.array([5, 6, 7])
print("rotation by unite quaternion :")
print(x)
print(utils.RotByQuat(x, q1))
print("\n")





x = np.array([1., 0, 0])
print("rotation by unitary quaternion about z, y, and x axis by pi/4 :")
print(x)
q6 = np.array([0.0, 0.0,  np.sqrt(2)/2,      np.sqrt(2)/2])
print("z", utils.RotByQuat(x, q6))

q6 = np.array([0.0, np.sqrt(2)/2, 0.0,       np.sqrt(2)/2])
print("y", utils.RotByQuat(x, q6))

q6 = np.array([np.sqrt(2)/2, 0.0, 0.0,       np.sqrt(2)/2])
print("x", utils.RotByQuat(x, q6))
print("\n")





print("rotation by unitary quaternion about z, y, and x axis by pi/3 :")
print(x)
q6 = np.array([0.0, 0.0,  1/2,      np.sqrt(3)/2])
print("z", utils.RotByQuat(x, q6))

q6 = np.array([0.0, 1/2, 0.0,       np.sqrt(3)/2])
print("y", utils.RotByQuat(x, q6))

q6 = np.array([1/2, 0.0, 0.0,       np.sqrt(3)/2])
print("x", utils.RotByQuat(x, q6))
print("\n")










