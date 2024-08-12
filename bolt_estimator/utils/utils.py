import numpy as np
from scipy.spatial.transform import Rotation 

'''
Class with some function used in other objects
'''
class utils():
    def __init__(self):
        pass

    def cross(a,b):
        # returns a ^ b [PINOCCHIO INCLUDES A CROSSPRODUCT]
        return np.array([b[1]*a[2] - b[2]*a[1],
                        b[2]*a[0] - b[0]*a[2],
                        b[0]*a[1] - b[1]*a[0]])

    def normalize(a):
        return a/np.linalg.norm(a)
    
    def normalizeQ(q:np.ndarray):
        q_out = q # copy position, quaternion and joints state
        q_out[3:7] = q[3:7]/np.linalg.norm(q[3:7]) # quaternion normalization
        return q_out

    
    def MatrixFromVectors(L, n=3):
        # L tuple of numpy vectors
        A = np.stack(L).reshape((-1, n))
        return A
    
    def scalar(a, b) -> float:
        # returns a scalar b
        return np.sum(a*b)

    def RotationMatrix(euler_array) -> np.ndarray:
        return Rotation.from_euler('zyx', euler_array).as_matrix()
    
    def rotation(euler_array, array_to_rotate) -> np.ndarray:
        R = Rotation.from_euler('xyz', euler_array).as_matrix()
        return R@array_to_rotate
    
    def RotByQuat(array_to_rotate,Q) -> np.ndarray :
        q = Quat_Utils()
        qstar = q.ConjugateQuat(Q)
        x = np.array([array_to_rotate[0], array_to_rotate[1], array_to_rotate[2], 0])
        return q.QuatProduct( q.QuatProduct(Q,x), qstar)[:-1]
    
    def S(x) -> np.ndarray:
        """ Skew-symetric operator """
        sx = np.array([[0,    -x[2],  x[1]],
                       [x[2],   0,   -x[0]],
                       [-x[1], x[0],    0 ]])
        return sx

    def InvQuat(x):
        q = Quat_Utils()
        return q.InvQuat(x)
    def ConjugateQuat(x):
        q = Quat_Utils()
        return q.ConjugateQuat(x)
    def QuatProduct(x, y):
        q = Quat_Utils()
        return q.QuatProduct(x, y)
    def RotateQuat(quat_to_rotate, Quat):
        q = Quat_Utils()
        return q.RotateQuat(quat_to_rotate, Quat)



class Quat_Utils():
    def __init__(self):
        pass

    def InvQuat(self, x):
        # invert a quaternion in scalar-last format
        return self.ConjugateQuat(x) / self.QuatNormSquared(x)

    def ConjugateQuat(self, x):
        # conjugate a quaternion in scalar-last format
        return np.array([-x[0], -x[1], -x[2], x[3]])
    
    def QuatNormSquared(self, x):
        return np.sum(x*x)
    
    def QuatProduct(self, x, y):
        # mulitplication of two quaternion in scalar-last format
        z = np.zeros((4,))
        z[0] = x[3]*y[0] + x[0]*y[3] + x[1]*y[2] - x[2]*y[1]
        z[1] = x[3]*y[1] + x[1]*y[3] + x[2]*y[0] - x[0]*y[2]
        z[2] = x[3]*y[2] + x[2]*y[3] + x[0]*y[1] - x[1]*y[0]
        z[3] = x[3]*y[3] - x[0]*y[0] - x[1]*y[1] - x[2]*y[2]

        return z
    
    def RotateQuat(self, quat_to_rotate, quat):
        # rotate a quaternion by a quaternion
        quat_inv = self.InvQuat(quat)
        return self.QuatProduct(quat_inv, self.QuatProduct(quat_to_rotate, quat))





'''
A class to have a common log for all code
Display logs on flight, or only when PrintLog() is called
'''
class Log():
    def __init__(self, name="", print_on_flight=True):
        self.print_on_flight = print_on_flight
        self.fulllog = "\n\n   ---   beginnig log:: " + name + " --- \n\n"
        if self.print_on_flight : print(self.fulllog)

    def LogTheLog(self, log, style="info", to_print=True):
        if style=="info":
            log = "  -> " + log
        elif style=="subinfo":
            log = "    ...  " + log
        elif style=="warn":
            log = "  -!- " + log
        elif style=="danger":
            log = "** ! ** " + log
        elif style=="title":
            log = "\n\n***\n\n " + log + "\n\n***\n\n "
        elif style=="subtitle":
            log = " ····> " + log
        if to_print and self.print_on_flight: print(log)
        self.fulllog  +=  log + "\n"
    def GetLog(self):
        return self.fulllog
    def PrintLog(self):
        print(self.GetLog())

'''
Classes of functions that returns its derivatives
Used to derive trajectory in speed and acceleration
'''
class Sinus():
    def __init__(self, a, w, x=None):
        self.a = a
        self.w = w
        if x is not None :
            self.evaluate(x)
        else:
            return None
    def evaluate(self, x):
        return self.a * np.sin(self.w * x)
    def deriv(self):
        return Cosinus(self.a*self.w, self.w)
    
        
class Cosinus():
    def __init__(self, a, w):
        self.a = a
        self.w = w
    def evaluate(self, x):
        return self.a * np.cos(self.w * x)
    def deriv(self):
        return Sinus(-self.a*self.w, self.w)
    

class Exp():
    def __init__(self, C, w):
        self.C = C
        self.w = w
    def evaluate(self, x):
        return self.C * np.exp(-self.w * x)
    def deriv(self):
        return Exp(-self.C*self.w, self.w)
    
