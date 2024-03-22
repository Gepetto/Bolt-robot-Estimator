import numpy as np


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
    
    def MatrixFromVectors(L, n=3):
        # L tuple of numpy vectors
        A = np.stack(L).reshape((-1, n))
        return A
    
    def scalar(a, b) -> float:
        # returns a scalar b
        return np.sum(a*b)


class Log():
    def __init__(self):
        self.fulllog = ""

    def LogTheLog(self, log, style="info", ToPrint=True):
        if style=="info":
            log = "  -> " + log
        elif style=="warn":
            log = "  *!* " + log
        elif style=="title":
            log = "\n\n***\n\n " + log + "\n\n***\n\n "
        if ToPrint : print(log)
        self.fulllog  +=  log + "\n"
    def GetLog(self):
        return self.fulllog
