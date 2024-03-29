import numpy as np
from Bolt_Utils import utils
from Bolt_Utils import Log
import matplotlib.pyplot as plt

from Bolt_Filter_Complementary import ComplementaryFilter


# quick gain model
def f(x, a, b):
    M = 100
    return (M/b - x)*a


X = np.linspace(0, 100, 20)
Y = f(X, 0.6, 10)
Z = f(X, 0.15, 3)


plt.clf()
plt.plot(X, Y)
plt.plot(X, Z)
plt.grid()
plt.show()







