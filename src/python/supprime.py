import numpy as np
from Bolt_Utils import utils
from Bolt_Utils import Log
import matplotlib.pyplot as plt

from Bolt_Filter_Complementary import ComplementaryFilter


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







