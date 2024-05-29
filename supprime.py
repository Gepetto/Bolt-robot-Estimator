import numpy as np
import matplotlib.pyplot as plt



X = np.linspace(0, 10, 100)
Y = np.sin(X) + np.cos(X)
Z = np.sin(X+np.pi/4)*2/1.4

plt.plot(X, Y)
plt.plot(X, Z)
plt.grid()
plt.show()
