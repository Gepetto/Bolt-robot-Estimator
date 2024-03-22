import numpy as np
from Bolt_Utils import utils
from Bolt_Utils import Log

from Bolt_Filter_Complementary import ComplementaryFilter


logs = Log()
logs.LogTheLog(" Starting log of" + "bob", ToPrint=False)
logs.LogTheLog("Initializing " + "non" + "...", style="title")
logs.LogTheLog("No legs are touching the ground", style="warn")
logs.LogTheLog("No legs are touching the ground", style="warn")
logs.LogTheLog("No legs are touching the ground", style="warn")
logs.LogTheLog("Initializing " + "oui" + "...", style="info")

print(logs.GetLog())
# create an array
array1 = np.array([[0, 1], 
                    [2, 3], 
                    [4, 5], 
                    [6, 7]])

print(utils.normalize(array1[1]))

# find the average across axis 1
average1 = np.average(array1, 1)

# find the average across axis 0
average2 = np.average(array1, 0)

#print('\naverage across axis 1:\n', average1)
print('\naverage across axis 0:\n', average2)

a = np.array([0, 1, 2])
b = np.array([3, 1, 2])
c = np.array([4, 1, 2])
d = np.array([4, 1, 2])

d[:-1] = d[1:]
#print( utils.MatrixFromVectors((a, b, c, d)) )



filter = ComplementaryFilter((0.0005, 3), talkative=True)

















