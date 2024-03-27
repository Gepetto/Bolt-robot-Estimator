import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('/home/nalbrecht/Bolt-Estimator/Bolt-robot---Estimator/src/python')
from Bolt_Utils import utils
from Bolt_Utils import Log
from TrajectoryGenerator import TrajectoryGenerator, Graphics, Metal
from Bolt_Filter_Complementary import ComplementaryFilter







def main(N=1000, NoiseLevel=20):

    # generate useful objects
    testlogger = Log("test", PrintOnFlight=True)
    grapher = Graphics(logger=testlogger)

    # get 


    # start generator
    generator = TrajectoryGenerator(logger=testlogger)
    generator.Generate("multiexp", NoiseLevel=NoiseLevel, N=N)

    TrueTraj, TrueSpeed, TrueAcc = generator.GetTrueTraj()
    pcom = np.load("/home/nalbrecht/Bolt-Estimator/Bolt-robot---Estimator/tests/com_pos.npy")
    vcom = np.load("/home/nalbrecht/Bolt-Estimator/Bolt-robot---Estimator/tests/vcom_pos.npy")

    #dataset = [TrueTraj, TrueSpeed, TrueAcc]
    #grapher.SetLegend(["traj", "speed", "acc"])
    #grapher.CompareNDdatas(dataset, "speed", "Test generator", StyleAdapter=False, AutoLeg=False, width=1.5)

    dataset = [pcom, vcom]
    grapher.SetLegend(["traj", "speed"], 3)
    grapher.CompareNDdatas(dataset, "speed", "Test generator", StyleAdapter=False, AutoLeg=False, width=1.5)
    print(pcom)




main()
