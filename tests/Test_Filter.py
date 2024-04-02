

import numpy as np
import sys
sys.path.append('/home/nalbrecht/Bolt-Estimator/Bolt-robot---Estimator/src/python')
from Bolt_Utils import utils, Log
from TrajectoryGenerator import TrajectoryGenerator, Metal
from Graphics import Graphics
from Bolt_Filter_Complementary import ComplementaryFilter


'''
A code to test a filter.

Outputs all data as graphs

'''

# the number of samples on which to test the filter
N = 1000
# the desired level of noise in the signal to be filtered
NoiseLevel=40
# the filter to test
filtertype =  "complementary"
parameters=(1/N, 2)
optparameters = (100, 0.005)
name="Standard complementary"

class Test_Filter():
    def __init__(self, FilterType="complementary",
                        parameters=(1/N, 2),
                        optparameters = (100, 0.005),
                        name="Standard complementary"):
            self.FilterType=FilterType
            self.parameters=parameters
            self.optparameters = optparameters
            self.name = name


    def RunTest(self, N, NoiseLevel, datatype):
        # generate useful objects
        testlogger = Log("test " + datatype + " with noise level " + str(NoiseLevel), PrintOnFlight=True)
        grapher = Graphics(logger=testlogger)

        # load custom data
        if datatype=="custom":
            pcom = np.load("/home/nalbrecht/Bolt-Estimator/Bolt-robot---Estimator/tests/com_pos_superplus.npy")
            vcom = np.load("/home/nalbrecht/Bolt-Estimator/Bolt-robot---Estimator/tests/vcom_pos_superplus.npy")
            trajX = pcom[0, :, 0]
            trajY = pcom[0, :, 1]
            trajZ = pcom[0, :, 2]
            AdaptedTraj = np.array([trajX, trajY, trajZ])
            speedX = vcom[0, :, 0]
            speedY = vcom[0, :, 1]
            speedZ = vcom[0, :, 2]
            AdaptedSpeed = np.array([speedX, speedY, speedZ])

            # number of samples
            N=max(pcom.shape)
            ndim=3

            # start generator
            generator = TrajectoryGenerator(logger=testlogger)
            generator.Generate(datatype, NoiseLevel=NoiseLevel, N=N, traj=AdaptedTraj)

        else : 
            # start generator
            generator = TrajectoryGenerator(logger=testlogger)
            generator.Generate(datatype, NoiseLevel=NoiseLevel, N=N)
            ndim=1
        
        if self.FilterType == "complementary":
            print("#########wtf", self.optparameters)
            memsize, integratorgain = self.optparameters
            self.filter = ComplementaryFilter(self.parameters, ndim=ndim, talkative=True, name=self.name, logger=testlogger, MemorySize=memsize, OffsetGain=integratorgain)
            
        # empty filter data filter
        FilterTraj = np.zeros((ndim, N))

        # get data
        TrueTraj, TrueSpeed, TrueAcc = generator.GetTrueTraj()
        NoisyTraj, NoisySpeed, NoisyAcc = generator.GetNoisyTraj()

        # run filter over time, with noisy data as inputs
        for k in range(N):
            FilterTraj[:, k] = self.filter.RunFilter(np.array(NoisyTraj[:,k]), np.array(NoisySpeed[:,k]) )

        # plotting
        dataset = [NoisyTraj, TrueTraj, FilterTraj]
        grapher.SetLegend(["Noisy position (" + str(NoiseLevel) + "%)", "True pos", "Filter out pos"], ndim)
        grapher.CompareNDdatas(dataset, "position", "Output on " + datatype + " traj. with noise level " + str(NoiseLevel) + "\n to filter " + self.filter.name, StyleAdapter=False, AutoLeg=False, width=1.3)

        # plotting error
        scaler = abs(np.max(TrueTraj) / np.min(TrueTraj))
        dataset = [abs(TrueTraj-FilterTraj)/scaler]
        grapher.SetLegend(["error of the filter " + self.filter.name], ndim)
        grapher.CompareNDdatas(dataset, "position", "Error on " + datatype + " traj. with noise level " + str(NoiseLevel) + "\n to filter " + self.filter.name, StyleAdapter=False, AutoLeg=False, width=1.5)




TF = Test_Filter(filtertype, parameters, optaparameters, name=name)
TF.RunTest(N, NoiseLevel=10, datatype="polynomial")
TF.RunTest(N, NoiseLevel=40, datatype="sinus")
TF.RunTest(N, NoiseLevel=80, datatype="polynomial9")
TF.RunTest(N, NoiseLevel=40, datatype="custom")












