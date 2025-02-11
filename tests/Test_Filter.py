

import numpy as np

from bolt_estimator.utils.Bolt_Utils import utils, Log
from bolt_estimator.utils.TrajectoryGenerator import TrajectoryGenerator, Metal
from bolt_estimator.utils.Graphics import Graphics
from bolt_estimator.estimator.Bolt_Filter_Complementary import ComplementaryFilter

from bolt_estimator.data.DataReader import DataReader


'''
A code to test a filter.

Outputs all data as graphs

'''



class Test_Filter():
    def __init__(self, FilterType="complementary",
                        parameters=(1/1000, 2),
                        optparameters = (100, 0.005),
                        name="Standard complementary"):
            self.FilterType=FilterType
            self.parameters=parameters
            self.optparameters = optparameters
            self.name = name

    def TestDim(self, WishedDim, InputedDim):
        testlogger = Log("testing dimension input ", PrintOnFlight=True)
        print("# ", self.optparameters)
        memsize, integratorgain = self.optparameters
        Filter = ComplementaryFilter(self.parameters, ndim=WishedDim, talkative=True, name=self.name, logger=testlogger, MemorySize=memsize, OffsetGain=integratorgain)
        x = np.ones(InputedDim)
        xdot = np.ones(InputedDim)
        Filter.RunFilter(x, xdot)



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
            print(" # shape ", AdaptedSpeed.shape)

            # number of samples
            N=max(pcom.shape)
            ndim=3

            # start generator
            generator = TrajectoryGenerator(logger=testlogger)
            generator.Generate(datatype, NoiseLevel=NoiseLevel, N=N, traj=AdaptedTraj)


        elif datatype=="simulated":
            datatype="custom"
            reader = DataReader(testlogger)
            reader.AutoLoad(4)
            a = reader.Get("a")[:, 1, :].copy()
            v = reader.Get("v")[:, 1, :].copy()
            
            AdaptedTraj = np.array([v[:, 0], v[:, 1], v[:, 2]])
            AdaptedSpeed = np.array([a[:, 0], a[:, 1], a[:, 2]])
            print(" # shape ", AdaptedSpeed.shape)

            # number of samples
            N=max(v.shape)
            ndim=3

            # start generator
            generator = TrajectoryGenerator(logger=testlogger)
            generator.Generate(datatype, NoiseLevel=NoiseLevel, N=N, traj=AdaptedTraj)
        
        elif datatype=="simulated 1D":
            datatype="custom"
            reader = DataReader(testlogger)
            reader.AutoLoad(4)
            a = reader.Get("a")[:, 1, 2].copy()
            v = reader.Get("v")[:, 1, 2].copy()
            
            AdaptedTraj = np.array([v])
            AdaptedSpeed = np.array([a])
            print(" # shape ", AdaptedSpeed.shape)

            # number of samples
            N=max(v.shape)
            ndim=1

            # start generator
            generator = TrajectoryGenerator(logger=testlogger)
            generator.Generate(datatype, NoiseLevel=NoiseLevel, N=N, traj=AdaptedTraj)


        else : 
            # start generator
            generator = TrajectoryGenerator(logger=testlogger)
            generator.Generate(datatype, NoiseLevel=NoiseLevel, N=N, amplitude=0.1)
            ndim=1
        
        if self.FilterType == "complementary":
            print(" # optparams ", self.optparameters)
            print(" # ndim ", ndim)
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
        grapher.CompareNDdatas(dataset, "position", "Output on " + datatype + " traj. with noise level " + str(NoiseLevel) + "\n to filter " + self.filter.name, StyleAdapter=False)

        # plotting error
        scaler = abs(np.max(TrueTraj) / np.min(TrueTraj))
        scaled_error = abs(TrueTraj-FilterTraj)/scaler
        dataset = [scaled_error]
        print(" error coeff : ", np.sum(scaled_error))
        grapher.SetLegend(["error of the filter " + self.filter.name], ndim)
        grapher.CompareNDdatas(dataset, "position", "Error on " + datatype + " traj. with noise level " + str(NoiseLevel) + "\n to filter " + self.filter.name, StyleAdapter=False, width=0.5)
        grapher.end()




# the number of samples on which to test the filter
N = 1000
# the desired level of noise in the signal to be filtered
NoiseLevel=40
# the filter to test
filtertype =  "complementary"
parameters=(1/N, 0.04)
optparameters = (50, 0.02)
name="Standard complementary"




TF = Test_Filter(filtertype, parameters, optparameters, name=name)
# TF.TestDim(3, 3)
# TF.RunTest(N, NoiseLevel=10, datatype="polynomial")
# TF.RunTest(N, NoiseLevel=40, datatype="sinus")
# TF.RunTest(N, NoiseLevel=30, datatype="polynomial9")
# TF.RunTest(N, NoiseLevel=20, datatype="custom")
# TF.RunTest(N, NoiseLevel=20, datatype="simulated")
TF.RunTest(N, NoiseLevel=5, datatype="simulated 1D")












