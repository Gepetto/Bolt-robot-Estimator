import sys
sys.path.append('/home/nalbrecht/Bolt-Estimator/Bolt-robot---Estimator/src/python')
from Bolt_Utils import utils
from Bolt_Utils import Log
from TrajectoryGenerator import TrajectoryGenerator, Metal
from Graphics import Graphics
from Bolt_Filter_Complementary import ComplementaryFilter
import numpy as np




def main(N=500, NoiseLevel=20, Drift=40):

    # generate useful objects
    testlogger = Log("test", PrintOnFlight=True)
    grapher = Graphics(logger=testlogger)

    # start generator
    generator = TrajectoryGenerator(logger=testlogger)
    generator.Generate("polynomial", NoiseLevel=NoiseLevel, Drift=Drift, N=N, amplitude=1, avgfreq=1/6, T=1)

    # start filter
    ComplementaryFilterT = ComplementaryFilter(parameters=(1/N, 2), 
                                        ndim=1, 
                                        talkative=True, 
                                        name="Standard complementary",
                                        logger=testlogger,
                                        MemorySize=150,
                                        OffsetGain=0.02)
    FilterTraj = []
    FilterSpeed = []
    FilterAcc = []
    FilterDriftSpeed = []




    # get generated data
    TrueTraj, TrueSpeed, TrueAcc = generator.GetTrueTraj()
    NoisyTraj, NoisySpeed, NoisyAcc = generator.GetNoisyTraj()
    DriftNoisyTraj, DriftNoisySpeed, DriftNoisyAcc = generator.GetDriftingNoisyTraj()
    DriftTraj, DriftSpeed, DriftAcc = generator.GetDriftingTraj()


    # run filter over time, with noisy data as inputs
    for k in range(N):
        FilterSpeed.append(ComplementaryFilterT.RunFilter(np.array(NoisySpeed[0,k]), np.array(NoisyAcc[0,k]) ))
        FilterDriftSpeed.append(ComplementaryFilterT.RunFilter(np.array(NoisySpeed[0,k]), np.array(DriftNoisyAcc[0,k]) ))


    #FilterTraj = np.array(FilterTraj).reshape(1, N)
    FilterSpeed = np.array(FilterSpeed).reshape(1, N)
    #FilterAcc = np.array(FilterAcc).reshape(1, N)

    #FilterTrajOffset = np.array(FilterTrajOffset).reshape(1, N)
    FilterDriftSpeed = np.array(FilterDriftSpeed).reshape(1, N)
    #FilterAccOffset = np.array(FilterAccOffset).reshape(1, N)

    dataset = [NoisySpeed, DriftNoisySpeed, TrueSpeed, DriftSpeed]
    grapher.SetLegend(["Noisy speed (" + str(NoiseLevel) + "%)", "Drifting noisy speed", "True speed", "Drifting Speed"], 1)
    grapher.CompareNDdatas(dataset, "speed", "Test CF, speed, sinusoidal", StyleAdapter=False, AutoLeg=False, width=0.8)

    dataset = [TrueSpeed, TrueAcc, DriftAcc, FilterDriftSpeed]
    grapher.SetLegend(["Theta ", "Omega", "Drifting Omega",  "Filter acting on\nnoisy Theta & Drifting noisy Omega"], 1)
    grapher.CompareNDdatas(dataset, "speed", "Test CF, theta, sinusoidal", StyleAdapter=True, AutoLeg=False, width=1.5)

    return dataset




dataset = main()