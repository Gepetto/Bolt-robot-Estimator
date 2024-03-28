import sys
sys.path.append('/home/nalbrecht/Bolt-Estimator/Bolt-robot---Estimator/src/python')
from Bolt_Utils import utils
from Bolt_Utils import Log
from TrajectoryGenerator import TrajectoryGenerator, Graphics, Metal
from Bolt_Filter_Complementary import ComplementaryFilter
import numpy as np




def main(N=100, NoiseLevel=20):

    # generate useful objects
    testlogger = Log("test", PrintOnFlight=True)
    grapher = Graphics(logger=testlogger)

    # start generator
    generator = TrajectoryGenerator(logger=testlogger)
    generator.Generate("polynomial2", NoiseLevel=NoiseLevel, N=N)

    # start filter
    ComplementaryFilterT = ComplementaryFilter(parameters=(1/N, 2), 
                                        ndim=1, 
                                        talkative=True, 
                                        name="Standard complementary",
                                        logger=testlogger)
    FilterTraj = []
    FilterSpeed = []
    FilterAcc = []
    
    # start filter with added offset compensation (pseudo-integrator)
    ComplementaryFilterO = ComplementaryFilter(parameters=(1/N, 2), 
                                        ndim=1, 
                                        talkative=True,
                                        name="Offset-compensed complementary",
                                        logger=testlogger,
                                        MemorySize=100,
                                        OffsetGain=0.3)

    FilterTrajOffset = []
    FilterSpeedOffset = []
    FilterAccOffset = []



    # get generated data
    TrueTraj, TrueSpeed, TrueAcc = generator.GetTrueTraj()
    NoisyTraj, NoisySpeed, NoisyAcc = generator.GetNoisyTraj()
    #print(NoisyTraj)

    # run filter over time, with noisy data as inputs
    for k in range(N):
        FilterSpeed.append(ComplementaryFilterT.RunFilter(np.array(NoisySpeed[0,k]), np.array(NoisyAcc[0,k]) ))
        FilterSpeedOffset.append(ComplementaryFilterO.RunFilterOffset(np.array(NoisySpeed[0,k]), np.array(NoisyAcc[0,k]) ))


    # turn list to array (sorry for ugliness)
    #FilterTraj = np.array(FilterTraj).reshape(1, N)
    FilterSpeed = np.array(FilterSpeed).reshape(1, N)
    #FilterAcc = np.array(FilterAcc).reshape(1, N)

    #FilterTrajOffset = np.array(FilterTrajOffset).reshape(1, N)
    FilterSpeedOffset = np.array(FilterSpeedOffset).reshape(1, N)
    #FilterAccOffset = np.array(FilterAccOffset).reshape(1, N)

    dataset = [NoisySpeed, TrueSpeed, FilterSpeed, FilterSpeedOffset]
    grapher.SetLegend(["Noisy speed (" + str(NoiseLevel) + "%)", "True speed", "Filter out speed", "Filter w/ offset comp. out speed"], 1)
    grapher.CompareNDdatas(dataset, "speed", "Test CF, speed, sinus (memory=100, offsetgain=0.3)", StyleAdapter=False, AutoLeg=False, width=1.5)
    return dataset




dataset = main()