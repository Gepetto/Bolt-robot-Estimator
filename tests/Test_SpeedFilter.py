import sys
sys.path.append('/home/nalbrecht/Bolt-Estimator/Bolt-robot---Estimator/src/python')
from Bolt_Utils import utils
from Bolt_Utils import Log
from TrajectoryGenerator import TrajectoryGenerator, Graphics, Metal
from Bolt_Filter_Complementary import ComplementaryFilter
import numpy as np




def main(N=1000, NoiseLevel=20):

    # generate useful objects
    testlogger = Log("test", PrintOnFlight=True)
    grapher = Graphics(logger=testlogger)

    # start generator
    generator = TrajectoryGenerator(logger=testlogger)
    generator.Generate("sinus", NoiseLevel=NoiseLevel, N=N)

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

    # start filter with different offset gains
    ComplementaryFilterO_1 = ComplementaryFilter(parameters=(1/N, 2), 
                                        ndim=1, 
                                        talkative=True,
                                        name="Offset-compensed complementary",
                                        logger=testlogger,
                                        MemorySize=100,
                                        OffsetGain=0.1)
    FilterTrajOffset1 = []
    ComplementaryFilterO_5 = ComplementaryFilter(parameters=(1/N, 2), 
                                        ndim=1, 
                                        talkative=True,
                                        name="Offset-compensed complementary",
                                        logger=testlogger,
                                        MemorySize=100,
                                        OffsetGain=0.5)
    FilterTrajOffset5 = []


    # get generated data
    TrueTraj, TrueSpeed, TrueAcc = generator.GetTrueTraj()
    NoisyTraj, NoisySpeed, NoisyAcc = generator.GetNoisyTraj()
    #print(NoisyTraj)


    for k in range(N):
        FilterTraj.append(ComplementaryFilterT.RunFilter(np.array(NoisyTraj[0,k]), np.array(NoisySpeed[0,k]) ))
        FilterTrajOffset.append(ComplementaryFilterO.RunFilterOffset(np.array(NoisyTraj[0,k]), np.array(NoisySpeed[0,k]) ))
        FilterTrajOffset1.append(ComplementaryFilterO_1.RunFilterOffset(np.array(NoisyTraj[0,k]), np.array(NoisySpeed[0,k]) ))
        FilterTrajOffset5.append(ComplementaryFilterO_5.RunFilterOffset(np.array(NoisyTraj[0,k]), np.array(NoisySpeed[0,k]) ))

    # turn list to array (sorry for ugliness)
    FilterTraj = np.array(FilterTraj).reshape(1, N)
    #FilterSpeed = np.array(FilterSpeed).reshape(1, N)
    #FilterAcc = np.array(FilterAcc).reshape(1, N)

    FilterTrajOffset = np.array(FilterTrajOffset).reshape(1, N)
    FilterTrajOffset1 = np.array(FilterTrajOffset1).reshape(1, N)
    FilterTrajOffset5 = np.array(FilterTrajOffset5).reshape(1, N)
    #FilterSpeedOffset = np.array(FilterSpeedOffset).reshape(1, N)
    #FilterAccOffset = np.array(FilterAccOffset).reshape(1, N)

    dataset0 = [NoisyTraj, TrueTraj, FilterTraj, FilterTrajOffset]
    dataset = [TrueTraj, FilterTraj, FilterTrajOffset1, FilterTrajOffset, FilterTrajOffset5]
    grapher.CompareNDdatas(dataset, "position", "essai", StyleAdapter=False)
    grapher.CompareNDdatas(dataset0, "position", "essai0", StyleAdapter=False)
    return dataset




dataset = main()