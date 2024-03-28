import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('/home/nalbrecht/Bolt-Estimator/Bolt-robot---Estimator/src/python')
from Bolt_Utils import utils
from Bolt_Utils import Log
from TrajectoryGenerator import TrajectoryGenerator, Graphics, Metal
from Bolt_Filter_Complementary import ComplementaryFilter







def main(N=100, NoiseLevel=50):

    # generate useful objects
    testlogger = Log("test", PrintOnFlight=True)
    grapher = Graphics(logger=testlogger)

    # get data and extract data
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

    '''
    # display data as they were read
    dataset = [pcom, vcom]
    grapher.SetLegend(["traj", "speed"], 3)
    grapher.CompareNDdatas(dataset, "pos&speed", "Constant's COM data", StyleAdapter=False, AutoLeg=False, width=1.5)

    # display extracted data
    dataset = [AdaptedTraj, AdaptedSpeed]
    grapher.SetLegend(["traj", "speed"], 3)
    grapher.CompareNDdatas(dataset, "pos&speed", "Adapted data from Constant's", StyleAdapter=False, AutoLeg=False, width=1.5)
    '''
    # N-DIMENSIONS TO BE CHANGED ACCORDINGLY
    ndim = 3
    # start generator
    generator = TrajectoryGenerator(logger=testlogger)
    generator.Generate("custom", NoiseLevel=NoiseLevel, N=max(pcom.shape), traj=AdaptedTraj)

    # start filter
    ComplementaryFilter1 = ComplementaryFilter(parameters=(1/N, 2), 
                                        ndim=ndim, 
                                        talkative=True, 
                                        name="Standard offset complementary",
                                        logger=testlogger,
                                        MemorySize=300,
                                        OffsetGain=0.004)
    FilterTraj1 = np.zeros((ndim, N))


    # start filter
    ComplementaryFilter2 = ComplementaryFilter(parameters=(1/N, 2), 
                                        ndim=ndim, 
                                        talkative=True, 
                                        name="Modified offset complementary",
                                        logger=testlogger,
                                        MemorySize=300,
                                        OffsetGain=0.004)
    FilterTraj2 = np.zeros((ndim, N))


    # start filter
    ComplementaryFilter3 = ComplementaryFilter(parameters=(1/N, 2), 
                                        ndim=ndim, 
                                        talkative=True, 
                                        name="quick-convergence offset complementary",
                                        logger=testlogger,
                                        MemorySize=300,
                                        OffsetGain=0.004)
    FilterTraj3 = np.zeros((ndim, N))



   


    # get data
    TrueTraj, TrueSpeed, TrueAcc = generator.GetTrueTraj()
    NoisyTraj, NoisySpeed, NoisyAcc = generator.GetNoisyTraj()

    # run filter over time, with noisy data as inputs
    for k in range(N):
        FilterTraj1[:, k] = ComplementaryFilter1.RunFilterOffset(np.array(NoisyTraj[:,k]), np.array(NoisySpeed[:,k]) )
        FilterTraj2[:, k] = ComplementaryFilter2.RunFilterOffset2(np.array(NoisyTraj[:,k]), np.array(NoisySpeed[:,k]) )
        FilterTraj3[:, k] = ComplementaryFilter3.RunFilterOffset3(np.array(NoisyTraj[:,k]), np.array(NoisySpeed[:,k]) )




    
    """
    dataset = [TrueTraj, TrueSpeed]
    grapher.SetLegend(["traj", "speed"], 3)
    grapher.CompareNDdatas(dataset, "pos&speed", "Data generated by Generator from Constant's", StyleAdapter=True, AutoLeg=False, width=1.5)
    """

    dataset = [NoisyTraj, TrueTraj, FilterTraj1]
    grapher.SetLegend(["noisy traj", "traj", "filtered traj"], ndim)
    grapher.CompareNDdatas(dataset, "position", "Noisy, True and Filtered trajectories\nStandard offset compensator", StyleAdapter=False, AutoLeg=False, width=1.5)
    


    dataset = [NoisyTraj, TrueTraj, FilterTraj2]
    grapher.SetLegend(["noisy traj", "traj", "filtered traj"], ndim)
    grapher.CompareNDdatas(dataset, "position", "Noisy, True and Filtered trajectories\nNew offset compensator", StyleAdapter=False, AutoLeg=False, width=1.5)

    dataset = [NoisyTraj, TrueTraj, FilterTraj3]
    grapher.SetLegend(["noisy traj", "traj", "filtered traj"], ndim)
    grapher.CompareNDdatas(dataset, "position", "Noisy, True and Filtered trajectories\nNew offset compensator + quick convergence gain", StyleAdapter=False, AutoLeg=False, width=1.5)


    dataset = [TrueTraj, FilterTraj1, FilterTraj2, FilterTraj3]
    grapher.SetLegend(["true traj", "filtered traj with offset", "filtered traj with nonzero mean",  "filtered traj with quick convergence gain"], ndim)
    grapher.CompareNDdatas(dataset, "position", "Noisy, True and Filtered trajectories\nNew offset compensator + quick convergence gain", StyleAdapter=False, AutoLeg=False, width=1.5)

    scaler = abs(np.max(TrueTraj) / np.min(TrueTraj))
    print(scaler)
    dataset = [abs(TrueTraj-FilterTraj1)/scaler , abs(TrueTraj-FilterTraj2)/scaler, abs(TrueTraj-FilterTraj3)/scaler]
    grapher.SetLegend(["error with offset", "errror with nonzero mean",  "error with quick convergence gain"], ndim)
    grapher.CompareNDdatas(dataset, "position", "Error for different offset compensator", StyleAdapter=False, AutoLeg=False, width=1.5)






    return None
    




main()
