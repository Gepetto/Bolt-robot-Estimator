import matplotlib.pyplot as plt
import numpy as np



from bolt_estimator.utils.Utils import utils, Log
from bolt_estimator.utils.TrajectoryGenerator import TrajectoryGenerator, Metal
from bolt_estimator.estimator.Filter_Complementary import ComplementaryFilter
from bolt_estimator.utils.Graphics import Graphics







def main(N=1000, NoiseLevel=50):

    # generate useful objects
    test_logger = Log("test", PrintOnFlight=True)
    grapher = Graphics(logger=test_logger)

    # get data and extract data
    pcom = np.load("./miscanellous/com_pos_superplus.npy")
    vcom = np.load("./miscanellous/vcom_pos_superplus.npy")
    trajX = pcom[0, :, 0]
    trajY = pcom[0, :, 1]
    trajZ = pcom[0, :, 2]
    adapted_traj = np.array([trajX, trajY, trajZ])
    speedX = vcom[0, :, 0]
    speedY = vcom[0, :, 1]
    speedZ = vcom[0, :, 2]
    adapted_speed = np.array([speedX, speedY, speedZ])

    # number of samples
    N=max(pcom.shape)

    '''
    # display data as they were read
    dataset = [pcom, vcom]
    grapher.SetLegend(["traj", "speed"], 3)
    grapher.CompareNDdatas(dataset, "pos&speed", "Constant's COM data", StyleAdapter=False, width=1.5)

    # display extracted data
    dataset = [adapted_traj, adapted_speed]
    grapher.SetLegend(["traj", "speed"], 3)
    grapher.CompareNDdatas(dataset, "pos&speed", "Adapted data from Constant's", StyleAdapter=False, width=1.5)
    '''
    # N-DIMENSIONS TO BE CHANGED ACCORDINGLY
    ndim = 3
    # start generator
    generator = TrajectoryGenerator(logger=test_logger)
    generator.Generate("custom", NoiseLevel=NoiseLevel, N=N, traj=adapted_traj)

    # start filter
    ComplementaryFilter1 = ComplementaryFilter(parameters=(1/N, 2), 
                                        ndim=ndim, 
                                        talkative=True, 
                                        name="Standard offset complementary",
                                        logger=test_logger,
                                        MemorySize=300,
                                        OffsetGain=0.004)
    filtered_traj_1 = np.zeros((ndim, N))


    # start filter
    ComplementaryFilter2 = ComplementaryFilter(parameters=(1/N, 2), 
                                        ndim=ndim, 
                                        talkative=True, 
                                        name="Modified offset complementary",
                                        logger=test_logger,
                                        MemorySize=300,
                                        OffsetGain=0.004)
    filtered_traj_2 = np.zeros((ndim, N))


    # start filter
    ComplementaryFilter3 = ComplementaryFilter(parameters=(1/N, 2), 
                                        ndim=ndim, 
                                        talkative=True, 
                                        name="quick-convergence offset complementary",
                                        logger=test_logger,
                                        MemorySize=300,
                                        OffsetGain=0.004)
    filtered_traj_3 = np.zeros((ndim, N))



   


    # get data
    true_traj, true_speed, true_acceleration = generator.GetTrueTraj()
    noisy_traj, noisy_speed, noisy_acceleration = generator.GetNoisyTraj()

    # run filter over time, with noisy data as inputs
    for k in range(N):
        filtered_traj_1[:, k] = ComplementaryFilter1.RunFilterOffset(np.array(noisy_traj[:,k]), np.array(noisy_speed[:,k]) )
        filtered_traj_2[:, k] = ComplementaryFilter2.RunFilterOffset(np.array(noisy_traj[:,k]), np.array(noisy_speed[:,k]) )
        filtered_traj_3[:, k] = ComplementaryFilter3.RunFilterOffset(np.array(noisy_traj[:,k]), np.array(noisy_speed[:,k]) )




    
    """
    dataset = [true_traj, true_speed]
    grapher.SetLegend(["traj", "speed"], 3)
    grapher.CompareNDdatas(dataset, "pos&speed", "Data generated by Generator from Constant's", StyleAdapter=True, width=1.5)
    """

    dataset = [noisy_traj, true_traj, filtered_traj_1]
    grapher.SetLegend(["noisy traj", "traj", "filtered traj"], ndim)
    grapher.CompareNDdatas(dataset, "position", "Noisy, True and Filtered trajectories\nStandard offset compensator", StyleAdapter=False, width=1.5)
    


    dataset = [noisy_traj, true_traj, filtered_traj_2]
    grapher.SetLegend(["noisy traj", "traj", "filtered traj"], ndim)
    grapher.CompareNDdatas(dataset, "position", "Noisy, True and Filtered trajectories\nNew offset compensator", StyleAdapter=False, width=1.5)

    dataset = [noisy_traj, true_traj, filtered_traj_3]
    grapher.SetLegend(["noisy traj", "traj", "filtered traj"], ndim)
    grapher.CompareNDdatas(dataset, "position", "Noisy, True and Filtered trajectories\nNew offset compensator + quick convergence gain", StyleAdapter=False, width=1.5)


    dataset = [true_traj, filtered_traj_1, filtered_traj_2, filtered_traj_3]
    grapher.SetLegend(["true traj", "filtered traj with offset", "filtered traj with nonzero mean",  "filtered traj with quick convergence gain"], ndim)
    grapher.CompareNDdatas(dataset, "position", "Noisy, True and Filtered trajectories\nNew offset compensator + quick convergence gain", StyleAdapter=False, width=1.5)

    scaler = abs(np.max(true_traj) / np.min(true_traj))
    print(scaler)
    dataset = [abs(true_traj-filtered_traj_1)/scaler , abs(true_traj-filtered_traj_2)/scaler, abs(true_traj-filtered_traj_3)/scaler]
    grapher.SetLegend(["error with offset", "errror with nonzero mean",  "error with quick convergence gain"], ndim)
    grapher.CompareNDdatas(dataset, "position", "Error for different offset compensator", StyleAdapter=False, width=1.5)
    grapher.end()






    return None
    




main()
