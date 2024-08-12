

from bolt_estimator.utils.utils import utils, Log
from bolt_estimator.utils.trajectory_generator import TrajectoryGenerator, Metal
from bolt_estimator.estimator.filter_complementary import ComplementaryFilter
import numpy as np
from bolt_estimator.utils.graphics import Graphics




def main(N=1000, noise_level=60):

    # generate useful objects
    test_logger = Log("test", print_on_flight=True)
    grapher = Graphics(logger=test_logger)

    # start generator
    generator = TrajectoryGenerator(logger=test_logger)
    generator.Generate("polynomial5", noise_level=noise_level, N=N)

    # start filter
    ComplementaryFilterT = ComplementaryFilter(parameters=(1/N, 2), 
                                        ndim=1, 
                                        talkative=True, 
                                        name="Standard complementary",
                                        logger=test_logger)
    filter_traj = []
    FilterSpeed = []
    FilterAcc = []
    
    # start filter with added offset compensation (pseudo-integrator)
    ComplementaryFilterO = ComplementaryFilter(parameters=(1/N, 2), 
                                        ndim=1, 
                                        talkative=True,
                                        name="offset-compensed complementary",
                                        logger=test_logger,
                                        memory_size=100,
                                        offset_gain=0.3)

    filter_traj_offset = []
    filter_speed_offset = []
    filter_acc_offset = []

    # start filter with different offset gains
    ComplementaryFilterO_1 = ComplementaryFilter(parameters=(1/N, 2), 
                                        ndim=1, 
                                        talkative=True,
                                        name="offset-compensed complementary 1",
                                        logger=test_logger,
                                        memory_size=20,
                                        offset_gain=0.3)
    filter_traj_offset1 = []
    ComplementaryFilterO_5 = ComplementaryFilter(parameters=(1/N, 2), 
                                        ndim=1, 
                                        talkative=True,
                                        name="offset-compensed complementary 5",
                                        logger=test_logger,
                                        memory_size=300,
                                        offset_gain=0.3)
    filter_traj_offset5 = []


    # get generated data
    true_traj, true_speed, true_acceleration = generator.GetTrueTraj()
    noisy_traj, noisy_speed, noisy_acceleration = generator.GetNoisyTraj()
    #print(noisy_traj)


    for k in range(N):
        filter_traj.append(ComplementaryFilterT.RunFilter(np.array(noisy_traj[0,k]), np.array(noisy_speed[0,k]) ))
        filter_traj_offset.append(ComplementaryFilterO.RunFilterOffset(np.array(noisy_traj[0,k]), np.array(noisy_speed[0,k]) ))
        filter_traj_offset1.append(ComplementaryFilterO_1.RunFilterOffset(np.array(noisy_traj[0,k]), np.array(noisy_speed[0,k]) ))
        filter_traj_offset5.append(ComplementaryFilterO_5.RunFilterOffset(np.array(noisy_traj[0,k]), np.array(noisy_speed[0,k]) ))

    # turn list to array (sorry for ugliness)
    filter_traj = np.array(filter_traj).reshape(1, N)
    #FilterSpeed = np.array(FilterSpeed).reshape(1, N)
    #FilterAcc = np.array(FilterAcc).reshape(1, N)

    filter_traj_offset = np.array(filter_traj_offset).reshape(1, N)
    filter_traj_offset1 = np.array(filter_traj_offset1).reshape(1, N)
    filter_traj_offset5 = np.array(filter_traj_offset5).reshape(1, N)
    #FilterSpeed_offset = np.array(FilterSpeed_offset).reshape(1, N)
    #FilterAcc_offset = np.array(FilterAcc_offset).reshape(1, N)

    dataset = [noisy_traj, true_traj, filter_traj, filter_traj_offset1, filter_traj_offset, filter_traj_offset5]
    grapher.SetLegend(["Noisy position (" + str(noise_level) + "%)", "True pos", "Filter out pos", "Filter w/OC out pos (memory=20)", "Filter w/OC out pos (memory=100)", "Filter w/OC out pos (memory=300)"], 1)
    grapher.CompareNDdatas(dataset, "position", "Test CF, position, polynomial (OCgain=0.3)", style_adapter=False, width=1.3)
    return dataset




dataset = main()
