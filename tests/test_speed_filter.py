

from bolt_estimator.utils.utils import utils, Log

from bolt_estimator.utils.trajectory_generator import TrajectoryGenerator, Metal
from bolt_estimator.utils.graphics import Graphics
from bolt_estimator.estimator.filter_complementary import ComplementaryFilter
import numpy as np



def main(N=500, noise_level=20, drift=40):

    # generate useful objects
    testlogger = Log("test", print_on_flight=True)
    grapher = Graphics(logger=testlogger)

    # start generator
    generator = TrajectoryGenerator(logger=testlogger)
    generator.Generate("polynomial", noise_level=noise_level, drift=drift, N=N, amplitude=1, avgfreq=1/6, T=1)

    # start filter
    ComplementaryFilterT = ComplementaryFilter(parameters=(1/N, 2), 
                                        ndim=1, 
                                        talkative=True, 
                                        name="Standard complementary",
                                        logger=testlogger,
                                        memory_size=150,
                                        offset_gain=0.02)
    filter_traj = []
    filter_speed = []
    filter_acc = []
    filterdrift_speed = []




    # get generated data
    true_traj, true_speed, true_acceleration = generator.GetTrueTraj()
    noisy_traj, noisy_speed, noisy_acceleration = generator.GetNoisyTraj()
    drift_noisy_traj, drift_noisy_speed, drift_noisy_acceleration = generator.GetDriftingNoisyTraj()
    drift_traj, drift_speed, drift_acc = generator.GetDriftingTraj()


    # run filter over time, with noisy data as inputs
    for k in range(N):
        filter_speed.append(ComplementaryFilterT.RunFilter(np.array(noisy_speed[0,k]), np.array(noisy_acceleration[0,k]) ))
        filterdrift_speed.append(ComplementaryFilterT.RunFilter(np.array(noisy_speed[0,k]), np.array(drift_noisy_acceleration[0,k]) ))


    #filter_traj = np.array(filter_traj).reshape(1, N)
    filter_speed = np.array(filter_speed).reshape(1, N)
    #filter_acc = np.array(filter_acc).reshape(1, N)

    #filter_trajOffset = np.array(filter_trajOffset).reshape(1, N)
    filterdrift_speed = np.array(filterdrift_speed).reshape(1, N)
    #filter_accOffset = np.array(filter_accOffset).reshape(1, N)


    dataset = [noisy_speed, drift_noisy_speed, true_speed, drift_speed]
    grapher.SetLegend(["Noisy speed (" + str(noise_level) + "%)", "drift_ing noisy speed", "True speed", "drift_ing Speed"], 1)
    grapher.CompareNDdatas(dataset, "speed", "Test CF, speed, sinusoidal", style_adapter=False, width=0.8)

    dataset = [true_speed, true_acceleration, drift_acc, filterdrift_speed]
    grapher.SetLegend(["Theta ", "Omega", "drift_ing Omega",  "Filter acting on\nnoisy Theta & drift_ing noisy Omega"], 1)
    grapher.CompareNDdatas(dataset, "speed", "Test CF, theta, sinusoidal", style_adapter=True,  width=1.5)

    dataset = [true_speed, true_acceleration, noisy_acceleration, filter_speed]
    grapher.SetLegend(["Theta ", "Omega", "Noisy Omega",  "Filter acting on\nnoisy Theta &  Omega"], 1)
    grapher.CompareNDdatas(dataset, "speed", "Test CF, theta, sinusoidal", style_adapter=True,  width=1.5)


    grapher.end()

    return dataset




dataset = main()
