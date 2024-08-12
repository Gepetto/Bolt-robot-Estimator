

from bolt_estimator.utils.utils import utils, Log

from bolt_estimator.data.data_reader import DataReader

from bolt_estimator.utils.trajectory_generator import TrajectoryGenerator, Metal
from bolt_estimator.utils.graphics import Graphics
from bolt_estimator.estimator.filter_complementary import ComplementaryFilter
import numpy as np



def main(N=500, noise_level=20, drift=40):

    # generate useful objects
    test_logger = Log("test", print_on_flight=True)
    grapher = Graphics(logger=test_logger)

    # start generator
    generator = TrajectoryGenerator(logger=test_logger)
    generator.Generate("sinus", N=500, T=1, noise_level=10, drift=20, amplitude=10, avgfreq=0.5, relative=True, traj=None, speed=None, smooth=True)

   



    # get generated data
    true_traj, true_speed, true_acceleration = generator.GetTrueTraj()
    noisy_traj, noisy_speed, noisy_acceleration = generator.GetNoisyTraj()
    drifting_noisy_traj, drifting_noisy_speed, drifting_noisy_acceleration = generator.GetDriftingNoisyTraj()
    drifting_traj, drifting_speed, drifting_acceleration = generator.GetDriftingTraj()


 
    dataset = [true_traj, true_speed, true_acceleration]
    grapher.SetLegend(["traj ", "speed", "acceleration"], 1)
    grapher.CompareNDdatas(dataset, "", "TestGenerator : true values", style_adapter=False, width=1.5)

    dataset = [noisy_traj, drifting_noisy_traj, true_traj, drifting_traj]
    grapher.SetLegend(["Noisy traj (" + str(noise_level) + "%)", "drifting noisy traj", "True traj", "drifting traj"], 1)
    grapher.CompareNDdatas(dataset, "position", "TestGenerator : positions", style_adapter=False, width=1)

    dataset = [noisy_speed, drifting_noisy_speed, true_speed, drifting_speed]
    grapher.SetLegend(["Noisy speed (" + str(noise_level) + "%)", "drifting noisy speed", "True speed", "drifting speed"], 1)
    #grapher.CompareNDdatas(dataset, "speed", "TestGenerator : speeds", style_adapter=False, width=0.8)

    dataset = [noisy_acceleration, drifting_noisy_acceleration, true_acceleration, drifting_acceleration]
    grapher.SetLegend(["Noisy acceleration (" + str(noise_level) + "%)", "drifting noisy acc", "True acceleration", "drifting acceleration"], 1)
    #grapher.CompareNDdatas(dataset, "acceleration", "TestGenerator : accelerations", style_adapter=False, , width=0.8)

    grapher.end()


    return dataset




def improve_constant_data():
    '''add acceleration to data'''
    
    # directory to load and add data to
    kfile = 2
    prefix = "/home/nalbrecht/Bolt-Estimator/bipedal-control/bipedal-control/Données cancer niels/" + str(kfile) + "/"
    N = 1000
    T = 10
    
    
    # generate useful objects
    test_logger = Log("test", print_on_flight=True)
    grapher = Graphics(logger=test_logger)
    
    # prepare reader
    Reader = DataReader(logger=test_logger)
    
    # loading .npy files in DataReader
    Reader.AutoLoad(kfile)
    
    # get base traj and speed
    traj = Reader.X[:, 1, :]
    print(traj)
    print(traj.shape)
    speed = Reader.V[:, 1, :]

    # start generator
    generator = TrajectoryGenerator(logger=test_logger)
    generator.Generate("custom", N=N, T=T, noise_level=10, drift=20, amplitude=10, 
                       avgfreq=0.5, relative=True, traj=traj, speed=speed, smooth=True)
    
    # compute speed to check consistency
    s = generator.MakeSpeedFromTrajectory(traj)
    # computing acceleration
    a = generator.MakeAccelerationFromSpeed(speed)
    # saving acceleration to X and V shape
    acc = np.zeros(N, 19, 3)
    acc[:, 1, :] = a
    
   
    np.save(prefix + "A_array_" + str(kfile) )
    
    grapher.SetLegend(["True", "computed"], ndim=3)
    grapher.CompareNDdatas([speed[:, 1, :].transpose(), s], datatype="speed of bolt's base")
    grapher.end()

    





























dataset = main()
