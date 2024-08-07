

from bolt_estimator.utils.Utils import utils, Log

from bolt_estimator.data.DataReader import DataReader

from bolt_estimator.utils.TrajectoryGenerator import TrajectoryGenerator, Metal
from bolt_estimator.utils.Graphics import Graphics
from bolt_estimator.estimator.Filter_Complementary import ComplementaryFilter
import numpy as np



def main(N=500, noise_level=20, drift=40):

    # generate useful objects
    test_logger = Log("test", PrintOnFlight=True)
    grapher = Graphics(logger=test_logger)

    # start generator
    generator = TrajectoryGenerator(logger=test_logger)
    generator.Generate("sinus", N=500, T=1, NoiseLevel=10, Drift=20, amplitude=10, avgfreq=0.5, relative=True, traj=None, speed=None, smooth=True)

   



    # get generated data
    true_traj, true_speed, true_acceleration = generator.GetTrueTraj()
    noisy_traj, noisy_speed, noisy_acceleration = generator.Getnoisy_traj()
    drifting_noisy_traj, drifting_noisy_speed, drifting_noisy_acceleration = generator.GetDriftingnoisy_traj()
    drifting_traj, drifting_speed, drifting_acceleration = generator.GetDriftingTraj()


 
    dataset = [true_traj, true_speed, true_acceleration]
    grapher.SetLegend(["Traj ", "Speed", "Acceleration"], 1)
    grapher.CompareNDdatas(dataset, "", "TestGenerator : true values", StyleAdapter=False, width=1.5)

    dataset = [noisy_traj, drifting_noisy_traj, true_traj, drifting_traj]
    grapher.SetLegend(["Noisy traj (" + str(noise_level) + "%)", "drifting noisy traj", "True traj", "drifting traj"], 1)
    grapher.CompareNDdatas(dataset, "position", "TestGenerator : positions", StyleAdapter=False, width=1)

    dataset = [noisy_speed, drifting_noisy_speed, true_speed, drifting_speed]
    grapher.SetLegend(["Noisy speed (" + str(noise_level) + "%)", "drifting noisy speed", "True speed", "drifting speed"], 1)
    #grapher.CompareNDdatas(dataset, "speed", "TestGenerator : speeds", StyleAdapter=False, width=0.8)

    dataset = [noisy_acceleration, drifting_noisy_acceleration, true_acceleration, drifting_acceleration]
    grapher.SetLegend(["Noisy acceleration (" + str(noise_level) + "%)", "drifting noisy acc", "True acceleration", "drifting acceleration"], 1)
    #grapher.CompareNDdatas(dataset, "acceleration", "TestGenerator : accelerations", StyleAdapter=False, , width=0.8)

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
    test_logger = Log("test", PrintOnFlight=True)
    grapher = Graphics(logger=test_logger)
    
    # prepare reader
    Reader = DataReader(logger=test_logger)
    
    # loading .npy files in DataReader
    Reader.AutoLoad(kfile)
    
    # get base traj and speed
    Traj = Reader.X[:, 1, :]
    print(Traj)
    print(Traj.shape)
    Speed = Reader.V[:, 1, :]

    # start generator
    generator = TrajectoryGenerator(logger=test_logger)
    generator.Generate("custom", N=N, T=T, noise_level=10, drift=20, amplitude=10, 
                       avgfreq=0.5, relative=True, traj=Traj, speed=Speed, smooth=True)
    
    # compute speed to check consistency
    s = generator.MakeSpeedFromTrajectory(Traj)
    # computing acceleration
    a = generator.MakeAccelerationFromSpeed(Speed)
    # saving acceleration to X and V shape
    Acc = np.zeros(N, 19, 3)
    Acc[:, 1, :] = a
    
   
    np.save(prefix + "A_array_" + str(kfile) )
    
    grapher.SetLegend(["True", "computed"], ndim=3)
    grapher.CompareNDdatas([Speed[:, 1, :].transpose(), s], datatype="speed of bolt's base")
    grapher.end()

    





























dataset = main()
