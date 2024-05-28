import sys
sys.path.append('/home/nalbrecht/Bolt-Estimator/Bolt-robot---Estimator/src/python')
from Bolt_Utils import utils
from Bolt_Utils import Log

from DataReader import DataReader

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
    generator.Generate("sinus", N=500, T=1, NoiseLevel=10, Drift=20, amplitude=10, avgfreq=0.5, relative=True, traj=None, speed=None, smooth=True)

   



    # get generated data
    TrueTraj, TrueSpeed, TrueAcc = generator.GetTrueTraj()
    NoisyTraj, NoisySpeed, NoisyAcc = generator.GetNoisyTraj()
    DriftNoisyTraj, DriftNoisySpeed, DriftNoisyAcc = generator.GetDriftingNoisyTraj()
    DriftTraj, DriftSpeed, DriftAcc = generator.GetDriftingTraj()


 
    dataset = [TrueTraj, TrueSpeed, TrueAcc]
    grapher.SetLegend(["Traj ", "Speed", "Acceleration"], 1)
    grapher.CompareNDdatas(dataset, "", "TestGenerator : true values", StyleAdapter=False, AutoLeg=False, width=1.5)

    dataset = [NoisyTraj, DriftNoisyTraj, TrueTraj, DriftTraj]
    grapher.SetLegend(["Noisy traj (" + str(NoiseLevel) + "%)", "Drifting noisy traj", "True traj", "Drifting traj"], 1)
    grapher.CompareNDdatas(dataset, "position", "TestGenerator : positions", StyleAdapter=False, AutoLeg=False, width=1)

    dataset = [NoisySpeed, DriftNoisySpeed, TrueSpeed, DriftSpeed]
    grapher.SetLegend(["Noisy speed (" + str(NoiseLevel) + "%)", "Drifting noisy speed", "True speed", "Drifting speed"], 1)
    #grapher.CompareNDdatas(dataset, "speed", "TestGenerator : speeds", StyleAdapter=False, AutoLeg=False, width=0.8)

    dataset = [NoisyAcc, DriftNoisyAcc, TrueAcc, DriftAcc]
    grapher.SetLegend(["Noisy acceleration (" + str(NoiseLevel) + "%)", "Drifting noisy acc", "True acceleration", "Drifting acceleration"], 1)
    #grapher.CompareNDdatas(dataset, "acceleration", "TestGenerator : accelerations", StyleAdapter=False, AutoLeg=False, width=0.8)

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
    testlogger = Log("test", PrintOnFlight=True)
    grapher = Graphics(logger=testlogger)
    
    # prepare reader
    Reader = DataReader(logger=testlogger)
    
    # loading .npy files in DataReader
    Reader.AutoLoad(kfile)
    
    # get base traj and speed
    Traj = Reader.X[:, 1, :]
    print(Traj)
    print(Traj.shape)
    Speed = Reader.V[:, 1, :]

    # start generator
    generator = TrajectoryGenerator(logger=testlogger)
    generator.Generate("custom", N=N, T=T, NoiseLevel=10, Drift=20, amplitude=10, 
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