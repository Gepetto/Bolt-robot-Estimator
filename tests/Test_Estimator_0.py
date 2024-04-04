import sys
sys.path.append('/home/nalbrecht/Bolt-Estimator/Bolt-robot---Estimator/src/python')
import numpy as np
from Bolt_Utils import Log
from Graphics import Graphics
from Bolt_Estimator_0 import Estimator
from DeviceEmulator import DeviceEmulator
from TrajectoryGenerator import TrajectoryGenerator




def main():
    N = 100

    # init updates
    testlogger = Log("test", PrintOnFlight=True)
    grapher = Graphics(logger=testlogger)
    generator = TrajectoryGenerator(logger=testlogger)
    device = DeviceEmulator(TrajectoryGenerator=generator, logger=testlogger)

    # generate traj
    device.GenerateTrajectory(N=N+10, NoiseLevelXY=15, NoiseLevelZ=5, Drift=10, NoiseLevelAttitude=10, T=3)
    # init estimator
    estimator = Estimator(device=device,
                    ModelPathth="",
                    UrdfPath="",
                    Talkative=True,
                    logger=testlogger,
                    AttitudeFilterType = "complementary",
                    parametersAF = (0, 0, 0, 0),
                    SpeedFilterType = "complementary",
                    parametersSF = (0, 0, 0, 0),
                    TimeStep = None,
                    IterNumber = N)
    
    # plot the pseudo-measured (eg noisy and drifting) generated trajectories
    traj, speed, acc = device.GetTranslation()
    Inputs_translations = [traj.T, speed.T, acc.T]
    theta, omega, omegadot = device.GetRotation()
    Inputs_rotations = [theta.T, omega.T, omegadot.T]

    grapher.SetLegend(["traj", "speed", "acc"], 3)
    grapher.CompareNDdatas(Inputs_translations, "", "Trajectory, speed and acceleration as inputed", StyleAdapter=False, AutoLeg=False, width=1.)

    grapher.SetLegend(["theta", "omega", "angular acc"], 3)
    grapher.CompareNDdatas(Inputs_rotations, "", "rotation and derivatives as inputed", StyleAdapter=False, AutoLeg=False, width=1.)

    
    # run the estimator as if
    # device will iterate over generated data each time estimator calls it
    #for j in range(N):
    #    estimator.Estimate()




    grapher.end()
    return None









if __name__ == "__main__":
    main()













