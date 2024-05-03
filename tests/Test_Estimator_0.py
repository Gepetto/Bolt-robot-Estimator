import sys
sys.path.append('/home/nalbrecht/Bolt-Estimator/Bolt-robot---Estimator/src/python')
import numpy as np
from scipy.spatial.transform import Rotation as R


from Bolt_Utils import Log
from Graphics import Graphics
from Bolt_Estimator_0 import Estimator
from DeviceEmulator import DeviceEmulator
from TrajectoryGenerator import TrajectoryGenerator


from Bolt_Filter_Complementary import ComplementaryFilter



def main():
    N = 5000
    T = 10

    # init updates
    testlogger = Log("test", PrintOnFlight=True)
    grapher = Graphics(logger=testlogger)
    generator = TrajectoryGenerator(logger=testlogger)
    device = DeviceEmulator(TrajectoryGenerator=generator, logger=testlogger)

    # generate traj
    device.GenerateTrajectory(N=N, NoiseLevelXY=10, NoiseLevelZ=10, Drift=30, NoiseLevelAttitude=10, T=T)
    # init estimator
    estimator = Estimator(device=device,
                    ModelPath="",
                    UrdfPath="",
                    Talkative=True,
                    logger=testlogger,
                    AttitudeFilterType = "complementary",
                    parametersAF = [2],
                    SpeedFilterType = "complementary",
                    parametersSF = [2],
                    TimeStep = T/N,
                    IterNumber = N)
    
    # plot the pseudo-measured (eg noisy and drifting) generated trajectories
    traj, speed, acc = device.GetTranslation()
    Inputs_translations = [traj.T, speed.T, acc.T]
    theta, omega, omegadot = device.GetRotation()
    Inputs_rotations = [theta.T, omega.T]

    grapher.SetLegend(["traj", "speed", "acc"], 3)
    grapher.CompareNDdatas(Inputs_translations, "", "Trajectory, speed and acceleration as inputed", StyleAdapter=False, width=1.)

    grapher.SetLegend(["theta", "omega"], 3)
    grapher.CompareNDdatas(Inputs_rotations, "", "rotation and rotation speed as inputed", StyleAdapter=False, width=1.)

    
    
    # run the estimator as if
    # device will iterate over generated data each time estimator calls it
    for j in range(N-1):
        estimator.Estimate()
    
    # get data as quat
    Qtheta_estimator = estimator.Get("theta_logs")
    Qtheta_imu = estimator.Get("theta_logs_imu")
    theta_estimator = np.zeros((3, N))
    theta_imu = np.zeros((3, N))
    for j in range(N-1):
        theta_estimator[:, j] = R.from_quat(Qtheta_estimator[:, j]).as_euler('xyz')
        theta_imu[:, j] = R.from_quat(Qtheta_imu[:, j]).as_euler('xyz')
    
    InOut_rotations = [theta.T[:2, :], theta_imu[:2, :], theta_estimator[:2, :]]
    grapher.SetLegend(["theta in", "theta imu","theta out"], 2)
    grapher.CompareNDdatas(InOut_rotations, "theta", "Noisy rot and out rot", StyleAdapter=True, width=1.)
    
    ag = device.AccG
    a = device.Acc
    grapher.SetLegend(["ag", "a"], 3)
    grapher.CompareNDdatas([ag.T, a.T], "acceleration", "g", StyleAdapter=True, width=1.)

    '''

    filter = ComplementaryFilter(ndim=3, parameters=[T/N, 2])
    theta_filter = np.zeros((3, N))

    # run filter over time, with noisy data as inputs
    for k in range(N):
        theta_filter[:, k] = filter.RunFilter(theta.T[:,k], omega.T[:,k])
    
    InOut_rotations = [theta.T[:2, :], theta_filter[:2, :]]
    grapher.SetLegend(["theta in","theta filter"], 2)
    grapher.CompareNDdatas(InOut_rotations, "theta", "theta", StyleAdapter=True, AutoLeg=False, width=1.)
    
    '''

    grapher.end()
    return None









if __name__ == "__main__":
    main()













