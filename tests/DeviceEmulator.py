
import numpy as np
import sys
sys.path.append('/home/nalbrecht/Bolt-Estimator/Bolt-robot---Estimator/src/python')
from Bolt_Utils import utils, Log
from TrajectoryGenerator import TrajectoryGenerator, Metal
from Graphics import Graphics
from Bolt_Filter_Complementary import ComplementaryFilter


'''
A Device Emulator class that emulates the device used in Bolt_Estimator
Send the estimator pseudo-data from IMU

'''


class DeviceEmulator():
    def __init__(self, TrajectoryGenerator) -> None:
        self.baseLinearAcceleration = np.zeros((3,))
        self.baseAngularVelocity = np.zeros((3,))
        self.baseOrientation = np.zeros((3,))
        self.q_mes = np.zeros((3,))
        self.v_mes = np.zeros((3,))
        self.generator = TrajectoryGenerator
        self.iter = 0
        return None
    
    def GenerateTrajectory(N, NoiseLevelXY, NoiseLevelZ, Drift, NoiseLevelAttitude, T) -> None:
        # generate X trajectory
        self.generator.Generate("polynomial5", NoiseLevelXY, N, amplitude=0.5, T=T)
        self.X, self.Xd, self.Xdd = generator.GetTrueTraj()
        self.XN, self.XNd, self.XNdd = generator.GetNoisyTraj()
        # generate Y trajectory
        self.generator.Generate("polynomial5", NoiseLevelXY, N, amplitude=0.5, T=T)
        self.Y, self.Yd, self.Ydd = generator.GetTrueTraj()
        self.YN, self.YNd, self.YNdd = generator.GetNoisyTraj()
        # generate Z trajectory
        self.generator.Generate("sinus", NoiseLevelZ, N, amplitude=0.05, T=T)
        self.z, self.Zd, self.Zdd = generator.GetTrueTraj()
        self.ZN, self.ZNd, self.ZNdd = generator.GetNoisyTraj()
        # ATTITUDE IN DEGREE (? CHK)
        # generate X, Y attitude
        self.generator.Generate("sinus", NoiseLevelAttitude, N, amplitude=15, T=T)
        self.RX, self.RXd, self.RXdd = generator.GetTrueTraj()
        self.RXN, self.RXNd, self.RXNdd = generator.GetNoisyTraj()

        self.generator.Generate("sinus", NoiseLevelAttitude, N, amplitude=10, T=T)
        self.RY, self.RYd, self.RYdd = generator.GetTrueTraj()
        self.RYN, self.RYNd, self.RYNdd = generator.GetNoisyTraj()
        # generate Z attitude
        self.generator.Generate("polynomial5", NoiseLevelAttitude/2, N, amplitude=5, Drift=Drift, T=T)
        self.RZ, self.RZd, self.RZdd = generator.GetTrueTraj()
        self.RZN, self.RZNd, self.RZNdd = generator.GetNoisyTraj()
    
    def IMUEmulator(self):
        # iterate over the data as if it were real time, and get it ready to be provided to the estimator

        self.baseLinearAcceleration = np.zeros((3,))
        self.baseAngularVelocity = np.zeros((3,))
        self.baseOrientation = np.zeros((3,))
        self.q_mes = np.zeros((3,))
        self.v_mes = np.zeros((3,))
        self.generator = TrajectoryGenerator

        self.iter += 1















