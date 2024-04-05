
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
    def __init__(self, TrajectoryGenerator, logger) -> None:
        self.baseLinearAcceleration = np.zeros((3,))
        self.baseAngularVelocity = np.zeros((3,))
        self.baseOrientation = np.zeros((3,))
        self.q_mes = np.zeros((6,))
        self.v_mes = np.zeros((6,))




        self.generator = TrajectoryGenerator
        if logger is not None :
            self.logger = logger
            self.talkative = True
        else:
            self.talkative = False
        self.iter = 0
        if self.talkative : self.logger.LogTheLog("started Device Emulator")

        return None
    
    def GenerateTrajectory(self,N, NoiseLevelXY, NoiseLevelZ, Drift, NoiseLevelAttitude, T) -> None:
        # generate trajectories and attitudes that are supposed to look like real ones
        self.Traj = np.zeros((N, 3))
        self.Speed = np.zeros((N, 3))
        self.Acc = np.zeros((N, 3))
        self.AccG = np.zeros((N, 3))

        self.Theta = np.zeros((N, 3))
        self.Omega = np.zeros((N, 3))
        self.OmegaDot = np.zeros((N, 3))
        
        # generate X trajectory
        self.generator.Generate("polynomial5", NoiseLevel=NoiseLevelXY, N=N, amplitude=0.5, T=T)
        self.X, self.Xd, self.Xdd = self.generator.GetTrueTraj()
        self.XN, self.XNd, self.XNdd = self.generator.GetNoisyTraj()
        # generate Y trajectory
        self.generator.Generate("polynomial5", NoiseLevel=NoiseLevelXY, N=N, amplitude=0.5, T=T)
        self.Y, self.Yd, self.Ydd = self.generator.GetTrueTraj()
        self.YN, self.YNd, self.YNdd = self.generator.GetNoisyTraj()
        # generate Z trajectory
        self.generator.Generate("sinus", NoiseLevel=NoiseLevelZ, N=N, amplitude=0.05, T=T)
        self.Z, self.Zd, self.Zdd = self.generator.GetTrueTraj()
        self.ZN, self.ZNd, self.ZNdd = self.generator.GetNoisyTraj()
        # ATTITUDE IN RADIANS
        # data from generator are [[x0, x1, x2...]], removing 1D
        # generate X, Y attitude
        self.generator.Generate("sinus", NoiseLevel=NoiseLevelAttitude, N=N, amplitude=15/57, T=T)
        self.RX, self.RXd, self.RXdd = self.generator.GetTrueTraj()
        self.RXN, self.RXNd, self.RXNdd = self.generator.GetNoisyTraj()

        self.generator.Generate("sinus", NoiseLevel=NoiseLevelAttitude, N=N, amplitude=10/57, T=T)
        self.RY, self.RYd, self.RYdd = self.generator.GetTrueTraj()
        self.RYN, self.RYNd, self.RYNdd = self.generator.GetNoisyTraj()
        # generate Z attitude
        self.generator.Generate("polynomial5", NoiseLevel=NoiseLevelAttitude/2, N=N, amplitude=5/57, Drift=Drift, T=T)
        self.RZ, self.RZd, self.RZdd = self.generator.GetTrueTraj()
        self.RZN, self.RZNd, self.RZNdd = self.generator.GetDriftingNoisyTraj()

        # put all that noisy data in the right variables
        self.Traj[:, 0], self.Traj[:, 1], self.Traj[:, 2] = self.XN[0], self.YN[0], self.ZN[0]
        self.Speed[:, 0], self.Speed[:, 1], self.Speed[:, 2] = self.XNd[0], self.YNd[0], self.ZNd[0]
        self.Acc[:, 0], self.Acc[:, 1], self.Acc[:, 2] = self.XNdd[0], self.YNdd[0], self.ZNdd[0]

        self.Theta[:, 0], self.Theta[:, 1], self.Theta[:, 2] = self.RXN[0], self.RYN[0], self.RZN[0]
        self.Omega[:, 0], self.Omega[:, 1], self.Omega[:, 2] = self.RXNd[0], self.RYd[0], self.RZNd[0]
        self.OmegaDot[:, 0], self.OmegaDot[:, 1], self.OmegaDot[:, 2] = self.RXNdd[0], self.RYNdd[0], self.RZNdd[0]

        # acceleration with gravity (ADDED WITHOUT ANY NOISE)

        for j in range(N):
            self.AccG[j, :] = self.Acc[j, :] + utils.rotation( np.array([0, 0, 9.81]), np.array([self.RX[0, j], self.RY[0, j], self.RZ[0, j]])  )


    def Read(self):
        # iterate over the data as if it were real time, and get it ready to be provided to the estimator

        self.baseLinearAcceleration = self.Acc[self.iter, :]
        self.baseLinearAccelerationGravity = self.AccG[self.iter, :]
        self.baseSpeed = self.Speed[self.iter, :]

        self.baseAngularVelocity = self.Omega[self.iter, :]
        self.baseOrientation = self.Theta[self.iter, :]
        self.q_mes = np.zeros((6,))
        self.v_mes = np.zeros((6,))

        self.offset_yaw_IMU = np.zeros(3)
        self.offset_speed_IMU = np.zeros(3)

        self.iter += 1
    
    def GetTranslation(self):
        # outputs data
        return self.Traj, self.Speed, self.Acc
    
    def GetRotation(self):
        # outputs data
        return self.Theta, self.Omega, self.OmegaDot

























