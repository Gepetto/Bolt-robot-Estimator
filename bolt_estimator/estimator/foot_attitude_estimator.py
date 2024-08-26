import numpy as np


'''
based on
    Tilt estimator for 3D non-rigid pendulum
    Mehdi Benallegue, Abdelaziz Benallegue, Yacine Chitour
    2018

Uses IMU and kinematics data to estimate the rotation betwwen world frame and foot frame
'''



class FootAttitudeEstimator():
    def __init__(self,
                parameters=[0.001, 2],
                dt=0.01,
                name="[Foot attitude estimator]",
                talkative=False,
                logger=None,
                ) -> None:
        self.name=name
        self.parameters = parameters
        self.talkative = talkative
        # number of filter run
        self.k = 0
        # timestep
        self.dt = dt
        # gains
        self.alpha, self.beta = self.parameters[0], self.parameters[1]
        # rotation matrix (contact foot to sensor)
        self.cRs = np.eye(3)
        self.cRsT = np.transpose(self.cRs)
        #print(self.cRs, self.cRsT)
        # rotation speed (contact foot to sensor)
        self.ws = np.zeros((3,3))
        # position of sensor in contact foot frame
        self.cPs = np.zeros((3, 1))
        # state variables and derivatives
        self.x1 = np.zeros((3, 1))
        self.x2 = np.zeros((3, 1))
        self.x1dot = np.zeros((3, 1))
        self.x2dot = np.zeros((3, 1))

        # logs
        if logger is not None : 
            self.logger = logger
            self.talkative = False
        if self.talkative : self.logger.LogTheLog("Estimator '" + self.name + "' initialized with parameters " + str(self.parameters))


    def Y1(self, yg):
        return self.cRsT @ (yg - self.cRsT@self.ws)
    def X1(self, y1):
        return self.S(self.cPs)@y1 - self.cPsdot
    def S(self, x):
        #print("S in Ben", x.shape)
        x1, x2, x3 = x[0,0], x[0,1], x[0,2]
        return np.array([[0, -x3, x2],
                        [x3, 0, -x1],
                        [-x2, x1, 0]])
    
    def RefreshKin(self, imu_kin_pos, imu_kin_rot) :
        self.cPsdot = (imu_kin_pos - self.cPs)/self.dt
        self.ws = (imu_kin_rot - self.cRs)/self.dt

        self.cPs = imu_kin_pos
        self.cRs = imu_kin_rot
        return None
        


    # run estimator
    # take :
        # Base pose with regard to the foot touching the ground
        # accelerometer and gyrometer acc. and rot.
    def Estimator(self, imu_kin_pos, imu_kin_rot, ya, yg) -> np.ndarray:
        # run estimator
        if self.k==0:
            # estimator runs for the first time
            pass
        
        # updates kinematics data
        self.RefreshKin(imu_kin_pos, imu_kin_rot)
        # updates variables
        self.y1 = self.Y1(yg)
        
        # compute derivatives        
        self.x1dot = -self.S(self.y1)@self.x1 + 9.81*self.x2 - self.cRs@ya + self.alpha * (self.X1(self.y1) - self.x1)
        self.x2dot = -self.S(self.y1 - self.beta*self.S(self.x2)*(self.X1(self.y1)-self.x1)) @ self.x2
        # updates x1 and x2 estimates
        self.x1 = self.x1 + self.dt * self.x1dot
        self.x2 = self.x2 + self.dt * self.x2dot
        
        # convert x2 to exepcted data format
        self.foot_attitude = self.x2
        self.k += 1
        return self.foot_attitude
    



    
