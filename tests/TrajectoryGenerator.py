import numpy as np
from numpy.polynomial import Polynomial
import sys
sys.path.append('/home/nalbrecht/Bolt-Estimator/Bolt-robot---Estimator/src/python')
from Bolt_Utils import Sinus, Cosinus, Exp



'''
Code that :
    simulates a robot's trajectory, following several simple models
    derives its speed and accceleration
    adds a white noise to all the upper data
    provides the noisy speed and acceleration to the Estimator class

'''







class TrajectoryGenerator:
    def __init__(self,  logger=None):
        if logger is not None :
            # uses the same logger class as the estimator class
            self.logger = logger
            self.talkative = True
        else:
            self.talkative = False
        
        # amplitude of generated signals
        self.amplitude = 1
        self.NoiseLevel = 10    # %
        self.Drift = 10         # %

        #initialize datas
        self.trajectory = np.zeros((3,1))
        self.speed = np.zeros((3,1))
        self.acceleration = np.zeros((3,1))
        self.noisy_trajectory, self.noisy_speed, self.noisy_acceleration = None, None, None

        if self.talkative : self.logger.LogTheLog("started Trajectory Generator", style="subtitle")
    
    def Generate(self, TypeOfTraj, N=1000, T=1, NoiseLevel=10, Drift=10, amplitude=1, avgfreq=3, relative=True, traj=None, speed=None, smooth=True):
        # time increment, in seconds
        self.dt = T/N
        self.N = N
        self.T = T
        # parameters
        self.avgfreq = avgfreq
        self.amplitude = amplitude
        self.NoiseLevel = NoiseLevel    # %
        self.Drift = Drift              # %
        # set up trajectory type, create trajectory, speed and acceleration
        self.smooth = smooth
        self.TypeOfTraj = TypeOfTraj
        self.TrajType(TypeOfTraj, traj=traj, speed=speed)
        # create fake noise
        self.NoiseMaker = Metal(self.trajectory, self.speed, self.acceleration, NoiseLevel, DriftingCoeff=Drift, logger=self.logger)
        self.noisy_trajectory, self.noisy_speed, self.noisy_acceleration = self.NoiseMaker.NoisyFullTrajectory()
        # create drift in data
        self.drifting_trajectory, self.drifting_speed, self.drifting_acceleration = self.NoiseMaker.DriftingFullTrajectory()
        self.noisy_drifting_trajectory, self.noisy_drifting_speed, self.noisy_drifting_acceleration = self.NoiseMaker.DriftingNoisyFullTrajectory()

        

        

    def GetTrueTraj(self):
        return self.trajectory, self.speed, self.acceleration
    def GetNoisyTraj(self):
        return self.noisy_trajectory, self.noisy_speed, self.noisy_acceleration
    def GetDriftingTraj(self):
        return self.drifting_trajectory, self.drifting_speed, self.drifting_acceleration
    def GetDriftingNoisyTraj(self):
        return self.noisy_drifting_trajectory, self.noisy_drifting_speed, self.noisy_drifting_acceleration
       

    

        
    def TrajType(self, TypeOfTraj, traj=None, speed=None):
        # calls the appropriate trajectory generator
        self.TypeOfTraj = TypeOfTraj
        # wether or not speed and acceleration has to be derived analytically
        self.analytical=True

        if TypeOfTraj =='linear':
            self.trajectory, self.speed, self.acceleration = self.TrajectoryLinear(self.N)
        elif TypeOfTraj[:10] == 'polynomial':
            # last character can be the polynom's order
            k=5
            if TypeOfTraj[-1].isnumeric(): k = int(TypeOfTraj[-1])
            self.trajectory, self.speed, self.acceleration = self.TrajectoryPolynomial(order=k)

        elif TypeOfTraj == "sinus" or TypeOfTraj == "sinusoidal":
            self.trajectory, self.speed, self.acceleration = self.TrajectorySinusoidal()

        elif TypeOfTraj == "exponential" or TypeOfTraj == "exp":
            self.trajectory, self.speed, self.acceleration = self.TrajectoryExponential()
        
        elif TypeOfTraj == "multiexponential" or TypeOfTraj == "multiexp":
            self.trajectory, self.speed, self.acceleration = self.TrajectoryMultiExponential(3)

        elif TypeOfTraj == 'custom':
            self.analytical = False
            if traj is not None :
                self.trajectory = traj.copy()
            else :
                self.logger.LogTheLog("No trajectory given !", style="warn")
            if speed is None:
                # speed is not given, derive speed and acceleration
                self.acceleration = self.MakeSpeedFromTrajectory(traj)
            else :
                # speed is given
                if speed.shape != self.trajectory.shape :
                    # check if data are consistent
                    self.logger.LogTheLog("Wrong dimensions for given custom speed and trajectory, deriving speed from traj", style="warn")
                    self.MakeAccelerationFromTrajectory(traj)
                else:
                    self.speed = speed.copy()
                    self.acceleration = self.MakeAccelerationFromSpeed(speed)
        else:
            self.logger.LogTheLog("undetermined trajectory type", style="warn")
            return None, None, None
        # returned data are 2D arrays
        self.logger.LogTheLog("generated " + TypeOfTraj + " trajectory")
        return self.trajectory, self.speed, self.acceleration




    def TrajectoryPolynomial(self, order=5):
        T_array = np.linspace(0, self.T, self.N)
        # randomly generates coeff that matches approx. speed
        coeffs = np.random.random(order+1)
        if self.T>1:
            coeffs = coeffs * (self.amplitude / (coeffs[0] * self.T**order))
        else :
            coeffs = coeffs * (self.amplitude / coeffs[-1])

        # trajectory to be followed
        traj = Polynomial(coeffs)
        speed = traj.deriv()
        acc = speed.deriv()
        # evaluate polynomial functions
        self.trajectory = traj(T_array).reshape((1, self.N))
        self.speed = speed(T_array).reshape((1, self.N))
        self.acceleration = acc(T_array).reshape((1, self.N))
        # all done
        return self.trajectory, self.speed, self.acceleration
    

    def TrajectorySinusoidal(self):
        T_array = np.linspace(0, self.T, self.N)
        # randomly generates coeff that matches approx. speed and frequency
        a = (np.random.random(1) + 0.5)*self.amplitude
        omega = 2*np.pi*self.avgfreq + 0.3*(np.random.random(1) - 0.5)*2*np.pi*self.avgfreq
        # trajectory to be followed
        traj = Sinus(a, omega)
        speed = traj.deriv()
        acc = speed.deriv()
        # evaluate sinusoidal functions
        self.trajectory = traj.evaluate(T_array).reshape((1, self.N))
        self.speed = speed.evaluate(T_array).reshape((1, self.N))
        self.acceleration = acc.evaluate(T_array).reshape((1, self.N))
        # all done
        return self.trajectory, self.speed, self.acceleration
    

    def TrajectoryExponential(self, ImposedInit=0):
        T_array = np.linspace(0, self.T, self.N)
        # randomly generates coeff that matches approx. speed and frequency
        if ImposedInit == 0:
            C = np.random.random(1)*amplitude*self.T
        else :
            C = ImposedInit
        w = (np.random.random(1)+0.5)*w
        # trajectory to be followed
        T = Exp(C, w)
        S = T.deriv()
        A = S.deriv()
        # evaluate exponential functions
        trajectory = T.evaluate(T_array).reshape((1, self.N))
        speed = S.evaluate(T_array).reshape((1, self.N))
        acceleration = A.evaluate(T_array).reshape((1, self.N))
        # all done
        return trajectory, speed, acceleration
    

    def TrajectoryMultiExponential(self, k, w=5): # will not work (because of changes in T and self.T)
        # for a succession of exponential
        T_array = np.linspace(0, self.T, self.N)
        # each line is a different exponential
        n = int(np.floor(self.N/k)) + 1 # n*k > N, data will be truncated
        self.trajectory, self.speed, self.acceleration = np.zeros((k, n)), np.zeros((k, n)), np.zeros((k, n))
         
        self.trajectory[0,:], self.speed[0,:], self.acceleration[0,:] = self.TrajectoryExponential(n, T_array[n], w)
        for j in range(1, k):
            self.trajectory[j,:], self.speed[j,:], self.acceleration[j,:] = self.TrajectoryExponential(n, T_array[j*n], w, ImposedInit=self.trajectory[j-1, 0])
        # flatten and truncate
        return self.trajectory.reshape((1, -1))[:, :self.N], self.speed.reshape((1, -1))[:, :self.N], self.acceleration.reshape((1, -1))[:, :self.N]
       
        
        
    def MakeSpeedFromTrajectory(self, traj):
        # derives the speed
        if self.talkative : self.logger.LogTheLog("Speed is derived numerically from trajectory", style="warn")
        D, N= np.shape(traj)
        self.speed = np.zeros((D,N))
        self.speed[:, 1:] = (traj[:, 1:] - traj[:, 0:-1])/self.dt
        if self.smooth : self.speed = self.SimpleSmoother(self.speed)
        return self.speed
        
    def MakeAccelerationFromTrajectory(self, traj):
        # derives the acceleration and the speed
        D, N= np.shape(traj)
        self.acceleration = np.zeros((D,N))
        self.MakeSpeedFromTrajectory(traj)
        self.acceleration[:, 1:] = (self.speed[:, 1:] - self.speed[:, 0:-1])/self.dt
        if self.smooth : self.acceleration = self.SimpleSmoother(self.acceleration)
        return self.acceleration

    def MakeAccelerationFromSpeed(self, speed):
        # derives the acceleration and the speed
        if self.talkative : logger.LogTheLog("Acceleration is derived numerically from speed", style="warn")
        D, N= np.shape(speed)
        self.acceleration = np.zeros((D,N))
        self.acceleration[:, 1:] = (speed[:, 1:] - speed[:, 0:-1])/self.dt
        if self.smooth : self.acceleration = self.SimpleSmoother(self.acceleration)
        return self.acceleration
    
    def SimpleSmoother(self, data, smoothing=70):
        # data must be a 2D array [[x], [y], [z]]
        Smoothdata = np.zeros((data.shape))
        # max differences on x, y, z (useless for now)
        scale = abs(np.max(data, axis=0) - np.min(data, axis=0))
        trigger = (100-smoothing)/100 * scale
        # translate date left and right
        MovedLeft = data.copy()
        MovedLeft[:, 1:-1] = data[:, :-2]
        MovedRight = data.copy()
        MovedRight[:, 1:-1] = data[:, 2:]
        # average data
        Smoothdata = (MovedLeft + MovedRight + data)/3
        return Smoothdata

    
    def AdaptTrajectory(self):
        # turns trajectory to accelerometer-like data
        # x,v,a -> w, a
        return self.rotspeed, self.acceleration







# a class to add noise 
class Metal:
    def __init__(self, traj, speed=None, acc=None, NoiseLevel=0.1, DriftingCoeff=0.05, logger=None):
        # the absolute or relative noise to add to the data
        self.NoiseLevel = NoiseLevel
        self.DriftingCoeff = DriftingCoeff
        # dimensions and length of traj
        self.D, self.N = np.shape(traj)

        # intiialize trajectory, speed, and acceleration
        self.RealTraj = traj.copy()
        self.RealSpeed, self.RealAcc = np.zeros((self.D, 1)), np.zeros((self.D, 1))
        if speed is not None : self.RealSpeed = speed.copy()
        if acc is not None : self.RealAcc = acc.copy()


        if logger is not None :
            # uses the same logger class as the estimator class
            self.logger = logger
            self.talkative = True
            self.logger.LogTheLog("started noise generator Metal")
            if speed is None : logger.LogTheLog("no speed provided", style="warn")
            if acc is None : logger.LogTheLog("no acceleration provided", style="warn")
        else:
            self.talkative = False
        

        
    def makeNoise(self, data, AutoAdjustAmplitude=True):
        # tune the noise level (absolute or proportionnal to signal)
        amplitude = self.NoiseLevel/100
        if AutoAdjustAmplitude : 
            amplitude = np.max(abs(data)) * self.NoiseLevel / 100
            #self.logger.LogTheLog(str(np.max(data)), style="warn")
        return data + np.random.normal(loc=0.0, scale=amplitude, size=(self.D,self.N))
    
    def makeDrift(self, data, AutoAdjustAmplitude=True):
        drift = self.DriftingCoeff/100
        if AutoAdjustAmplitude : 
            drift = (np.max(abs(data)) - np.min(data)) * self.DriftingCoeff / 100
        return data + np.linspace(np.zeros(self.D), drift*np.ones(self.D), self.N).T
    
    def NoisyFullTrajectory(self):
        self.logger.LogTheLog("Adding noise on x, y, z provided data")
        return self.makeNoise(self.RealTraj), self.makeNoise(self.RealSpeed), self.makeNoise(self.RealAcc)
    
    def DriftingFullTrajectory(self):
        self.logger.LogTheLog("Adding drift on x, y, z provided data")
        return self.makeDrift(self.RealTraj), self.makeDrift(self.RealSpeed), self.makeDrift(self.RealAcc)
    
    def DriftingNoisyFullTrajectory(self):
        self.logger.LogTheLog("Adding drift and noise on x, y, z provided data")
        return self.makeDrift(self.makeNoise(self.RealTraj)), self.makeDrift(self.makeNoise(self.RealSpeed)), self.makeDrift(self.makeNoise(self.RealAcc))
    
    def measurePos(self):
        self.NoisyTraj = self.makeNoise(self.RealTraj)
        return self.NoisyTraj
    
    def measureSpeed(self):
        self.NoisySpeed = self.makeNoise(self.RealSpeed)
        return self.NoisySpeed
    
    def measureAcc(self):
        self.NoisyAcc = self.makeNoise(self.RealAcc)
        return self.NoisyAcc
    

        
 













    







