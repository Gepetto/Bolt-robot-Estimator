import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
import sys
sys.path.append('/home/nalbrecht/Bolt-Estimator/Bolt-robot---Estimator/src/python')
from Bolt_Utils import Sinus, Cosinus



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
        
        # max average displacement speed during trajectory
        self.displacement = 0.3 # bolt envisionned average walking speed, m/s

        #initialize datas
        self.trajectory = np.zeros((3,1))
        self.speed = np.zeros((3,1))
        self.acceleration = np.zeros((3,1))
        self.noisy_trajectory, self.noisy_speed, self.noisy_acceleration = None, None, None

        if self.talkative : self.logger.LogTheLog("started Trajectory Generator")
    
    def Generate(self, TypeOfTraj, N=1000, T=1, NoiseLevel=0.1, relative=True):
        # time increment, in seconds
        self.dt = T/N
        self.N = N
        self.T = T
        # set up trajectory type, create trajectory, speed and acceleration
        self.TypeOfTraj = TypeOfTraj
        self.TrajType(TypeOfTraj, self.N, self.T)
        if not self.analytical :
            # speed and acceleration could not be derived analytically
            self.speed, self.acceleration = self.MakeSpeedFromTrajectory(self.trajectory), self.MakeAccelereationFromTrajectory(self.trajectory)
        # create fake noise
        self.NoiseMaker = Metal(self.trajectory, self.speed, self.acceleration, NoiseLevel)
        self.noisy_trajectory, self.noisy_speed, self.noisy_acceleration = self.NoiseMaker.NoisyFullTrajectory()
        

    def GetTrueTraj(self):
        return self.trajectory, self.speed, self.acceleration
    def GetNoisyTraj(self):
        return self.noisy_trajectory, self.noisy_speed, self.noisy_acceleration
    
        


        
    
    

        
    def TrajType(self, TypeOfTraj, N, T, traj=np.zeros((1, 1))):
        # calls the appropriate trajectory generator
        self.TypeOfTraj = TypeOfTraj
        # wether or not speed and acceleration has to be derived analytically
        self.analytical=True

        if TypeOfTraj =='linear':
            self.trajectory, self.speed, self.acceleration = self.TrajectoryLinear(N)
        elif TypeOfTraj[:10] == 'polynomial':
            # last character can be polynom order
            k=5
            if TypeOfTraj[-1].isalpha(): k = int(TypeOfTraj[-1])
            self.trajectory, self.speed, self.acceleration = self.TrajectoryPolynomial(N, T, order=k)

        elif TypeOfTraj == "sinus" or TypeOfTraj == "sinusoidal":
            self.trajectory, self.speed, self.acceleration = self.TrajectorySinusoidal(N, T)

        elif TypeOfTraj == 'custom':
            self.analytical = False
            self.trajectory = traj
        else:
            self.logger.LogTheLog("undetermined trajectory type", style="warn")
            return None, None, None
        # returned data is a 2D array
        self.logger.LogTheLog("generated " + TypeOfTraj + " trajectory")
        return self.trajectory, self.speed, self.acceleration



    
    def TrajectoryLinear(self, N):
        # chose an angle
        alpha = np.random.random() *np.pi/2
        if self.talkative : self.logger.LogTheLog('angle chosen (°) : '+ str(alpha*180/np.pi))
        self.trajectory = np.zeros((3,N))
        def PositionLaw(x):
            # so that speed is no constant
            return np.sin(x/N/2*np.pi)
        self.trajectory[0,:] = np.cos(alpha)*PositionLaw(np.linspace(0,N, num=N))
        self.trajectory[1,:] = np.sin(alpha)*PositionLaw(np.linspace(0,N, num=N))
        return self.trajectory, self.speed, self.acceleration



    def TrajectoryPolynomial(self, N, T, order=5):
        # math object 'X' called t because it refers to time here
        t = Polynomial([1, 0])
        T_array = np.linspace(0, T, N)
        # randomly generates coeff that matches approx. speed
        coeffs = np.random.random(order)*self.displacement*self.T 
        # trajectory to be followed
        traj = Polynomial(coeffs)
        speed = traj.deriv()
        acc = speed.deriv()
        # evaluate polynomial functions
        self.trajectory = traj(T_array).reshape((1, N))
        self.speed = speed(T_array).reshape((1, N))
        self.acceleration = acc(T_array).reshape((1, N))
        # all done
        return self.trajectory, self.speed, self.acceleration
    


    def TrajectorySinusoidal(self, N, T):
        T_array = np.linspace(0, T, N)
        # randomly generates coeff that matches approx. speed and frequency
        a = np.random.random(1)*self.displacement*self.T
        omega = np.random.random(1)* 2*np.pi* 5 
        # trajectory to be followed
        traj = Sinus(a, omega)
        speed = traj.deriv()
        acc = speed.deriv()
        # evaluate sinusoidal functions
        self.trajectory = traj.evaluate(T_array).reshape((1, N))
        self.speed = speed.evaluate(T_array).reshape((1, N))
        self.acceleration = acc.evaluate(T_array).reshape((1, N))
        # all done
        return self.trajectory, self.speed, self.acceleration
       
        
        
    def MakeSpeedFromTrajectory(self, traj):
        # derives the speed
        if self.analytical :
            if self.talkative : logger.LogTheLog("The speed should be derived analytically, yet is derived numerically", style="warn")
        D, N= np.shape(traj)
        self.speed = np.zeros((3,N))
        self.speed[:, 1:] = (traj[:, 1:] - traj[:, 0:-1])/dt
        
    def MakeAccelereationFromTrajectory(self, traj):
        # derives the acceleration
        D, N= np.shape(traj)
        self.MakeSpeedFromTrajectory(traj)
        self.acc[:, 1:] = (self.speed[:, 1:] - self.speed[:, 0:-1])/dt
        








# a class to add noise 
class Metal:
    def __init__(self, traj, speed=None, acc=None, NoiseLevel=0.1, logger=None):
        # the absolute or relative noise to add to the data
        self.NoiseLevel = NoiseLevel
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
        amplitude = self.NoiseLevel
        if AutoAdjustAmplitude : 
            amplitude = abs(np.max(data) * self.NoiseLevel / 100)
        return data + np.random.normal(loc=0.0, scale=amplitude, size=(self.D,self.N))
    
    def NoisyFullTrajectory(self):
        return self.makeNoise(self.RealTraj), self.makeNoise(self.RealSpeed), self.makeNoise(self.RealAcc)
    
    def measurePos(self):
        self.NoisyTraj = self.makeNoise(self.RealTraj)
        return self.NoisyTraj
    
    def measureSpeed(self):
        self.NoisySpeed = self.makeNoise(self.RealSpeed)
        return self.NoisySpeed
    
    def measureAcc(self):
        self.NoisyAcc = self.makeNoise(self.RealAcc)
        return self.NoisyAcc
    

        
 














    
# a class for plotting the computed trajectories and comparing them
class Graphics:
    def __init__(self, logger=None):
        self.currentColor = 0
        self.colors = ['#8f2d56', '#73d2de', '#ffbc42', '#218380', '#d81159', '#fe7f2d', '#3772ff']
        if logger is not None :
            self.logger = logger
            self.logger.LogTheLog("started Graphics")

    def start(self, titre):
        # initialize a pretty plot
        plt.figure(dpi=120)
        plt.grid('lightgrey')
        if titre != '':
            plt.title(titre)
        
        
    def plot2DTraj(self, trajs, titre=''):
        # plot a x, y traj in its plane
        # data : [ [x] [y] ]
        self.start(titre)
        for traj in trajs :
            plt.plot(traj[0], traj[1], self.colors[self.currentColor])
            self.currentColor = (self.currentColor+1)%len(self.colors)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
    
    def CompareNDdatas(self, dataset, datatype="speed", titre='', StyleAdapter=False, width=1):
        # plot a X or X,Y or X,Y,Z dataset evolving over time
        # enable StyleAdapter when datas are not very different from one another
        # data : [ [data1: [x][y][z]   ] [data2: [x][y][z]   ] ]
        self.start(titre)
        ndim = len(dataset[0])
        if datatype=="position":
            legends = [' - x', ' - y', ' - z'][:ndim]
        elif datatype=="speed":
            legends = [' - Vx', ' - Vy', ' - Vz'][:ndim]
        elif datatype=="acceleration":
            legends = [' - Ax', ' - Ay', ' - Az'][:ndim]
        if StyleAdapter:
            style = ['-', '--', '-+']
        else:
            style = ['-']
        k=0
        for data in dataset :
            for line in data :
                plt.plot(line, self.colors[self.currentColor], linestyle=style[k], linewidth=width)
                self.currentColor = (self.currentColor+1)%len(self.colors)
            k = (k+1)%len(style)
        plt.legend([str(k) + leg for k in range(len(dataset)) for leg in legends])
        plt.xlabel('sample (N)')
        plt.ylabel(datatype + 's')
        plt.show()
    





    







