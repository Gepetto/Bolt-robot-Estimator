import numpy as np
import matplotlib.pyplot as plt

'''
Code that :
    simulates a robot's trajectory
    mesures it, adding fake measure noise
    tries to derive the robot's trajectory based on the noisy measurement

'''


# time increment, s. Just to change speeds accordingly
dt = 0.1



class Robot:
    def __init__(self):
        self.trajectory = np.zeros((3,1))
        self.speed = np.zeros((3,1))
    
    

        
    def TrajectoryGenerator(self, TypeOfTraj, N, traj=np.zeros((1, 1))):
        if TypeOfTraj =='linear':
            self.TrajectoryLinear(N)
        elif TypeOfTraj == 'custom':
            self.trajectory = traj
    
    def TrajectoryLinear(self, N):
        # chose an angle
        alpha = np.random.random() *np.pi/2
        print('angle chosen (°) ', alpha*180/np.pi)
        self.trajectory = np.zeros((3,N))
        def PositionLaw(x):
            # so that speed is no constant
            return np.sin(x/N/2*np.pi)
        self.trajectory[0,:] = np.cos(alpha)*PositionLaw(np.linspace(0,N, num=N))
        self.trajectory[1,:] = np.sin(alpha)*PositionLaw(np.linspace(0,N, num=N))
        
        
    def MakeSpeedFromTrajectory(self, traj):
        # derives the speed
        D, N= np.shape(traj)
        self.speed = np.zeros((3,N))
        self.speed[:, 1:] = (traj[:, 1:] - traj[:, 0:-1])/dt
        
    def MakeAccelereationFromTrajectory(self, traj):
        # derives the speed
        D, N= np.shape(traj)
        self.speed = np.zeros((3,N))
        self.speed[:, 1:] = (traj[:, 1:] - traj[:, 0:-1])/dt
        
    def update(self):
        self.MakeSpeedFromTrajectory(self.trajectory)


# a class to measure noisily 
class IMU:
    def __init__(self, traj, noiseLevel=0.1):
        self.noiseLevel = noiseLevel
        self.D, self.N = np.shape(traj)
        self.realData = traj.copy()
        self.noisyData = np.zeros((self.D, self.N))
        
    def makeNoise(self):
        self.noisyData = self.realData + np.random.normal(loc=0.0, scale=self.noiseLevel, size=(self.D,self.N))
    
    def measurePos(self):
        self.makeNoise()
        return self.noisyData
    
    def measureSpeed(self):
        self.MakeSpeedFromTrajectory()
        return self.noisySpeed
        
    
    def MakeSpeedFromTrajectory(self):
        self.noisySpeed = np.zeros((3,self.N))
        self.noisySpeed[:, 1:] = (self.noisyData[:, 1:] - self.noisyData[:, 0:-1])/dt
    
    
# a class for plotting the computed trajectories and comparing them
class graphics:
    def __init__(self):
        self.currentColor = 0
        self.colors = ['#8f2d56', '#73d2de', '#ffbc42', '#218380', '#d81159', '#fe7f2d', '#3772ff']
    def start(self, titre):
        plt.clf()
        plt.figure(dpi=120)
        plt.grid('lightgrey')
        if titre != '':
            plt.title(titre)
        
        
    def plot2DTraj(self, trajs, titre=''):
        self.start(titre)
        for traj in trajs :
            plt.plot(traj[0], traj[1], self.colors[self.currentColor])
            self.currentColor = (self.currentColor+1)%len(self.colors)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
    
    def plotSpeedXYZ(self, speeds, titre=''):
        self.start(titre)
        for speed in speeds :
            for line in speed :
                plt.plot(line, self.colors[self.currentColor])
                self.currentColor = (self.currentColor+1)%len(self.colors)
        plt.legend([str(k) + spud for k in range(len(speeds)) for spud in [' - Vx', ' - Vy', ' - Vz']])
        plt.xlabel('Number of iteration')
        plt.ylabel('speeds')
        plt.show()




# a class to derive the position based on the IMU measures
class Estimateur:
    def __init__(self, func):
        # func must be able to take an array on argument
        self.DynamicLaw = func
    
 
    
    







def main():
    bolt = Robot()
    oscar = graphics()
    bolt.TrajectoryGenerator('linear', 100)#, np.array([[0, 1, 2, 3, 4, 6, 8, 10], 
                                                    #[0, 0, 1, 2, 2, 1, 1, 0],
                                                    #[0, 0, 0, 0, 0, 0, 0, 0]]))
    bolt.update()
    '''
    print('position réelle')
    print(bolt.trajectory)
    print('vitesse réelle')
    print(bolt.speed)
    '''
    imu = IMU(bolt.trajectory, 0.001)
    
    '''
    print('position bruitée')
    print(imu.measurePos())
    
    print('vitesse bruitée')
    print(imu.measureSpeed())
    '''
    
    oscar.plot2DTraj([bolt.trajectory, imu.measurePos()])
    oscar.plotSpeedXYZ([bolt.speed, imu.measureSpeed()])



main()