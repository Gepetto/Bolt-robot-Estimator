import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R


import sys
sys.path.append('/home/nalbrecht/Bolt-Estimator/Bolt-robot---Estimator/tests')
from Graphics import Graphics
sys.path.append('/home/nalbrecht/Bolt-Estimator/Bolt-robot---Estimator/src/python')
from Bolt_Utils import Log

from DataImprover import improve
from TrajectoryGenerator import TrajectoryGenerator, Metal


class DataReader():
    def __init__(self, logger):
        self.T = np.zeros(1)
        self.Q = np.zeros(1)
        self.Qd = np.zeros(1)
        self.X = np.zeros(1)
        self.V = np.zeros(1)
        self.A = np.zeros(1)
        self.W = np.zeros(1)
        self.Tau = np.zeros(1)
        self.LCF = np.zeros(1)
        self.RCF = np.zeros(1)
        self.LeftContact = np.array([False])
        self.SampleLength = 1

        self.logger = logger
        self.logger.LogTheLog("started DataReader")
        self.grapher = Graphics(logger=self.logger)
        
        self.groundH = 0.031

        self.LeftFootID = 10
        self.RightFootID = 18
    
    def Printer(self, file, Z):
        self.logger.LogTheLog(file[-25:] + '  of shape  '+str(Z.shape), "subinfo")
        

    def Load(self,   t_file=None, q_file=None, qang_file=None, qd_file=None, x_file=None, theta_file=None, 
             v_file=None, a_file=None, w_file=None, tau_file=None, lcf_file=None, rcf_file=None, contact_file=None):
        self.logger.LogTheLog("DataReader : loading...")
        if t_file is not None :
            self.T = np.load(t_file)
            self.Printer(t_file, self.T)
            t0 = self.T[0]
            t1 = self.T[-1]
            self.logger.LogTheLog(f"time ranging from {t0} to {t1} ", "subinfo")
        if q_file is not None :
            self.Q = np.load(q_file)
            self.Printer(q_file, self.Q)
        if qang_file is not None :
            self.Qang = np.load(qang_file)
            self.Printer(qang_file, self.Qang)
        if qd_file is not None :
            self.Qd = np.load(qd_file)
            self.Printer(qd_file, self.Qd)
        if x_file is not None :
            self.X = np.load(x_file)
            self.Printer(x_file, self.X)
            self.groundH = self.X[0,self.RightFootID,2] + 0.002
            self.logger.LogTheLog("ground height is "+ str(self.groundH), "subinfo")
        if theta_file is not None :
            self.Theta = np.load(theta_file)
            self.Printer(theta_file, self.Theta)
        if v_file is not None :
            self.V = np.load(v_file)
            self.Printer(v_file, self.V)
        if a_file is not None :
            self.A = np.load(a_file)
            self.Printer(a_file, self.A)
        if w_file is not None :
            self.W = np.load(w_file)
            self.Printer(w_file, self.W)
        if tau_file is not None :
            self.Tau = np.load(tau_file)
            self.Printer(tau_file, self.Tau)
        if lcf_file is not None :
            self.LCF = np.load(lcf_file)
            self.Printer(lcf_file, self.LCF)
        if rcf_file is not None :
            self.RCF = np.load(rcf_file)
            self.Printer(rcf_file, self.RCF)
        if contact_file is not None :
            self.LeftContact = np.load(contact_file)[:3]
            self.Printer(contact_file, self.LeftContact)
            #print(self.LeftContact)
      
        self.SampleLength = len(self.T)
        self.dt = (t1-t0)/self.SampleLength
        self.logger.LogTheLog("DataReader : loaded data, number of samples = " + str(self.SampleLength) + ", dt = " + str(self.dt))
        

    
    def AutoLoad(self, k, acc='included', q_angular="not included"):
        self.logger.LogTheLog("DataReader : Auto loading")
        kfile = str(k)
        #prefix = "/home/nalbrecht/Bolt-Estimator/bipedal-control/bipedal-control/"
        #prefix = "/home/nalbrecht/Bolt-Estimator/bipedal-control/bipedal-control/Données cancer niels/" + kfile + "/"
        prefix = "/home/nalbrecht/Bolt-Estimator/Bolt-robot---Estimator/data/" + kfile + "/"
        self.prefix=prefix
        
        self.t_file = prefix + "T_array_" + kfile + ".npy"
        self.q_file = prefix + "Q_array_" + kfile + ".npy"
        self.qd_file = prefix + "Qd_array_" + kfile + ".npy"
        self.x_file = prefix + "X_array_" + kfile + ".npy"
        self.theta_file = prefix + "Theta_array_" + kfile + ".npy"
        self.v_file = prefix + "V_array_" + kfile + ".npy"
        if acc != 'included':
            self.a_file = None
        else :
            self.a_file = prefix + "A_array_" + kfile + ".npy"
        if q_angular != 'included':
            self.qang_file = None
        else :
            self.qang_file = prefix + "Qang_array_" + kfile + ".npy"
        self.w_file = prefix + "W_array_" + kfile + ".npy"
        self.tau_file = prefix + "Tau_array_" + kfile + ".npy"
        self.rcf_file = prefix + "RCF_array_" + kfile + ".npy"
        self.lcf_file = prefix + "LCF_array_" + kfile + ".npy"
        self.leftcontact_file = prefix + "C_array_" + kfile + ".npy"

        self.Load(t_file=self.t_file,  q_file=self.q_file,  qang_file=self.qang_file, qd_file=self.qd_file,  
                  x_file=self.x_file, theta_file=self.theta_file,
                  v_file=self.v_file,  a_file=self.a_file,    w_file=self.w_file,    tau_file=self.tau_file,
                  lcf_file=self.lcf_file,  rcf_file=self.rcf_file,  contact_file=self.leftcontact_file)
        
        self.filenames = [self.t_file,    self.q_file,   self.qang_file, self.qd_file,   self.x_file, 
                          self.theta_file,self.v_file,    self.a_file,   
                          self.w_file,    self.tau_file,
                          self.lcf_file,  self.rcf_file]
        
    
    def AutoImproveData(self, k, N=1000):
        self.logger.LogTheLog("DataReader : Improving data resolution to N="+str(N), "info")
        j, = self.T.shape
        if N ==j:
            self.logger.LogTheLog("DataReader : data seems to be of the right size already", "warn")
        for filename in self.filenames :
            if filename is not None:
                improve(N, filename, filename, talk=False)
                self.logger.LogTheLog("improved ..."+ filename[-23:], "subinfo")
            else :
                self.logger.LogTheLog("DataReader : nonexistent filename in AutoImproveData, skipped it", "warn")

        self.AutoLoad(k, acc="included", q_angular="not included")
        
    
    
    def AddAcceleration(self, k):
        """ create an acceleration data on every frame by deriving speed, and save it"""
        self.logger.LogTheLog("DataReader : Adding acceleration to dataset", "info")
        # get dimensions
        N, nframe, _ = self.X.shape
        Acc = np.zeros(self.V.shape)
        generator = TrajectoryGenerator(logger=self.logger)
        
        for FrameID in range(nframe):
            # prepare data
            Traj = self.X[:, FrameID, :].copy()
            Speed = self.V[:, FrameID, :].copy()        
            # load data in generator
            generator.Generate("custom", N=1, T=1, NoiseLevel=10, Drift=20, amplitude=10, 
                               avgfreq=0.5, relative=True, traj=Traj, speed=Speed, smooth=False)
            
            # compute speed to check consistency
            s = generator.MakeSpeedFromTrajectory(Traj, 1e-3)
            # computing acceleration
            a = generator.MakeAccelerationFromSpeed(Speed, 1e-3)
            # saving acceleration to X and V shape    
            Acc[:, FrameID, :] = a
        np.save(self.prefix + "A_array_" + str(k), Acc)
        self.grapher.SetLegend(["True", "computed"], ndim=3)
        self.grapher.CompareNDdatas([Speed.transpose(), [s]], datatype="speed of bolt's base", mitigate=[0])

    
    def AddAngularQ(self, k):
        """ the old Q from simulation was a rotation matrix. We need the encoder angle."""
        self.logger.LogTheLog("DataReader : Adding angular q to dataset as Qang", "info")
        # get dimensions
        N, nq, _, _ = self.Q.shape
        Qang = np.zeros((N, nq))
        generator = TrajectoryGenerator(logger=self.logger)
        
        for Qid in range(nq):
            # prepare data
            RotMatrixes = self.Q[:, Qid, :, :].copy()
            Rotation = R.from_matrix(RotMatrixes)
            # rotation to an angle (quaternion is in scalar-last format)
            angle = np.arccos(Rotation.as_quat()[:, -1]) * 2
            #print(RotMatrixes)
            #print(angle)
            Qang[:, Qid] = angle
            
        np.save(self.prefix + "Qang_array_" + str(k), Qang)
        self.grapher.SetLegend(["left hip", "left knee"], ndim=1)
        self.grapher.CompareNDdatas([[Qang[:, 3]], [Qang[:, 4]]], title="angle from bolt articulation")

    
    
    def __AdaptDimQlike(self, Q_like):
        SampleLength, _, n = Q_like.shape 
        NewQ_like = np.zeros((SampleLength, n))
        NewQ_like[:, :] = Q_like[:, 0, :]
        return NewQ_like
    
    
    def AdaptDimQQd(self, k):
        """ adapt dimension of loaded Q and Qd arrays and save them"""
        self.logger.LogTheLog("DataReader : Adapting shape of Q and Qd", "info")
        Q = self.__AdaptDimQlike(self.Q)
        Qd = self.__AdaptDimQlike(self.Qd)
        np.save(self.prefix + "Q_array_" + str(k), Q)
        np.save(self.prefix + "Qd_array_" + str(k), Qd)
    
    
    def Get(self, data):
        if data=="x":
            return self.X
        elif data=="v":
            return self.V
        elif data=="a":
            return self.A
        elif data=="q":
            return self.Q
        elif data=="qd":
            return self.Qd
        elif data=="theta":
            return self.Theta
        elif data=="omega":
            return self.W
        elif data=="tau":
            return self.Tau
        elif data=="rcf":
            return self.RCF
        elif data=="lcf":
            return self.LCF
        
        
    
    def EndPlot(self):
        self.grapher.end()
    
    def Contact(self):
        # use position of feet to determin which one is touching the ground
        # TOUCHY
        self.Rcontactindex = []
        for k in range(len(self.X[:, self.RightFootID, 2])):
            z = self.X[k, self.RightFootID, 2]
            if z<self.groundH and  z> 0.0: 
                self.Rcontactindex.append(k)
        self.Lcontactindex = []
        for k in range(len(self.X[:, self.LeftFootID, 2])):
            z = self.X[k, self.LeftFootID, 2]
            if z<self.groundH and  z> 0.0: 
                self.Lcontactindex.append(k)
        self.RContact = np.where( (self.X[:, self.RightFootID, 2]<self.groundH), 0, 1).reshape((1, -1))
        self.LContact = np.where( (self.X[:, self.LeftFootID, 2]<self.groundH), 0, 1).reshape((1, -1))
    
    
    def PlotContact(self):
        self.grapher.SetLegend(['position'], ndim=3)
        self.grapher.CompareNDdatas([self.X[:, self.RightFootID, :].transpose()], datatype='position', title='right foot position')#, selectmarker=self.Rcontactindex)
        self.grapher.CompareNDdatas([self.X[:, self.LeftFootID, :].transpose()], datatype='position', title='left foot position')#, selectmarker=self.Lcontactindex)
        self.grapher.SetLegend(['contact R', 'contact L'], ndim=1)
        self.grapher.CompareNDdatas([self.RContact, self.LContact], datatype='position', width=1.5, StyleAdapter=True, title='ground contacts')
        #self.grapher.end()
    
    def PlotBaseTrajectory(self):
        self.PlotTrajectory(1, "base")
    def PlotFeetTrajectory(self):
        self.PlotTrajectory(self.LeftFootID, "left foot")  
        self.PlotTrajectory(self.RightFootID, "right foot")  
    
    def PlotTrajectory(self, frameID=1, frameName="base"):
        self.grapher.SetLegend(["position of bolt's " + frameName], ndim=3)
        self.grapher.CompareNDdatas([self.X[:, frameID, :].transpose()], datatype='position', title= frameName + ' Position')#' with Right foot contact times highlighted', selectmarker=self.Rcontactindex)
    
    def PlotSpeed(self, frameID=1, frameName="base"):
        self.grapher.SetLegend(["speed of bolt's " + frameName], ndim=3)
        self.grapher.CompareNDdatas([self.V[:, frameID, :].transpose()], datatype='speed', title=frameName + ' speed')
        #self.grapher.end()
    
    def PlotAcceleration(self, frameID=1, frameName="base"):
        self.grapher.SetLegend(["acceleration of bolt's " + frameName], ndim=3)
        self.grapher.CompareNDdatas([self.A[:, frameID, :].transpose()], datatype='acceleration', title= frameName + ' acceleration')
  
    def PlotQ(self):
        self.grapher.SetLegend(["left joints", "right joints"], ndim=3)
        self.grapher.CompareNDdatas([self.Q[:, -6:-3].transpose(), self.Q[:, -3:].transpose()], datatype='radian', title= 'joints angle')
  
        
        
        
        
    def PlotTorqueJoint(self, jointID=3):
        self.grapher.SetLegend(["Torques, left leg"], ndim=1)
        Tau = self.Tau[:, jointID:jointID+1].copy()
        self.grapher.CompareNDdatas([Tau.T], datatype='torque', title='torques in joint '+ str(jointID))#, selectmarker=self.Lcontactindex[0:25])

    def PlotTorques(self, side='left'):
        if side=='left':
            self.grapher.SetLegend(["Torques, left leg"], ndim=3)
            self.grapher.CompareNDdatas([self.Tau[:, 0:3].transpose()], datatype='torque', title='left torques', selectmarker=self.Lcontactindex[0:25])
        elif side=='right' :
            self.grapher.SetLegend(["Torques, right leg"], ndim=3)
            self.grapher.CompareNDdatas([self.Tau[:, 3:].transpose()], datatype='torque', title='right torques')
        elif side=='both':
            self.grapher.SetLegend(["Torques, left leg", "Torques, right leg"], ndim=3)
            self.grapher.CompareNDdatas([self.Tau[:, :3].transpose(), self.Tau[:, 3:].transpose()], datatype='torque', title='both legs torques')
        #self.grapher.end()
    
    def PlotTorquesAndFeet(self):
        self.grapher.SetLegend(["Right foot position", "Torques, right leg"], ndim=3)
        self.grapher.CompareNDdatas([self.X[:,self.RightFootID, :].transpose(), self.Tau[:self.SampleLength, 3:].transpose()], datatype='torques, pos', title='right torques and traj', ignore=[0,1])
        
        self.grapher.SetLegend(["Left foot position", "Torques, left leg"], ndim=3)
        self.grapher.CompareNDdatas([self.X[:,self.LeftFootID, :].transpose(), self.Tau[:self.SampleLength, 0:3].transpose()], datatype='torques, pos', title='left torques and traj', ignore=[0,1])
        
    def PlotForces(self):
        self.grapher.SetLegend(["Left foot contact Forces"], ndim=3)
        self.grapher.CompareNDdatas([self.LCF.transpose()], datatype='force', title='left contact force')
        self.grapher.SetLegend(["Right foot contact Forces"], ndim=3)
        self.grapher.CompareNDdatas([self.RCF.transpose()], datatype='force', title='right contact force')
        self.grapher.SetLegend(["Left Z force", "Right Z force", "Total Z force"], ndim=1)
        self.grapher.CompareNDdatas([self.LCF[:,2].reshape(1, -1), self.RCF[:,2].reshape(1, -1), self.LCF[:,2].reshape(1, -1)+self.RCF[:,2].reshape(1, -1)], datatype='force', title='left and right contact forces, vertical', mitigate=[2])
        
    def PlotTorqueForce(self):
        # PARCE QUE LE BRAS DE LEVIER ENTRE LE GENOUX ET LE COUDE EST DE 1/8e DE m
        self.grapher.SetLegend(["Left Z force", "Left Torque n°2 x8"], ndim=1)
        self.grapher.CompareNDdatas([self.LCF[:,2].reshape(1, -1), self.Tau[:, 2].reshape(1, -1)*8], datatype='torques, force', title='left contact forces and left torque comparison', StyleAdapter=True)
        
    def PlotLeftFootCorrelation(self):
        self.grapher.SetLegend(["Left Z force x1", "Left Torque n°2 x8", "Left foot pos x50", "Right foot pos x50"], ndim=1)
        self.grapher.CompareNDdatas([self.LCF[:,2].reshape(1, -1), self.Tau[:, 2].reshape(1, -1)*8, self.X[:,self.LeftFootID, 2].reshape(1, -1)*50, self.X[:,self.RightFootID, 2].reshape(1, -1)*50], datatype='torques, force, position', title='left contact force, torque and pos comparison (dimensionless)', StyleAdapter=True)

    def SuperPlotLeftFootCorrelation(self):
        self.grapher.SetLegend(["Left Foot contact force, z", "Left Knee Torque", "Left foot height", "Contact Probability using Trigger", "Contact Probability using Sigmoid"], ndim=1)
        #y = self.__SigmoidDiscriminator(self.Tau[:, 2], center=0.8, stiffness=6)
        y = self.__TriggerDiscrimination(self.LCF[:, 2], 4, 10)
        z = self.__SigmoidDiscriminator(self.LCF[:, 2], center= 4, stiffness=6)
        self.grapher.CompareNDdatas([self.LCF[:,2].reshape(1, -1)/6, self.Tau[:, 2].reshape(1, -1), self.X[:,self.LeftFootID, 2].reshape(1, -1)*5, y.reshape(1, -1),  z.reshape(1, -1)], datatype='torques, force, position', title='left contact force, torque and pos comparison (dimensionless)', StyleAdapter=True)

        
    def __SigmoidDiscriminator(self, x, center, stiffness=5):
        """Bring x data between 0 and 1, such that P(x=center)=0.5. The greater stiffness, the greater dP/dx (esp. around x=center)"""
        b0 = stiffness
        b = b0/center
        return 1/ (1 + np.exp(-b*x + b0))
    
    def __TriggerDiscrimination(self, x, LowerThresold=4, UpperThresold=10):
        y = np.zeros(len(x))
        for i in range(1, len(y)):
            if x[i] < LowerThresold :
                y[i] = 0
            elif x[i] > UpperThresold :
                y[i] = 1
            else :
                y[i] = y[i-1]
        return y


    
    



def main(k=6):
    # getting ready
    logger = Log(PrintOnFlight=True)
    Reader = DataReader(logger=logger)

    
    
    # in case data is straight out of a simulation, improve sampling and add acceleration
    # load without acceleration
    """
    Reader.AutoLoad(k, acc='not included', q_angular="not included")
    
    Reader.AddAcceleration(k)
    
    Reader.AdaptDimQQd(k)
    
    # improve resolution of .npy files (to execute only once per set of files)
    Reader.AutoLoad(k, acc="included", q_angular="not included")
    Reader.AutoImproveData(k, 5000)
    """
    

    # loading .npy files in DataReader
    Reader.AutoLoad(k, acc='included', q_angular="not included")
    
    
    # check for contact indexes
    Reader.Contact()
    # Reader.PlotContact()
    
    # plot base position and speed
    # Reader.PlotBaseTrajectory()
    # Reader.PlotSpeed(1, "base")
    # Reader.PlotFeetTrajectory()
    # Reader.PlotAcceleration(1, "base")
    # Reader.PlotAcceleration(4, "leg")
    
    # plotting torques and forces
    Reader.PlotTorques('left')
    # Reader.PlotTorqueJoint(0)
    # Reader.PlotTorquesAndFeet()
    # Reader.PlotForces()
    # Reader.PlotTorqueForce()
    # Reader.SuperPlotLeftFootCorrelation()
    Reader.PlotQ()
    
    
    Reader.EndPlot()
    
    return Reader.Get("q")

    
    
if __name__ == "__main__":
    ZZ = main()
    #A = Q[:, 3, :, :]
