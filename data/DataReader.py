import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('/home/nalbrecht/Bolt-Estimator/Bolt-robot---Estimator/tests')
from Graphics import Graphics
sys.path.append('/home/nalbrecht/Bolt-Estimator/Bolt-robot---Estimator/src/python')
from Bolt_Utils import Log


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
        

    def Load(self,   t_file=None, q_file=None, qd_file=None, x_file=None, v_file=None, a_file=None, 
                     w_file=None, tau_file=None, lcf_file=None, rcf_file=None, contact_file=None):
        self.logger.LogTheLog("DataReader : loading...")
        if t_file is not None :
            self.T = np.load(t_file)
            self.Printer(t_file, self.T)
        if q_file is not None :
            self.Q = np.load(q_file)
            self.Printer(q_file, self.Q)
        if qd_file is not None :
            self.Qd = np.load(qd_file)
            self.Printer(qd_file, self.Qd)
        if x_file is not None :
            self.X = np.load(x_file)
            self.Printer(x_file, self.X)
            self.groundH = self.X[0,self.RightFootID,2] + 0.002
            self.logger.LogTheLog("ground height is "+ str(self.groundH), "subinfo")
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
            print(self.LeftContact)
      
        self.SampleLength = len(self.T)
        self.logger.LogTheLog("DataReader : loaded data, number of samples = " + str(self.SampleLength))
    
    def AutoLoad(self, k):
        self.logger.LogTheLog("DataReader : Auto loading")
        kfile = str(k)
        #prefix = "/home/nalbrecht/Bolt-Estimator/bipedal-control/bipedal-control/"
        prefix = "/home/nalbrecht/Bolt-Estimator/bipedal-control/bipedal-control/Données cancer niels/" + kfile + "/"
        t_file = prefix + "T_array_" + kfile + ".npy"
        q_file = prefix + "Q_array_" + kfile + ".npy"
        qd_file = prefix + "Qd_array_" + kfile + ".npy"
        x_file = prefix + "X_array_" + kfile + ".npy"
        v_file = prefix + "V_array_" + kfile + ".npy"
        a_file = prefix + "A_array_" + kfile + ".npy"
        w_file = prefix + "W_array_" + kfile + ".npy"
        tau_file = prefix + "Tau_array_" + kfile + ".npy"
        rcf_file = prefix + "RCF_array_" + kfile + ".npy"
        lcf_file = prefix + "LCF_array_" + kfile + ".npy"
        leftcontact_file = prefix + "C_array_" + kfile + ".npy"

        self.Load(t_file=t_file,  q_file=q_file,  qd_file=qd_file,  x_file=x_file, 
                  v_file=v_file,  a_file=None,    w_file=w_file,    tau_file=tau_file,
                  lcf_file=lcf_file,  rcf_file=rcf_file,  contact_file=leftcontact_file)
        
    def Get(self):
        return self.Tau, self.LCF
    
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
        self.grapher.SetLegend(["speed of bolt's " + frameName], ndim=3)
        self.grapher.CompareNDdatas([self.V[:, frameID, :].transpose()], datatype='speed', title=frameName + ' speed')
        #self.grapher.end()
        
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




def main():
    # getting ready
    logger = Log(PrintOnFlight=True)
    Reader = DataReader(logger=logger)
    
    # loading .npy files in DataReader
    Reader.AutoLoad(2)
    # check for contact indexes
    Reader.Contact()
    Reader.PlotContact()
    # plot base position and speed
    # Reader.PlotBaseTrajectory()
    # Reader.PlotFeetTrajectory()
    Reader.PlotTorques('left')
    # Reader.PlotTorquesAndFeet()
    # Reader.PlotForces()
    # Reader.PlotTorqueForce()
    Reader.PlotLeftFootCorrelation()
    Reader.EndPlot()
    
    return Reader.Get()
    
    

Tau, LCF = main()
