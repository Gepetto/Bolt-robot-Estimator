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
        self.SampleLength = 1

        self.logger = logger
        self.logger.LogTheLog("started DataReader")
        self.grapher = Graphics(logger=self.logger)
        
        self.groundH = 0.031
    
    def Printer(self, file, Z):
        self.logger.LogTheLog(file[-25:] + '  of shape  '+str(Z.shape), "subinfo")
        

    def Load(self, t_file=None, q_file=None, qd_file=None, x_file=None, v_file=None, a_file=None, w_file=None, tau_file=None):
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
            self.groundH = self.X[0,18,2] + 0.002
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
        
            
            
        self.SampleLength = len(self.T)
        self.logger.LogTheLog("DataReader : loaded data, number of samples = " + str(self.SampleLength))
    
    def AutoLoad(self, k):
        self.logger.LogTheLog("DataReader : Auto loading")
        kfile = str(k)
        prefix = "/home/nalbrecht/Bolt-Estimator/bipedal-control/bipedal-control/"
        t_file = prefix + "T_array_" + kfile + ".npy"
        q_file = prefix + "Q_array_" + kfile + ".npy"
        qd_file = prefix + "Qd_array_" + kfile + ".npy"
        x_file = prefix + "X_array_" + kfile + ".npy"
        v_file = prefix + "V_array_" + kfile + ".npy"
        a_file = prefix + "A_array_" + kfile + ".npy"
        w_file = prefix + "W_array_" + kfile + ".npy"
        tau_file = prefix + "Tau_array_" + kfile + ".npy"

        self.Load(t_file=t_file,  q_file=q_file,  qd_file=qd_file,  x_file=x_file, 
                  v_file=v_file,  a_file=None,    w_file=w_file,    tau_file=tau_file)
    
    def Contact(self):
        # use position of feet to determin which one is touching the ground
        # TOUCHY
        self.Rcontactindex = []
        for k in range(len(self.X[:, 18, 2])):
            z = self.X[k, 18, 2]
            if z<self.groundH and  z> 0.0: 
                self.Rcontactindex.append(k)
        self.Lcontactindex = []
        for k in range(len(self.X[:, 10, 2])):
            z = self.X[k, 10, 2]
            if z<self.groundH and  z> 0.0: 
                self.Lcontactindex.append(k)
        self.RContact = np.where( (self.X[:, 18, 2]<self.groundH), 0, 1).reshape((1, -1))
        self.LContact = np.where( (self.X[:, 10, 2]<self.groundH), 0, 1).reshape((1, -1))
    
    def PlotContact(self):
        self.grapher.SetLegend(['position'], ndim=3)
        self.grapher.CompareNDdatas([self.X[:, 18, :].transpose()], datatype='position')#, selectmarker=self.Rcontactindex)
        self.grapher.CompareNDdatas([self.X[:, 10, :].transpose()], datatype='position')#, selectmarker=self.Lcontactindex)
        self.grapher.SetLegend(['contact R', 'contact L'], ndim=1)
        self.grapher.CompareNDdatas([self.RContact, self.LContact], datatype='position', width=1.5, StyleAdapter=True)
        self.grapher.end()
    
    def PlotBaseTrajectory(self):
        self.PlotTrajectory(1, "base")
    def PlotFeetTrajectory(self):
        self.PlotTrajectory(10, "left foot")  
        self.PlotTrajectory(18, "right foot")  
    
    def PlotTrajectory(self, frameID=1, frameName="base"):
        self.grapher.SetLegend(["position of bolt's " + frameName], ndim=3)
        self.grapher.CompareNDdatas([self.X[:, frameID, :].transpose()], datatype='position', title= frameName + ' Position')#' with Right foot contact times highlighted', selectmarker=self.Rcontactindex)
        self.grapher.SetLegend(["speed of bolt's " + frameName], ndim=3)
        self.grapher.CompareNDdatas([self.V[:, frameID, :].transpose()], datatype='speed')
        self.grapher.end()
        
    def PlotTorques(self, k):
        self.grapher.SetLegend(["Torques"], ndim=3)
        self.grapher.CompareNDdatas([self.Tau[:, 5, :].transpose()], datatype='torques')
        self.grapher.end()
        





def main():
    # getting ready
    logger = Log(PrintOnFlight=True)
    Reader = DataReader(logger=logger)
    
    # loading .npy files in DataReader
    Reader.AutoLoad(0)
    # check for contact indexes
    Reader.Contact()
    Reader.PlotContact()
    # plot base position and speed
    #Reader.PlotBaseTrajectory()
    Reader.PlotFeetTrajectory()
    Reader.PlotTorques(4)
    
    return Reader.Q, Reader.Tau
    

Q, Tau = main()
