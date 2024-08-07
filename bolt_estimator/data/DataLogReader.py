import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R


from bolt_estimator.utils.Graphics import Graphics
from bolt_estimator.utils.Utils import Log

from bolt_estimator.data.DataImprover import improve
from bolt_estimator.utils.TrajectoryGenerator import TrajectoryGenerator, Metal


class DataReader():
    def __init__(self, logger, start):
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
        self.s = start

        self.logger = logger
        self.logger.LogTheLog("started DataReader")
        self.grapher = Graphics(logger=self.logger)
        
        self.groundH = 0.031

        self.LeftFootID = 10
        self.RightFootID = 18

        # data get useless after fall
        self.fall = -1
    
    def Printer(self, file, Z):
        self.logger.LogTheLog(file[-25:] + '  of shape  '+str(Z.shape), "subinfo")
    
    def LoadAndPlotLog(self, file, ndim, title, toprint=False):
        #prefix = "/home/nalbrecht/Bolt-Estimator/Bolt-robot---Estimator/data/simu/"
        #prefix = "/home/nalbrecht/Bolt-Estimator/bipedal-control/"
        prefix = "./"
        filename = prefix + file
        Y = np.load(filename)
        self.logger.LogTheLog("loaded file " + filename+ " in " + prefix, "info")
        self.logger.LogTheLog("data shape : " + str(Y.shape), "info")
        self.grapher.SetLegend(["logs"], ndim=ndim)
        if toprint :
            print(Y)
        self.grapher.CompareNDdatas([Y], datatype='logs', title=title, StyleAdapter=True)
    
    def LoadAndPlotDualLogs(self, file1, file2, ndim, title, toprint=False, transpose=False):
        #prefix = "/home/nalbrecht/Bolt-Estimator/Bolt-robot---Estimator/data/simu/"
        #prefix = "/home/nalbrecht/Bolt-Estimator/bipedal-control/"
        prefix = "./"
        self.prefix = prefix
        filename1 = prefix + file1
        filename2 = prefix + file2
        if transpose : 
            Y = np.load(filename1).T
        else :
            Y = np.load(filename1)
        Z = np.load(filename2).T
        
        # d, n1 = Y.shape
        # np.concatenate((np.zeros((d, 30)), Y), axis=1)
        _, n1 = Y.shape
        _, n2 = Z.shape
        n = min(n1, n2-30)
        n = min(n, self.fall)
        print(n)
        print(self.fall)
        self.logger.LogTheLog("2 loaded files " + filename1 + " in " + prefix, "info")
        self.logger.LogTheLog("data shapes : " + str(Y.shape) + " and " + str(Z.shape), "info")
        self.grapher.SetLegend(["logs", "true"], ndim=ndim)
        if toprint :
            print(Y)
        self.grapher.CompareNDdatas([Y[:, :n], Z[:, self.s:n+self.s]], datatype='logs', title=title, StyleAdapter=True)
        self.grapher.SetLegend(["error"], ndim=ndim)
        self.grapher.CompareNDdatas([Y[:, :n]-Z[:, self.s:n+self.s]], datatype='error', title=title + ' (error)', StyleAdapter=True)
        

    def Load(self,   t_file=None, q_file=None, qd_file=None, x_file=None, theta_file=None, theta_euler_file=None, 
             v_file=None, a_file=None, ag_file=None, w_file=None, tau_file=None, lcf_file=None, rcf_file=None, 
             contact_file=None, base_pos_file=None):
        self.logger.LogTheLog("DataReader : loading...")
        self.dt = 1e-3
        if t_file is not None :
            self.T = np.load(t_file)
            self.Printer(t_file, self.T)
            t0 = self.T[0]
            t1 = self.T[-1]
            self.dt = (t1-t0)/self.SampleLength
            self.logger.LogTheLog(f"time ranging from {t0} to {t1} ", "subinfo")
        if q_file is not None :
            self.Q = np.load(q_file)
            self.Printer(q_file, self.Q)
        if base_pos_file is not None :
            self.X = np.load(base_pos_file)
            self.Printer(base_pos_file, self.X)

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
        if theta_euler_file is not None :
            self.ThetaEuler = np.load(theta_euler_file)
            self.Printer(theta_euler_file, self.ThetaEuler)
        if v_file is not None :
            self.V = np.load(v_file)
            self.Printer(v_file, self.V)
        if a_file is not None :
            self.A = np.load(a_file)
            self.Printer(a_file, self.A)
        if ag_file is not None :
            self.Ag = np.load(ag_file)
            self.Printer(ag_file, self.Ag)
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
        
        self.logger.LogTheLog("DataReader : loaded data, number of samples = " + str(self.SampleLength) + ", dt = " + str(self.dt))
        
    
    

    
    
    def AutoLoadSimulatedData(self, style="standing"):
        self.logger.LogTheLog("DataReader : Auto loading simulated data")
        #prefix = "/home/nalbrecht/Bolt-Estimator/Bolt-robot-Estimator/data/" + kfile + "/"
        #prefix = "/home/niels/Supaéro/Stage 2A/Gepetto/Code/Bolt-robot-Estimator/data/Data_" + style + "/true_"
        prefix = "/data/" + kfile + "/"
        self.prefix=prefix
        
        self.q_file = prefix + "q_logs.npy"
        self.qd_file = prefix + "qdot_logs.npy"
        self.x_file = prefix + "pos_logs.npy"
        self.theta_euler_file = prefix + "theta_logs.npy"
        self.v_file = prefix + "speed_logs.npy"
        self.a_file = prefix + "a_logs.npy"
        self.ag_file = prefix + "ag_logs.npy"

        self.w_file = prefix + "omega_logs.npy"
        self.tau_file = prefix + "tau_logs.npy"


        self.Load(t_file=None,           q_file=self.q_file,         qd_file=self.qd_file,  
                  x_file=None,    theta_file=None,           theta_euler_file=self.theta_euler_file,
                  v_file=self.v_file,    a_file=self.a_file,  ag_file=self.ag_file,       w_file=self.w_file, tau_file=self.tau_file,
                  lcf_file=None,  rcf_file=None,   contact_file=None, base_pos_file=self.x_file)
        
        self.filenames = [self.q_file,           self.qd_file,   self.x_file, 
                          self.theta_euler_file, self.v_file,    self.a_file,  self.ag_file, 
                          self.w_file,     self.tau_file
                          ]
        
        
    
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

        self.AutoLoad(k, acc="included", theta_euler="included")
        
    
    
    
    
    def Get(self, data):
        if data=="x":
            return self.X
        elif data=="v":
            return self.V
        elif data=="a":
            return self.A
        elif data=="ag":
            return self.Ag
        elif data=="q":
            return self.Q
        elif data=="qd":
            return self.Qd
        elif data=="theta":
            return self.Theta
        elif data=="theta_euler":
            return self.ThetaEuler
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
    
  


    
    
def LogLoading():
    # getting ready
    START = 25
    logger = Log(PrintOnFlight=True)
    Reader = DataReader(logger=logger, start=START)
    #Reader.LoadAndPlotLog("theta_out.npy", 4, "theta")
    #Reader.LoadAndPlotLog("c_out.npy", 3, "c out")
    Reader.fall = 4500-1
    #Reader.LoadAndPlotDualLogs("g_out.npy", "true_g_logs.npy", 3, "g out and true")
    #Reader.LoadAndPlotDualLogs("g_tilt.npy", "true_g_logs.npy", 3, "g tilt and true")
    # Reader.LoadAndPlotDualLogs("v_out.npy", "true_speed_logs.npy", 3, "v out and true", False)
    # Reader.LoadAndPlotDualLogs("v_tilt.npy", "true_speed_logs.npy", 3, "v tilt and true", True)
    Reader.LoadAndPlotDualLogs("c_switch.npy", "true_pos_logs.npy", 3, "pos from switch and true")
    # Reader.LoadAndPlotDualLogs("c_out.npy", "true_pos_logs.npy", 3, "pos out and true")
    # Reader.LoadAndPlotDualLogs("theta_out.npy", "true_theta_logs.npy", 4, "theta out and true")
    # Reader.LoadAndPlotDualLogs("theta_tilt.npy", "true_theta_logs.npy", 4, "theta tilt and true")
    #Reader.LoadAndPlotLog("a.npy", 3, "a")
    #Reader.LoadAndPlotLog("q.npy", 6, "q")
    #Reader.LoadAndPlotLog("qdot.npy", 6, "qdot")
    #Reader.LoadAndPlotLog("Contact.npy", 2, "contact out")
    #Reader.LoadAndPlotDualLogs("w.npy", "true_omega_logs.npy", 3, "omega out and true")

    
    # Reader.LoadAndPlotDualLogs("com_pos_logs.npy", "true_pos_logs.npy", 3, "pos command and true", transpose=True)
    # Reader.LoadAndPlotDualLogs("com_speed_logs.npy", "true_speed_logs.npy", 3, "v command and true", transpose=True)
    # Reader.LoadAndPlotDualLogs("com_omega_logs.npy", "true_omega_logs.npy", 3, "omega command and true", transpose=True)
    
    # convert to euler
    theta_true = R.from_quat(np.load(Reader.prefix + "true_theta_logs.npy")).as_euler('xyz').T
    theta_com = R.from_quat(np.load(Reader.prefix + "com_theta_logs.npy")).as_euler('xyz').T
    theta_out = R.from_quat(np.load(Reader.prefix + "theta_out.npy").T).as_euler('xyz').T

    _, n1 = theta_true.shape
    _, n2 = theta_com.shape
    n = min(min(n1, n2), Reader.fall)
    
    Reader.grapher.SetLegend(["Theta true", "Theta command"], ndim=3)
    Reader.grapher.CompareNDdatas([theta_true[:, :n], theta_com[:, :n]], datatype='radian', title='Attitude as Euler', mitigate=[0])
    Reader.grapher.SetLegend(["Error on theta"], ndim=3)
    Reader.grapher.CompareNDdatas([theta_true[:, :n]-theta_com[:, :n]], datatype='radian', title='Error on Attitude com as Euler')


    Reader.grapher.SetLegend(["Theta true", "Theta out"], ndim=3)
    Reader.grapher.CompareNDdatas([theta_true[:, START:n+START], theta_out[:, :n]], datatype='radian', title='Attitude as Euler', mitigate=[0])
    Reader.grapher.SetLegend(["Error on theta"], ndim=3)
    Reader.grapher.CompareNDdatas([theta_true[:, START:n+START]-theta_out[:, :n]], datatype='radian', title='Error on Attitude out as Euler')
   

    Reader.EndPlot()


def main_simu(style="walking"):
    # getting ready
    logger = Log(PrintOnFlight=True)
    Reader = DataReader(logger=logger)
    
    # loading
    Reader.AutoLoadSimulatedData(style)




    
if __name__ == "__main__":
    #main_simu()
    LogLoading()
