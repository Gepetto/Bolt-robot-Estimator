import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R


from bolt_estimator.utils.Graphics import Graphics
from bolt_estimator.utils.Utils import Log

from bolt_estimator.data.DataImprover import improve
from bolt_estimator.utils.TrajectoryGenerator import TrajectoryGenerator, Metal


class DataReader():
    def __init__(self):
        self.logger = Log("EstimatorLogReader")
        self.grapher = Graphics(logger=self.logger)
        pass
    def LoadLogs(self, prefix=None, out=True, tilt=False, contact=True, imu=False, kin=False) -> None :
        # load logs
        if prefix is None :
            prefix = "data/"
        else :
            prefix=prefix
        # np.save(prefix + "blub", np.zeros(3))
        # print("done")

        
        if out :
            # base velocitie & co, post-filtering logs
            self.log_v_out = np.load(prefix + "estimator_logs_v_out.npy")
            self.log_w_out = np.load(prefix + "estimator_logs_w_out.npy")
            self.log_a_out = np.load(prefix + "estimator_logs_a_out.npy")
            self.log_theta_out = np.load(prefix + "estimator_logs_theta_out.npy")
            self.log_g_out = np.load(prefix + "estimator_logs_g_out.npy")
            self.log_p_out = np.load(prefix + "estimator_logs_p_out.npy")
        
        if imu :
            # imu input data log
            self.log_v_imu = np.load(prefix + "estimator_logs_v_imu.npy")
            self.log_w_imu = np.load(prefix + "estimator_logs_w_imu.npy")
            self.log_a_imu = np.load(prefix + "estimator_logs_a_imu.npy")
            self.log_theta_imu = np.load(prefix + "estimator_logs_theta_imu.npy")
        if kin :
            # forward kinematics data log
            self.log_v_kin = np.load(prefix + "estimator_logs_v_kin.npy")
            self.log_z_kin = np.load(prefix + "estimator_logs_z_kin.npy")
            self.log_q = np.load(prefix + "estimator_logs_q.npy")
            self.log_qdot = np.load(prefix + "estimator_logs_qdot.npy")
            self.log_theta_kin = np.load(prefix + "estimator_logs_theta_kin.npy")
            self.log_w_kin = np.load(prefix + "estimator_logs_w_kin.npy")
        if tilt : 
            # tilt log 
            self.log_v_tilt = np.load(prefix + "estimator_logs_v_tilt.npy")
            self.log_g_tilt = np.load(prefix + "estimator_logs_g_tilt.npy")
            self.log_theta_tilt = np.load(prefix + "estimator_logs_theta_tilt.npy")
        if contact : 
            # contact logs
            self.log_contact_forces = np.load(prefix + "estimator_logs_contact_forces.npy")
            self.log_contact_bool = np.load(prefix + "estimator_logs_contact_bool.npy")
            self.log_contact_prob = np.load(prefix + "estimator_logs_contact_prob.npy")
            # Contact switch log
            self.log_p_switch = np.load(prefix + "estimator_logs_p_switch.npy")

        # time
        self.TimeStamp = np.load(prefix + "estimator_logs_t.npy")
        # logs
        self.logs = np.load(prefix + "estimator_logs_logs.npy")
        
        return None
    
    def Printer(self, file, Z):
        self.logger.LogTheLog(file[:] + '  of shape  '+str(Z.shape), "subinfo")
    
    def PlotLog(self, Y, title):
        self.logger.LogTheLog(f"data shape : {Y.shape}", "info")
        if np.ndim(Y) == 1:
            Y = np.reshape(Y, (1, -1))
        ndim, nt = Y.shape
        
        self.grapher.SetLegend([title], ndim=ndim)
        self.grapher.CompareNDdatas([Y], datatype='logs', title=title, StyleAdapter=True)
    
    def PlotLogOut(self):
        self.PlotLog(self.log_v_out, "v_out estimator")
        self.PlotLog(self.log_a_out, "a_out estimator")
        self.PlotLog(self.log_w_out, "omega_out estimator")
        self.PlotLog(self.log_theta_out, "theta_out estimator")
        self.PlotLog(self.log_g_out , "g_out estimator")
        self.PlotLog(self.log_p_out , "p_out estimator")
    def PlotLogTilt(self):
        self.PlotLog(self.log_theta_tilt, "theta_tilt estimator")
        self.PlotLog(self.log_g_tilt , "g_tilt estimator")
        self.PlotLog(self.log_v_tilt , "v_tilt estimator")
    
    def PlotLogIMU(self):
        self.PlotLog(self.log_theta_imu, "theta_imu estimator")
        self.PlotLog(self.log_v_imu , "v_imu estimator")

    def PlotLogContact(self):
        self.PlotLog(self.log_contact_forces, "contact forces estimator")
        self.PlotLog(self.log_contact_bool, "contact bool estimator")
        self.PlotLog(self.log_contact_prob, "contact prob estimator")
    def PrintLogs(self):
        print(self.logs)
        
    
def main():
    reader = DataReader()
    reader.LoadLogs(prefix=None, out=True, tilt=False, contact=False, imu=False, kin=False)
    reader.PlotLogOut()
    

if __name__ == "__main__":
    main()




