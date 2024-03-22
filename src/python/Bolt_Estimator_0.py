import numpy as numpy
import pinocchio as pin
import time as t
from quaternion import quaternion, from_rotation_vector, rotate_vectors

from Bolt_Utils import utils
from Bolt_Utils import Log

from Bolt_ContactEstimator import ContactForcesEstimator
from Bolt_Filter import Filter
from Bolt_Filter_Complementary import ComplementaryFilter


"""
An dummy estimator for Bolt Bipedal Robot

    Program description

    Class estimator Description

License, authors, LAAS

"""




class Estimator():
    def __init__(self, 
                ModelPathth="",
                UrdfPath="",
                Talkative=True, 
                AttitudeFilterType = "complementary",
                SpeedFilterType = "complementary",
                TimeStep = None,
                MemorySize = 100,
                IterNumber = 1000) -> None:

        self.MsgName = "Bolt Estimator v0.3"
        self.Talkative=Talkative
        self.logs = Log()
        logs.LogTheLog(" Starting log of" + self.MsgName, ToPrint=False)
        if self.Talkative : logs.LogTheLog("Initializing " + self.MsgName + "...", style="title")
        
        # loading data from file
        self.robot = pin.RobotWrapper.BuildFromURDF(UrdfPath, ModelPath)
        self.FeetIndexes = [0, 0]
        if self.Talkative : logs.LogTheLog("URDF built")

        if TimeStep is not None :
            self.TimeStep = TimeStep
        else:
            self.TimeStep = 0.01 # ??? aucune id
        # initializes data & logs with np.zeros arrays
        self.InitImuData()
        self.InitKinematicsData()
        self.InitLogMatrixes()

        # check that sensors can be read 
        self.ReadSensor()
        if self.Talkative : logs.LogTheLog("Sensors read, initial data acquired")
        
        # set desired filters types for attitude and speed
        # for the time being, complementary only
        if AttitudeFilterType=="complementary":
            self.attitudeFilter = ComplementaryFilter(parameters=(0.001, 50), name="attitude complementary filter", talkative=Talkative)
        if self.Talkative : logs.LogTheLog("Filter of type " + AttitudeFilterType + " added")
        
        if SpeedFilterType=="complementary":
            self.SpeedFilter = ComplementaryFilter(parameters=(0.001, 50), name="speed complementary filter", talkative=Talkative)
        if self.Talkative : logs.LogTheLog("Filter of type " + SpeedFilterType + " added")

        # returns info on Slips, Contact Forces, Contact with the ground
        self.ContactEstimator = ContactEstimator()
        if self.Talkative : logs.LogTheLog("Contact Estimator added ")

        
        self.MemorySize = MemorySize
        self.AllTimeAcceleration, self.AllTimeq = np.zeros((3, self.MemorySize)), np.zeros((3, self.MemorySize))
        if self.Talkative : logs.LogTheLog(self.MsgName +" initialized successfully.")


    def ExternalDataCaster(self, DataType, ReceivedData) -> None:
        # In case data from elsewhere needs to be converted to another format, or truncated
        if DataType == "acceleration":
            self.a = np.array(ReceivedData)
        #...
        return None


    def InitImuData(self) -> None :
        # initialize data to the right format
        self.a_imu = np.zeros((3,))            
        self.w_imu = np.zeros((3,))
        self.DeltaTheta = np.zeros((4,))
        self.DeltaV = np.zeros((3,))
        return None
    def InitKinematicsData(self) -> None :
        # initialize data to the right format
        # base kinematics
        self.v_fk = np.zeros((3,))
        self.z_fk = np.zeros((1,))
        # motors positions & velocities
        self.q = np.zeros((12, ))
        self.qp = np.zeros((12, ))
        return None
    def InitLogMatrixes(self) -> None :
        # initialize data to the right format
        # base velocitie & co, post-filtering logs
        self.log_v_out = np.zeros([3, self.IterNumber])
        self.log_w_out = np.zeros([3, self.IterNumber])
        self.log_a_out = np.zeros([3, self.IterNumber])
        self.log_theta_out = np.zeros([3, self.IterNumber])
        # imu data log
        self.log_v_imu = np.zeros([3, self.IterNumber])
        self.log_w_imu = np.zeros([3, self.IterNumber])
        self.log_a_imu = np.zeros([3, self.IterNumber])
        self.log_theta_imu = np.zeros([3, self.IterNumber])
        # forward kinematics data log
        self.log_v_fk = np.zeros([3, self.IterNumber])
        self.log_z_fk = np.zeros([1, self.IterNumber])
        self.log_q = np.zeros([12, self.IterNumber])
        self.log_qp = np.zeros([12, self.IterNumber])

        # other logs
        self.iter = 0.
        return None

    def UpdateLogMatrixes(self) -> None :
        if self.iter > self.IterNumber:
            # Logs matrices' size will not be sufficient
            if Talkative : logs.LogTheLog("Excedind planned number of executions, IterNumber = " + str(self.IterNumber), style="warn")

            """
            # base velocitie & co, post-filtering logs
            self.log_v_out
            self.log_w_out
            self.log_a_out
            self.log_theta_out
            # imu data log
            self.log_v_imu
            self.log_w_imu
            self.log_a_imu
            self.log_theta_imu 
            """
        # update logs with latest data
        # base velocitie & co, post-filtering logs
        self.log_v_out[:, self.iter] = self.v
        self.log_w_out[:, self.iter] = self.w
        self.log_a_out[:, self.iter] = self.a
        self.log_theta_out[:, self.iter] = self.theta
        # imu data log
        self.log_v_imu[:, self.iter] = self.v_imu
        self.log_w_imu[:, self.iter] = self.w_imu
        self.log_a_imu[:, self.iter] = self.a_imu
        self.log_theta_imu[:, self.iter] = self.theta_imu
        # forward kinematics data log
        self.log_v_fk[:, self.iter] = self.v_fk
        self.log_z_fk[:, self.iter] = self.z_fk
        self.log_q[:, self.iter] = self.q
        self.log_qp[:, self.iter] = self.qp
        return None





 

    def Get(self, data="acceleration") -> np.ndarray:
        # getter for all internal pertinent data
        if data=="acceleration":
            return self.a
        elif data=="rotation_speed":
            return self.w

        elif data=="attitude":
            return quaternion(R)
        elif data=="com_position":
            return self.c
        elif data=="com_speed":
            return self.cdot
        
        elif data=="base_speed":
            return self.v
        
        elif data=="feet_contact":
            #return self.ContactEstimator.LegsonGround(Kinpos, self.a, self.Fcontact) unconsistent with return type
            return self.ContactEstimator.ContactForces()
        
        # ...
        else :
            logs.LogTheLog("Could not get data '" + data + "'. Unrecognised data getter.", style="warn")
            return None



    def ReadSensor(self, device) -> None:
        # base acceleration, acceleration with gravity and rotation speed from IMU
        self.a_imu = device.baseLinearAcceleration # COPIED FROM SOLO CODE, CHECK CONSISTENCY WITH BOLT MASTERBOARD
        self.ag_imu = 0.
        self.w_imu = device.baseAngularVelocity
        # integrated data from IMU
        self.DeltaTheta = device.baseOrientation - offset_yaw_IMU # to be found
        self.DeltaV = 0.
        # Kinematic data from encoders
        self.q = device.q_mes
        self.qp = device.v_mes
        # data from forward kinematics
        self.v_fk = 
        self.z_fk = 
        # torques from motors
        self.tau = 

        #self.ExternalDataCaster("acceleration", self.a)
        self.UpdateMemory()
        return None
    
    def UpdateMemory(self) -> None:
        # updates a set of previously acquired data
        # to optimize

        # acceleration
        self.AllTimeAcceleration[:-1] = self.AllTimeAcceleration[1:]
        self.AllTimeAcceleration[-1] = self.a
        # q
        self.AllTimeq[:-1] = self.AllTimeq[1:]
        self.AllTimeq[-1] = self.q
        # ...

        return None
    
    def AcquireInitialData(self) -> None:
        # initializes averaged data while the robot starts
        # might be useless
        self.ReferenceAcceleration = np.average(self.AllTimeAcceleration, axis=0)
        self.ReferenceKinematics = np.average(self.AllTimeq, axis=0)
        self.ReferenceOrientation = 0.
        return None


    
    def KinematicAttitude(self, KinPos) -> np.ndarray:
        # uses robot model and rotation speed to provide attitude estimate based on encoder data
        LeftContact, RightContact = self.ContactEstimator.LegsonGround(Kinpos, self.a, self.Fcontact)
        if LeftContact :
            if self.Talkative : logs.LogTheLog("left foot touching the ground")
            robot.forwardKinematics(self.q, [self.v,[self.a]])
            pass
        elif RightContact :
            if self.Talkative : logs.LogTheLog("right foot touching the ground")
            pass
        else :
            if self.Talkative : logs.LogTheLog("No legs are touching the ground", style="warn")
        return R


    
    def IMUAttitude(self) -> np.ndarray:
        # average the robot acceleration over a long time to find direction of gravity, 
        # compares it to his current acceleration and returns a rotation matrix

        # self.AllTimeAcceleration = [ [ax, ay, az], ...]

        # COMPLÈTEMENT BIDON CORRIGER CETTE MERDE ASAP
        try :
            AvgAcc = np.average(self.AllTimeAcceleration, Axis=0)
        except :
            if self.Talkative : logs.LogTheLog("Could not compute average acceleration. Using default acceleration instead.", style="warn")
            AvgAcc = np.array([0, 0, 1])
        DeltAcc = self.a - AvgAcc
        
        # compute rotation matrix [TO DERIVE AS QUATERNION]
        u1 = utils.normalize(DelAcc)
        u2 = utils.cross(u1, np.array([0, 0, 1]))
        if u2 == np.array([0, 0, 0]):
            u2[1] = 1
        u3 = utils.cross(u1, u2)
        R = utils.MatrixFromVectors((u1, u2, u3))
        return R
    
    def GyroAttitude(self) -> np.ndarray:
        # Uses integrated angular velocity to derive rotation matrix 
        # 3DM-CX5-AHRS sensor returns Δθ
        return self.ReferenceOrientation + self.DeltaTheta

    def AttitudeFusion(self, KinPos, Gyro) -> None :
        # uses attitude Kinematic estimate and gyro data to provide attitude estimate
        AttitudeFromKin = self.KinematicAttitude(KinPos)
        AttitudeFromIMU = self.IMUAttitude()
        AttitudeFromGyro = self.GyroAttitude()
        return None


    def IMUSpeed(self) -> np.ndarray:
        # direclty uses IMU data to approximate speed
        return self.ReferenceSpeed + self.DeltaV

    def KinematicSpeed(self) -> np.ndarray:
        # uses Kinematic data (the waist of the robot) 
        # along with contact and rotation speed information to approximate speed

        self.q 
        self.w
        LeftContact, RightContact = self.ContactEstimator.LegsonGround(Kinpos, self.a, self.Fcontact)
        if LeftContact and RightContact :
            # both feet on the ground. Will use pinocchio estimate that induce the lowest base speed
            pass
        return None

    def SpeedFusion(self) -> None:
        # uses Kinematic-derived speed estimate and gyro (?) to estimate speed
        return None
    
    def RunFilter(self):
        pass





def main():
    pass

if __name__ == "__main__":
    main()