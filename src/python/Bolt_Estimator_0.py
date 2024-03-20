import numpy as numpy
import pinocchio as pin
import time as t
from quaternion import quaternion, from_rotation_vector, rotate_vectors

from Bolt_Utils import utils
from Bolt_ContactEstimator import ContactForcesEstimator
from Bolt_Filter import Filter
from Bolt_Filter_Complementary import ComplementaryFilter



class Estimator():
    def __init__(self, 
                ModelPathth="",
                UrdfPath="",
                Talkative=True, 
                FilterType = "complementary",
                MemorySize = 100) -> None:
        self.MsgName = "Bolt Estimator v0.0"
        self.Talkative=Talkative
        if self.Talkative : print("Initializing " + self.MsgName + " ...")
        self.robot = pin.RobotWrapper.BuildFromURDF(UrdfPath, ModelPath)
        if self.Talkative : print("  -> URDF built")

        # self.q = 0. etc
        self.ReadSensor()
        if self.Talkative : print("  -> Sensors read, initial data acquired")
        
        if FilterType=="complementary":
            self.filter = ComplementaryFilter(parameters=(0.001, 2.5), name="attitude complementary filter", talkative=Talkative)
        if self.Talkative : print("  -> Filter of type " + FilterType + " added")

        # returns info on Slips, Contact Forces, Contact with the ground
        self.ContactEstimator = ContactEstimator()
        if self.Talkative : print("  -> Contact Estimator added ")

        self.MemorySize = MemorySize
        self.AllTimeAcceleration, self.AllTimeq = np.zeros(self.MemorySize), np.zeros(self.MemorySize)
        if self.Talkative : print(self.MsgName +" initialized successfully.")





 

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
            print("  *!* Could not get data '" + data + "'. Unrecognised data getter.")
            return None



    def ReadSensor(self) -> None:
        # acceleration and rotation speed from IMU
        self.a = 0.
        self.q = 0.
        # integrated data from IMU
        self.DeltaTheta = 0.
        self.DeltaV = 0.
        # Kinematic data from motors
        self.q = 0.
        # torques from motors
        self.tau=0.

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
        self.ReferenceAcceleration = np.average(self.AllTimeAcceleration, axis=0)
        self.ReferenceKinematics = np.average(self.AllTimeq, axis=0)
        self.ReferenceOrientation = 0.
        return None


    
    def KinematicAttitude(self, KinPos) -> np.ndarray:
        # uses robot model to provide attitude estimate based on encoder data
        LeftContact, RightContact = self.ContactEstimator.LegsonGround(Kinpos, self.a, self.Fcontact)
        if LeftContact : 
            robot.forwardKinematics(self.q, [self.v,[self.a]])
            pass
        elif RightContact :
            pass
        else :
            if self.Talkative : print("  *!* No legs are touching the ground")
        return R


    
    def IMUAttitude(self) -> np.ndarray:
        # average the robot acceleration over a long time to find direction of gravity, 
        # compares it to his current acceleration and returns a rotation matrix

        # self.AllTimeAcceleration = [ [ax, ay, az], ...]

        # COMPLÈTEMENT BIDON CORRIGER CETTE MERDE ASAP
        try :
            AvgAcc = np.average(self.AllTimeAcceleration, Axis=0)
        except :
            if self.Talkative : print("  *!* Could not compute average acceleration. Using default acceleration instead.")
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

    def KinematicSpeed(self) -> None:
        # uses Kinematic data (the waist of the robot) to approximate speed
        return None

    def SpeedFusion(self) -> None:
        # uses Kinematic-derived speed estimate and gyro (?) to estimate speed
        return None





def main():
    pass

if __name__ == "__main__":
    main()