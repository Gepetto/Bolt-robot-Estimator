import numpy as numpy
import pinocchio as pin
import time as t
from quaternion import quaternion, from_rotation_vector, rotate_vectors

from Bolt_Utils import utils
from Bolt_ContactEstimator import ContactForcesEstimator
from Bolt_Filter import Filter

class Estimator():
    def __init__(self, 
                ModelPathth="",
                UrdfPath="",
                Talkative=True, 
                FilterType = "complementary"):
        self.MsgName = "Bolt Estimator v0.0"
        self.Talkative=Talkative
        if self.Talkative : print("Initializing " + self.MsgName + " ...")
        self.robot = pin.RobotWrapper.BuildFromURDF(UrdfPath, ModelPath)
        if self.Talkative : print("  -> URDF built")
        # self.q = 0. etc
        self.ReadSensor()
        if self.Talkative : print("  -> Sensors read, initial data acquired")
        if FilterType=="complementary":
            self.filter = Filter()
        if self.Talkative : print("  -> Filter of type " + FilterType + " added")
        # returns info on Slips, Contact Forces, Contact with the ground
        self.ContactEstimator = ContactEstimator()
        if self.Talkative : print(self.MsgName +" initialized successfully.")


    def Get(self, data="acceleration"):
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
            return self.ContactEstimator.LegsonGround(Kinpos, self.a, self.Fcontact)
        
        # ...
        else :
            print("  *!* Could not get data '" + data + "'. Unrecognised data getter.")



    def ReadSensor(self):
        # acceleration and rotation speed from IMU
        self.a = 0.
        self.q = 0.
        # Kinematic data from motors
        self.q = 0.
        # torques from motors
        self.tau=0.
        return None

    
    def KinematicAttitude(self, KinPos):
        # uses robot model to provide attitude estimate based on encoder data
        LeftContact, RightContact = self.ContactEstimator.LegsonGround(Kinpos, self.a, self.Fcontact)
        if LeftContact : 
            pass
        elif RightContact :
            pass
        else :
            if self.Talkative : print("  *!* No legs are touching the ground")
        return None


    
    def IMUAttitude(self):
        # average the robot acceleration over a long time to find direction of gravity
        # self.AllTimeAcceleration = [ [ax, ay, az], ...]
        try :
            AvgAcc = np.average(self.AllTimeAcceleration, Axis=0)
        except :
            print("  *!* Could not compute average acceleration. Using default acceleration instead.")
            AvgAcc = np.array([0, 0, 1])
        
        # compute rotation matrix [TO DERIVE AS QUATERNION]
        u1 = utils.normalize(AvgAcc)
        u2 = utils.cross(u1, np.array([0, 0, 1]))
        if u2 == np.array([0, 0, 0]):
            u2[1] = 1
        u3 = utils.cross(u1, u2)
        R = utils.MatrixFromVectors((u1, u2, u3))
        return R




    def AttitudeFusion(self, KinPos, Gyro):
        # uses attitude Kinematic estimate and gyro data to provide attitude estimate
        AttitudeFromKin = self.KinematicAttitude(KinPos)
        AttitudeFromIMU = self.IMUAttitude()

        return None

    def KinematicSpeed(self):
        # uses Kinematic data (the waist of the robot) to approximate speed
        return None

    def SpeedFusion(self):
        # uses Kinematic-derived speed estimate and gyro (?) to estimate speed
        return None





def main():
    pass

if __name__ == "__main__":
    main()