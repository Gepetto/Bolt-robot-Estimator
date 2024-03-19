import numpy as numpy
import pinocchio as pin
import time as t
from quaternion import quaternion, from_rotation_vector, rotate_vectors


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
        if self.Talkative : print(self.MsgName +" initialized successfully.")


    def Get(self, data="acceleration"):
        # getter for all internal pertinent data
        if data=="acceleration":
            return self.a
        elif data=="attitude":
            return quaternion
        # ...



    def ReadSensor(self):
        # acceleration and rotation speed from IMU
        self.a = 0.
        # cinematic data from motors
        self.w = 0.
        # torques from motors
        self.tau=0.
        return None

    
    def CinematicAttitude(self, CinPos):
        # uses robot model to provide attitude estimate based on encoder data
        LeftContact, RightContact = self.LegsonGround(Cinpos, self.a, self.Fcontact)

        return None

    ''' ! pas sûr que ça soit pas le travail de l'estimateur de contact ! '''
    def LegsOnGround(self, Cinpos, Acc, Fcontact):
        # return if left and right leg are in contact with the ground
        LeftContact, RightContact = False, False
        return LeftContact, RightContact


    def AttitudeFusion(self, CinPos, Gyro):
        # uses attitude cinematic estimate and gyro data to provide attitude estimate
        CinAttitude = self.CinematicAttitude(CinPos)
        return None

    def CinematicSpeed(self):
        # uses cinematic data (the waist of the robot) to approximate speed
        return None

    def SpeedFusion(self):
        # uses cinematic-derived speed estimate and gyro (?) to estimate speed
        return None





def main():
    pass

if __name__ == "__main__":
    main()