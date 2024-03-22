import numpy as np
import pinocchio as pin


class ContactForcesEstimator():
    def __init__(self, robot, LeftFootFrameID, RightFootFrameID, logger) -> None:
        print("Contact Forces Estimator started")
        self.robot = robot # for pinocchio computations
        self.LeftFootFrameID = LeftFootFrameID
        self.RightFootFrameID = RightFootFrameID
        self.logger = logger
    
    def ContactForces(self, torques, position) -> tuple[np.ndarray, np.ndarray]:
        return None

    def LegsOnGround(self, Kinpos, Acc, Fcontact) -> tuple[bool, bool]:
        # return wether or not left and right legs are in contact with the ground
        LeftContact, RightContact = False, False
        return LeftContact, RightContact


    def LegsOnGroundKin(self, Kinpos, vertical) -> tuple[bool, bool]:
        # return iwether or notf left and right legs are in contact with the ground
        # uses the z distance from foot to base, with kinematics only and a vertical direction provided by IMU
        
        self.robot.framesForwardKinematics(q=Kinpos)
        robot.updateFramePlacements()

        LeftFootPos = gait.rdata.oMf[self.LeftFootFrameID].translation
        RightFootPos = gait.rdata.oMf[self.RightFootFrameID].translation
        
        VerticalDistLeft = scalar(LeftFootPos, vertical)
        VerticalDistRight = scalar(RightFootPos, vertical)

        if isapprox(VerticalDistLeft, VerticalDistRight, eps=3e-3):
            # both feet might be in contact
            LeftContact, RightContact = True, True
        elif VerticalDistLeft > VerticalDistRight :
            # Left foot is much further away than right foot, and therefore should be in contact with the ground
            LeftContact, RightContact = True, False
        else : 
            # Right foot is much further away than right foot, and therefore should be in contact with the ground
            LeftContact, RightContact = False, True
        return LeftContact, RightContact


    
    def Slips(self) -> tuple[bool, bool]:
        # uses Kinematics to compute feet acceleration and determin if the feet is slipping
        LeftSlipProb, RightSlipProb = 0.0, 0.0
        # 0 , 0.8 : Left feet stable, right feet potentially slipping
        return LeftSlipProb, RightSlipProb

    def FeetPositions(self) -> tuple[np.ndarray, np.ndarray]:
        # parce que Constant en a besoin
        pass
