import numpy as np



class ContactForcesEstimator():
    def __init__(self) -> None:
        print("Contact Forces Estimator started")
    
    def ContactForces(self, current, position) -> tuple[np.ndarray, np.ndarray]:
        return None

    def LegsOnGround(self, Kinpos, Acc, Fcontact) -> tuple[bool, bool]:
        # return if left and right leg are in contact with the ground
        LeftContact, RightContact = False, False
        return LeftContact, RightContact
    
    def Slips(self) -> tuple[bool, bool]:
        # uses Kinematics to compute feet acceleration and determin if the feet is slipping
        LeftSlipProb, RightSlipProb = 0.0, 0.0
        # 0 , 0.8 : Left feet stable, right feet potentially slipping
        return LeftSlipProb, RightSlipProb

    def FeetPositions(self) -> tuple[np.ndarray, np.ndarray]:
        # parce que Constant en a besoin
        pass
