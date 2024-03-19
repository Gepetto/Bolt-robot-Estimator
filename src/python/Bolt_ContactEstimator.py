



class ContactForcesEstimator():
    def __init__(self):
        print("Contact Forces Estimator started")
    
    def ContactForces(self, current, position):
        return None

    def LegsOnGround(self, Kinpos, Acc, Fcontact):
        # return if left and right leg are in contact with the ground
        LeftContact, RightContact = False, False
        return LeftContact, RightContact
    
    def Slips(self):
        # uses Kinematics to compute feet acceleration and determin if the feet is slipping
        LeftSlipProb, RightSlipProb = 0.0, 0.0
        # 0 , 0.8 : Left feet stable, right feet potentially slipping
        return LeftSlipProb, RightSlipProb

    def FeetPositions(self):
        # parce que constant en a besoin
        pass
