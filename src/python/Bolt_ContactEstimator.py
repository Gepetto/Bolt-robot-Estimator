import numpy as np
import pinocchio as pin


class ContactEstimator():
    def __init__(self, 
                robot, 
                LeftFootFrameID  : int, 
                RightFootFrameID : int, 
                IterNumber       : int, 
                logger           ) -> None:
        
        self.robot = robot # for pinocchio computations
        # feet id
        self.LeftFootFrameID = LeftFootFrameID
        self.RightFootFrameID = RightFootFrameID
        self.logger = logger
        self.logger.LogTheLog("Contact Forces Estimator started", style="subtitle")
        
        self.mass = pin.computeTotalMass(self.robot.model)
        self.logger.LogTheLog("Mass of robot : " + str(self.mass), style="subinfo")

        self.InitVariables()
        self.InitLogMatrixes()
        self.IterNumber = IterNumber
        self.iter = 0
    
    def InitLogMatrixes(self) -> None:
        # contact forces logs
        self.Log_LcF_3d = np.zeros([3, self.IterNumber])
        self.Log_LcF_1d = np.zeros([3, self.IterNumber])
        self.Log_RcF_3d = np.zeros([3, self.IterNumber])
        self.Log_RcF_1d = np.zeros([3, self.IterNumber])
        # left and right trust in contacts logs
        self.Log_Delta1D = np.zeros([2, self.IterNumber])
        self.Log_Delta3D = np.zeros([2, self.IterNumber])
        self.Log_DeltaVertical = np.zeros([2, self.IterNumber])
        # slipping logs
        self.Log_Mu = np.zeros([2, self.IterNumber])
        self.Log_SlipProb = np.zeros([2, self.IterNumber])
        # probability of contact and trust in contact
        self.Log_Contact = np.zeros([2, self.IterNumber], dtype=bool)
        self.Log_Trust = np.zeros([2, self.IterNumber])
        return None
    

    def UpdateLogMatrixes(self) -> None:
        # contact forces logs
        self.Log_LcF_3d[:, self.iter] = self.LcF_3d
        self.Log_LcF_1d[:, self.iter] = self.LcF_1d
        self.Log_RcF_3d[:, self.iter] = self.RcF_3d
        self.Log_RcF_1d[:, self.iter] = self.RcF_1d
        # left and right trust in contacts logs
        self.Log_Delta1D[:, self.iter] = [self.DeltaL, self.DeltaR]
        self.Log_Delta3D[:, self.iter] = [self.DeltaL3d, self.DeltaR3d]
        self.Log_DeltaVertical[:, self.iter] = [self.DeltaLV, self.DeltaRV]
        # slipping logs
        self.Log_Mu[:, self.iter] = [self.MuL, self.MuR]
        self.Log_SlipProb[:, self.iter] = [self.SlipProbL, self.SlipProbR]
        # probability of contact and trust in contact
        self.Log_Contact[:, self.iter] = [self.LeftContact, self.RightContact]
        self.Log_Trust[:, self.iter] = [self.TrustL, self.TrustR]
        # number of updates
        self.iter += 1
        return None
        
        
        
    def InitVariables(self) -> None:
        # deltas and slips
        self.DeltaL, self.DeltaL3d, self.DeltaLV, self.MuL, self.TrustL = 0., 0., 0., 0., 0.
        self.DeltaR, self.DeltaR3d, self.DeltaRV, self.MuR, self.TrustR = 0., 0., 0., 0., 0.
        self.SlipProbL = 0.
        self.SlipProbR = 0.
        # contact and trust
        self.ContactProbL = 0.
        self.ContactProbR = 0.
        self.TrustL = 0.
        self.TrustR = 0.
        self.LeftContact = False
        self.RightContact = False

        # contact forces
        self.LcF_1d = np.zeros(3)
        self.RcF_1d = np.zeros(3)
        self.LcF_3d = np.zeros(3)
        self.RcF_3d = np.zeros(3)

        # foot position
        self.RightFootPos = np.zeros(3)
        self.LeftFootPos = np.zeros(3)

        
        
    def __ProbDiscriminator(self, x, center, stiffness=5):
        """Bring x data between 0 and 1, such that P(x=center)=0.5. The greater stiffness, the greater dP/dx (esp. around x=center)"""
        b0 = stiffness
        b = b0/center
        return 1/ (1 + np.exp(-b*x + b0))
    
    
    def ContactForces1d(self, Torques, Positions, Dynamic=0.4, Resolution=10) -> tuple[np.ndarray, np.ndarray]:
        # this function assumes that the weight is somehow distributed over both feet, and search an approximation of that distribution
        # NO EXTERNAL FORCES
        # contact forces ASSUMED VERTICAL
        qmes = Torques
        Deltas = []
        DeltaMin = qmes
        CF = (0, 0)
        
        # Dynamic states that contact forces sum is in [weight * (1-dynamic) , weight * (1+dynamic)]
        # minimal contact force is 0
        MinForce = 0
        # maximal contact force is weight * (1+dynamic)
        MaxForce = self.mass*9.81 * (1+Dynamic)
        # the lower and upper indices such that contact forces sum is in [weight * (1-dynamic) , weight * (1+dynamic)]
        upper = int(np.floor(Resolution*(1+Dynamic)))
        lower = int(np.ceil(Resolution*(1-Dynamic/2)))
        #print(f'upper : {upper}, lower : {lower}')
        # possible force range from min to max force
        PossibleForce_Left = np.linspace(MinForce, MaxForce, upper)
        PossibleForce_Right = PossibleForce_Left.copy()        
        
        for i in range(upper):
            # left contact force takes all possible values
            # for a given value of the left contact force, contact forces sum is in [weight * (1-dynamic) , weight * (1+dynamic)]
            ReasonnableMin = max(0, lower-i)
            ReasonnableMax = max(0, upper-i)
            for j in range( ReasonnableMin, ReasonnableMax):
                weight = PossibleForce_Left[i] + PossibleForce_Right[j]
                #print(f'Indices :  {i} ; {j}     Valeur :  {np.floor(weight)}       Répartition :  {np.floor(PossibleForce_Left[i])}/{np.floor(PossibleForce_Right[j])}')
                
                # both our contact forces are within reasonnable range. Looking for the most likely forces repartition.
                # contact forces ASSUMED VERTICAL
                LcF = [0, 0, PossibleForce_Left[i]]
                RcF = [0, 0, PossibleForce_Right[j]]

                qpin = pin(LcF, RcF, Positions)
                Deltas.append( np.abs(qpin-qmes))
                if Deltas[-1] < DeltaMin:
                    # error is minimal
                    DeltaMin = Deltas[-1]
                    CF = (LcF, RcF)
        return CF
    
    
    def ContactForces3d(self, Torques, Positions, frames=[10,18]) -> tuple[np.ndarray, np.ndarray]:
        # simply uses jacobian and torques
        # set left foot and right foot data apart
        LF_id, RF_id = frames[0], frames[1]
        
        # compute jacobian matrixes for both feet
        JL = pin.computeFrameJacobian(self.bolt.model, self.bolt.data, self.q, LF_id).copy()
        JR = pin.computeFrameJacobian(self.bolt.model, self.bolt.data, self.q, RF_id).copy()
        
        # compute 6d contact wrenches
        LcF = np.linalg.inv(np.transpose(JL))@Torques
        RcF = np.linalg.inv(np.transpose(JR))@Torques
        
        return LcF[:3], RcF[:3]


    def ConsistencyChecker(self, ContactForce1d, ContactForce3d) -> tuple[float, float, float]:
        """ Check the consistency of a contact force computed by two functions."""
        # difference between both contact forces
        Delta = np.abs(np.linalg.norm(ContactForce1d) - np.linalg.norm(ContactForce3d))
        Delta3d = np.linalg.norm(ContactForce1d - ContactForce3d)
        DeltaVertical = np.linalg.norm(ContactForce1d[2] - ContactForce3d[2])
        
        return Delta, Delta3d, DeltaVertical
    
    
    def ContactProbability(self, ContactForce1d, ContactForce3d, thresold=0.3) -> float:
        # check consistency in contact forces estimates
        # warning thresold
        thresold = self.mass*9.81*thresold

        # average vertical force
        VerticalForce = (ContactForce1d[2] + ContactForce3d[2]) * 0.5
        return self.__ProbDiscriminator(VerticalForce, thresold, 5)
         
            
    def Slips(self, ContactForce3d, FootFrameID, MuTrigger=0.2, FootAcc=0., AccTrigger=2., mingler=1.) -> float:
        """ Compute a coefficient about wether or not foot is slipping
        Args :
        Return : an average (weighted with mingler) of computed slipping probabilities
        """   
        # slipery level : horizontal norm over vertical DISTANCE
        # can be < 0
        Mu = np.sqrt(ContactForce3d[0]**2 + ContactForce3d[1]**2) / ContactForce3d[2]
        Slip0 = self.__ProbDiscriminator(Mu, MuTrigger, 7)

        # uses kinematics to compute foot horizontal speed
        
        # uses Kinematics to compute foot acceleration and determin if the feet is slipping
        HorizontalAcc = np.linalg.norm(FootAcc[:2])
        Slip1 = self.__ProbDiscriminator(HorizontalAcc, AccTrigger, 5)
        # 0 , 0.8 : feet stable, feet potentially slipping
        SlipProb = mingler * Slip0 + (1-mingler) * Slip1    
  
        return SlipProb 
        
        
    def TrustContact(self,Delta, Delta3d, DeltaVertical, Mu, coeffs = (0.1, 0.2, 0.1, 0.1)) -> float:
        c1, c2, c3, c4 = coeffs
        Trust = 1.0 - ( c1*Delta + c2*Delta3d + c3*DeltaVertical + c4*Mu)
        return Trust


    def ContactBool(self, Contact, ContactProb, Trust, ProbThresold, TrustThresold) -> bool:
        if ContactProb > ProbThresold :
            if ContactProb > 1.2*ProbThresold:
                # high probability of contact
                Contact = True
            elif Trust > TrustThresold:
                # mid probability but high trust
                Contact = True
            else :
                # mid probability and low trust
                pass # no change to variable (keeping previous state)
        else :
            # low contact probability
            if Trust < TrustThresold:
                # low trust
                pass # no change
            else :
                # high trust
                Contact = False
        return Contact


    def LegsOnGround(self, KinPos, Acc, Torques, ProbThresold=0.5, TrustThresold=0.5) -> tuple[bool, bool]:
        """Return wether or not left and right legs are in contact with the ground
        Args = 
        Return = 
        """
        
        # get the contact forces
        self.LcF_1d, self.RcF_1d = self.ContactForces1d(Torques=Torques, Positions=KinPos, Dynamic=0.4, Resolution=10)
        self.LcF_3d, self.RcF_3d = self.ContactForces3d(Torques=Torques, Positions=KinPos, frames=[self.LeftFootFrameID, self.RightFootFrameID])

        # get deltas
        self.DeltaL, self.DeltaL3d, self.DeltaLV = self.ConsistencyChecker(self.LcF_1d, self.LcF_3d)
        self.DeltaR, self.DeltaR3d, self.DeltaRV = self.ConsistencyChecker(self.RcF_1d, self.RcF_3d)

        # get probability of slip
        self.SlipProbL = self.Slips(self.LcF_3d, self.LeftFootFrameID, MuTrigger=0.2, FootAcc=0., AccTrigger=2., mingler=1.)
        self.SlipProbR = self.Slips(self.RcF_3d, self.RightFootFrameID, MuTrigger=0.2, FootAcc=0., AccTrigger=2., mingler=1.)

        # compute probability of contact based on god's sacred will and some meth
        self.ContactProbL = self.ContactProbability(self.LcF_1d, self.LcF_3d, thresold=0.3)
        self.ContactProbR = self.ContactProbability(self.RcF_1d, self.RcF_3d, thresold=0.3)

        # compute trust in contact
        self.TrustL = self.TrustContact(self.DeltaL, self.DeltaL3d, self.DeltaLV, self.MuL, coeffs=(0.1, 0.2, 0.1, 0.1))
        self.TrustR = self.TrustContact(self.DeltaR, self.DeltaR3d, self.DeltaRV, self.MuR, coeffs=(0.1, 0.2, 0.1, 0.1))



        # log contact forces and deltas
        self.UpdateLogMatrixes()
        
        self.LeftContact = self.ContactBool(self.LeftContact, self.ContactProbL, self.TrustL, ProbThresold, TrustThresold)
        self.RightContact = self.ContactBool(self.RightContact, self.ContactProbR, self.TrustR, ProbThresold, TrustThresold)
         
        
        return self.LeftContact, self.RightContact
    
    def LegsOnGroundTorque(self, Torques, LowerThresold=0.3, UpperThresold=1.2)-> tuple[bool, bool]:
        # knees torques index
        LeftKneeID, RightKneeID = 2, 4
        





    def LegsOnGroundKin(self, Kinpos, vertical) -> tuple[bool, bool]:
        # return iwether or notf left and right legs are in contact with the ground
        # uses the z distance from foot to base, with kinematics only and a vertical direction provided by IMU
        
        self.robot.framesForwardKinematics(q=Kinpos)
        self.robot.updateFramePlacements()

        LeftFootPos = self.robot.gait.rdata.oMf[self.LeftFootFrameID].translation
        RightFootPos = self.robot.gait.rdata.oMf[self.RightFootFrameID].translation
        
        VerticalDistLeft = scalar(LeftFootPos, vertical)
        VerticalDistRight = scalar(RightFootPos, vertical)

        if pin.isapprox(VerticalDistLeft, VerticalDistRight, eps=3e-3):
            # both feet might be in contact
            LeftContact, RightContact = True, True
        elif VerticalDistLeft > VerticalDistRight :
            # Left foot is much further away than right foot, and therefore should be in contact with the ground
            LeftContact, RightContact = True, False
        else : 
            # Right foot is much further away than right foot, and therefore should be in contact with the ground
            LeftContact, RightContact = False, True
        return LeftContact, RightContact


    def FeetPositions(self) -> tuple[np.ndarray, np.ndarray]:
        # parce que Constant en a besoin
        return self.LeftFootPos, self.RightFootPos
