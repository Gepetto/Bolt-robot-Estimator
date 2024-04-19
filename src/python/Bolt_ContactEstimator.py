import numpy as np
import pinocchio as pin


class ContactEstimator():
    def __init__(self, robot, LeftFootFrameID, RightFootFrameID, logger) -> None:
        
        self.robot = robot # for pinocchio computations
        self.LeftFootFrameID = LeftFootFrameID
        self.RightFootFrameID = RightFootFrameID
        self.logger = logger
        self.logger.LogTheLog("Contact Forces Estimator started", style="subtitle")
    
    
    
    
    def ContactForces1d(self, RobotWeight, Torques, Positions, Dynamic=0.4, Resolution=10) -> tuple[np.ndarray, np.ndarray]:
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
        MaxForce = RobotWeight*9.81 * (1+Dynamic)
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
    
    
    def ConsistencyChecker(self, thresold=0.2):
        
        thresold = weight*9.81*thresold
        LeftConfidence, RightConfidence = 0.5, 0.5
        
        LcF_1, RcF_1 = self.ContactForces1d()
        LcF_3, RcF_3 = self.ContactForces3d()
        # average vertical force
        RcF = (RcF_1[2] + RcF_3[2]) * 0.5
        
        self.logger.LogTheLog("checking forces estimation consistency")
        
        # deltas between forces
        DeltaR = np.linalg.norm(RcF_1) - np.linalg.norm(RcF_3)
        DeltaR3d = np.linalg.norm(RcF_1 - RcF_3)
        DeltaRv = np.linalg.norm(RcF_1[2] - RcF_3[2])
        # confidence in non-silp : horizontal norm over vertical norm
        MuR = np.sqrt(RcF_3[0]**2 + RcF_3[1]**2) / RcF_3[2]
        
        # to tune
        b, b0 = 1.5, 8.0
        PR = 1/ (1 + np.exp(-b*RcF + b0))
        
        RightConfidence = 1.0 - ( 0.1*DeltaR + 0.2*DeltaR3d + 0.1*DeltaRv + 0.1*MuR)
        
        if DeltaR < thresold :
             # norms are consistent
             pass
         
         
            
             
        
        
    


    def LegsOnGround(self, Kinpos, Acc, Fcontact) -> tuple[bool, bool]:
        # return wether or not left and right legs are in contact with the ground
        LeftContact, RightContact = False, False
        return LeftContact, RightContact


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


    
    def Slips(self) -> tuple[bool, bool]:
        # uses Kinematics to compute feet acceleration and determin if the feet is slipping
        LeftSlipProb, RightSlipProb = 0.0, 0.0
        # 0 , 0.8 : Left feet stable, right feet potentially slipping
        
        
        LcF_3, RcF_3 = self.ContactForces3d()
        # slipery level : horizontal norm over vertical norm
        MuR = np.sqrt(RcF_3[0]**2 + RcF_3[1]**2) / RcF_3[2]       
        return LeftSlipProb, RightSlipProb

    def FeetPositions(self) -> tuple[np.ndarray, np.ndarray]:
        # parce que Constant en a besoin
        pass
