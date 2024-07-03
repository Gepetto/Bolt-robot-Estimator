import numpy as np
import pinocchio as pin
from bolt_estimator.utils.Bolt_Utils import utils


class ContactEstimator():
    def __init__(self, 
                robot, 
                LeftFootFrameID  : int=10, 
                RightFootFrameID : int=18, 
                LeftKneeFrameID  : int=4,
                RightKneeFrameID : int=8,
                LeftKneeTorqueID : int=2,
                RightKneeTorqueID : int=5,
                IterNumber       : int=1000,
                dt               : float=1e-3,
                MemorySize       : int=5,
                Logging          : bool=True,
                Talkative        : bool=True,
                logger=None           ) -> None:
        
        # pinocchio
        self.bolt = robot
        self.nq = self.bolt.model.nq
        self.nv = self.bolt.model.nv
        # data for different computations
        self.data1D = self.bolt.model.createData()
        self.data3D = self.bolt.model.createData()
        self.dataT = self.bolt.model.createData()
        self.dt = dt


        # Q includes base position, which is updated by pinocchio, and position from encoders
        self.Q = pin.neutral(self.bolt.model)
        self.Qd = pin.utils.zero(self.nv)
        self.Qdd = np.zeros(self.nv)
        self.Tau = np.zeros(6)
        
        # feet id
        self.LeftFootFrameID = LeftFootFrameID
        self.RightFootFrameID = RightFootFrameID
        self.C_id = [LeftFootFrameID, RightFootFrameID]
        # knees id
        self.LeftKneeFrameID = LeftKneeFrameID
        self.RightKneeFrameID = RightKneeFrameID
        self.LeftKneeTorqueID = LeftKneeTorqueID
        self.RightKneeTorqueID = RightKneeTorqueID
        
        # logging options
        if logger is None :
            self.Talkative=False
        else :
            self.Talkative=Talkative
            self.logger = logger
            self.logger.LogTheLog("Contact Forces Estimator started", style="subtitle")
        self.Logging = Logging

        # compute bolt mass
        self.mass = pin.computeTotalMass(self.bolt.model)
        self.logger.LogTheLog("Mass of robot : " + str(self.mass), style="subinfo")
        
        # initialize all variables
        self.IterNumber = IterNumber
        self.InitVariables()
        if self.Logging : self.InitLogMatrixes()
        self.iter = 0

        self.TorqueContacts = [False, False]

        # average contact probability over the last k=memorysize results
        self.MemorySize = MemorySize
        
        # coefficient for averaging contact forces
        self.coeffs = [0.0, 0.3, 0.7] #1D, 3D, torques-based
        if sum(self.coeffs) != 1 and self.Talkative: self.logger.LogTheLog("Coeffs sum not equal to 1", "warn")
        self.c1, self.c2, self.c3 = self.coeffs
    
    
    
    
    def InitLogMatrixes(self) -> None:
        
        # contact forces logs
        self.Log_LcF_3d = np.zeros([3, self.IterNumber])
        self.Log_LcF_1d = np.zeros([3, self.IterNumber])
        self.Log_RcF_3d = np.zeros([3, self.IterNumber])
        self.Log_RcF_1d = np.zeros([3, self.IterNumber])
        self.Log_LcF_T = np.zeros([3, self.IterNumber])
        self.Log_RcF_T = np.zeros([3, self.IterNumber])
        # left and right trust in contacts logs
        self.Log_Delta1D = np.zeros([2, self.IterNumber])
        self.Log_Delta3D = np.zeros([2, self.IterNumber])
        self.Log_DeltaVertical = np.zeros([2, self.IterNumber])
        # slipping logs
        self.Log_Mu = np.zeros([2, self.IterNumber])
        self.Log_SlipProb = np.zeros([2, self.IterNumber])
        # boolean contact, probability of contact and trust in contact
        self.Log_Contact = np.zeros([2, self.IterNumber], dtype=bool)
        self.Log_ContactProbability = np.zeros([2, self.IterNumber])
        self.Log_ContactProbability_F = np.zeros([2, self.IterNumber])
        self.Log_ContactProbability_T = np.zeros([2, self.IterNumber])
        self.Log_Trust = np.zeros([2, self.IterNumber])
        return None
    

    def UpdateLogMatrixes(self) -> None:
        LogIter = self.iter
        if self.iter >= self.IterNumber:
            # Logs matrices' size will not be sufficient
            self.logger.LogTheLog("Excedind planned number of executions for ContactEstimator, IterNumber = " + str(self.IterNumber), style="warn", ToPrint=self.Talkative)
            LogIter = self.IterNumber-1

        # contact forces logs
        self.Log_LcF_3d[:, LogIter] = self.LcF_3d[:]
        self.Log_LcF_1d[:, LogIter] = self.LcF_1d[:]
        self.Log_RcF_3d[:, LogIter] = self.RcF_3d[:]
        self.Log_RcF_1d[:, LogIter] = self.RcF_1d[:]
        self.Log_LcF_T[:, LogIter] = self.LcF_T[:]
        self.Log_RcF_T[:, LogIter] = self.RcF_T[:]
        # left and right trust in contacts logs
        self.Log_Delta1D[:, LogIter] = [self.DeltaL, self.DeltaR]
        self.Log_Delta3D[:, LogIter] = [self.DeltaL3d, self.DeltaR3d]
        self.Log_DeltaVertical[:, LogIter] = [self.DeltaLV, self.DeltaRV]
        # slipping logs
        self.Log_Mu[:, LogIter] = [self.MuL, self.MuR]
        self.Log_SlipProb[:, LogIter] = [self.SlipProbL, self.SlipProbR]
        # boolean contact, probability of contact and trust in contact
        self.Log_Contact[:, LogIter] = [self.LeftContact, self.RightContact]
        self.Log_ContactProbability[:, LogIter] = [self.ContactProbL, self.ContactProbR]
        self.Log_ContactProbability_F[:, LogIter] = [self.ContactProbL_F, self.ContactProbR_F]
        self.Log_ContactProbability_T[:, LogIter] = [self.ContactProbL_T, self.ContactProbR_T]
        self.Log_Trust[:, LogIter] = [self.TrustL, self.TrustR]
        # number of updates
        self.iter += 1
        return None
    
    def Get(self, data="cf_1d"):
        # current contact forces
        if data=='current_cf_1d':
            return self.LcF_1d.copy(), self.RcF_1d.copy()
        elif data=='current_cf_3d':
            return self.LcF_3d.copy(), self.RcF_3d.copy()
        elif data=='current_cf_torques':
            return self.LcF_T.copy(), self.RcF_T.copy()
        elif data=='current_cf_averaged':
            LcF = self.c1 * self.LcF_1d + self.c2 * self.LcF_3d + self.c3 * self.LcF_T
            RcF = self.c1 * self.RcF_1d + self.c2 * self.RcF_3d + self.c3 * self.RcF_T
            return LcF, RcF
        
        # contact forces log
        elif data=='cf_1d':
            return self.Log_LcF_1d, self.Log_RcF_1d
        elif data=='cf_3d':
            return self.Log_LcF_3d, self.Log_RcF_3d
        elif data=='cf_torques':
            return self.Log_LcF_T, self.Log_RcF_T
        
        # trusts and co
        elif data=='delta_1d':
            return self.Log_Delta1D
        elif data=='delta_3d':
            return self.Log_Delta3D
        elif data=='mu':
            return self.Log_Mu
        elif data=='slip_prob':
            return self.Log_SlipProb
        elif data=="trust":
            return self.Log_Trust
        
        # bool log
        elif data=="contact_bool":
           return self.Log_Contact
        elif data=="contact_prob":
           return self.Log_ContactProbability
        elif data=="contact_prob_force":
           return self.Log_ContactProbability_F
        elif data=="contact_prob_torque":
           return self.Log_ContactProbability_T
                
        
        
        
        
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
        self.LcF_T = np.zeros(3)
        self.RcF_T = np.zeros(3)

        # foot position
        self.RightFootPos = np.zeros(3)
        self.LeftFootPos = np.zeros(3)

        # contact memory
        self.PastProbL = 0.
        self.PastProbR = 0.
     
        
    def __SigmoidDiscriminator(self, x, center, stiffness=5) -> np.ndarray:
        """Bring x data between 0 and 1, such that P(x=center)=0.5. The greater stiffness, the greater dP/dx (esp. around x=center)"""
        b0 = stiffness
        b = b0/center
        if b*x < -100 :
            return 1
        return 1/ (1 + np.exp(-b*x + b0))
    
    def __ApplyForce(self, ContactForces):
        """ return forces in the right fomat for pinocchio """
        # volé à victor
        ### Build force list for ABA
        forces = [ pin.Force.Zero() for _ in self.bolt.model.joints ]
        # I am supposing here that all contact frames are on separate joints. This is asserted below:
        #assert( len( set( [ cmodel.frames[idf].parentJoint for idf in contactIds ]) ) == len(contactIds) )
        
        for f,idf in zip(ContactForces,self.C_id):
            # Contact forces introduced in ABA as spatial forces at joint frame.
            forces[self.bolt.model.frames[idf].parent] = self.bolt.model.frames[idf].placement * pin.Force(f, 0.*f)
        forces_out = pin.StdVec_Force()
        for f in forces:
            forces_out.append(f)
        return forces_out
    



    
    def ContactForces1d(self, Torques, Q, Qd, BaseAccelerationIMU, Dynamic=0.4, Resolution=10) -> tuple[np.ndarray, np.ndarray]:
        """this function assumes that the weight is somehow distributed over both feet, and search an approximation of that distribution"""
        # NO EXTERNAL FORCES
        # contact forces ASSUMED VERTICAL
        a_mes = BaseAccelerationIMU
        Deltas = []
        DeltaMin = 1e6 #np.linalg.norm(a_mes)
        CF = (np.zeros(3), np.zeros(3))
        

        # update data from encoders
        self.Q[:] = Q[:]
        self.Qd[:] = Qd[:]
        TauPin = np.zeros(self.bolt.model.nv)
        TauPin[-6:] = Torques[:]
        
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
                #print(f'Indices :  {i} ; {j}           Répartition :  {np.floor(PossibleForce_Left[i])}/{np.floor(PossibleForce_Right[j])}')
                
                # both our contact forces are within reasonnable range. Looking for the most likely forces repartition.
                # contact forces ASSUMED VERTICAL
                LcF = np.array([0, 0, PossibleForce_Left[i]])
                RcF = np.array([0, 0, PossibleForce_Right[j]])
                Forces = self.__ApplyForce([LcF, RcF])
                
                # compute forward dynamics
                self.Qdd = pin.aba(self.bolt.model, self.bolt.data, self.Q, self.Qd, TauPin, Forces)

                pin.forwardKinematics(self.bolt.model, self.bolt.data, self.Q, self.Qd, np.zeros(self.bolt.model.nv))
                a_pin = pin.getFrameAcceleration(self.bolt.model, self.bolt.data, 1, pin.ReferenceFrame.LOCAL).linear

                
                # update speed and position
                self.Qd += self.Qdd*self.dt
                self.Q = pin.integrate(self.bolt.model, self.Q, self.Qd*self.dt)

                
                dxx = np.linalg.norm(a_pin-a_mes)
                #print(f"LCF : {LcF}\n RCF : {RcF}\n a_pin : {a_pin}\n delta  : {dxx}")
                Deltas.append( np.linalg.norm(a_pin-a_mes))
                if Deltas[-1] < DeltaMin:
                    # error is minimal
                    DeltaMin = Deltas[-1]
                    CF = (LcF, RcF)
        return CF
    
    def ContactForces3d_(self, frames=[10,18]) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(3), np.zeros(3)

    def ContactForces3d(self, frames=[10,18]) -> tuple[np.ndarray, np.ndarray]:
        """ compute contact forces using jacobian and torques"""
        # set left foot and right foot data apart
        LF_id, RF_id = frames[0], frames[1]
        
        # adapt dimension of Torques
        TauPin = np.zeros(self.bolt.model.nv)
        TauPin[-6:] = self.Tau[:]

        # update data
        pin.forwardKinematics(self.bolt.model, self.data3D, self.Q)
        pin.updateFramePlacements(self.bolt.model, self.data3D)
        pin.computeAllTerms(self.bolt.model, self.data3D, self.Q, self.Qd)
        
        # compute jacobian matrixes for both feet
        JL = pin.computeFrameJacobian(self.bolt.model, self.data3D, self.Q, LF_id).copy()
        JR = pin.computeFrameJacobian(self.bolt.model, self.data3D, self.Q, RF_id).copy()
        # mass and generalized gravity
        M = self.data3D.M
        g = pin.computeGeneralizedGravity(self.bolt.model, self.data3D, self.Q)
        b = self.data3D.nle
      
        # compute 6d contact wrenches
        LcF = -np.linalg.pinv(JL.T) @ (TauPin - g - b)
        RcF = -np.linalg.pinv(JR.T) @ (TauPin - g - b)


        # compute contact wrench using method from ETH zurich, RD2017 p73 3.61
        
        """
        pin.computeJointJacobiansTimeVariation(self.bolt.model, self.bolt.data, self.Q, self.Qd)
        pin.forwardKinematics()
        JLd = pin.getJointJacobianTimeVariation(self.bolt.model, self.bolt.data, 3, pin.ReferenceFrame.WORLD)
        u = self.Qd
        
        t1 = JL @ np.linalg.pinv(M) @ JL.T
        t2 = JL @ np.linalg.pinv(M) @ (TauPin - g)
        t3 = JLd@u
        
        LcF = np.linalg.pinv(t1) @ (t2 + t3)     
        """

        
        return LcF[:3], RcF[:3]


    def ConsistencyChecker_DEPRECATED(self, ContactForce1d, ContactForce3d) -> tuple[float, float, float]:
        """ Check the consistency of a contact force computed by two functions."""
        # difference between both contact forces norms      
        Delta = np.abs(np.linalg.norm(ContactForce1d) - np.linalg.norm(ContactForce3d))
   
        Delta3d = np.linalg.norm(ContactForce1d - ContactForce3d)
        DeltaVertical = np.linalg.norm(ContactForce1d[2] - ContactForce3d[2])
        
        return Delta, Delta3d, DeltaVertical
    
    
    def ConsistencyChecker(self, ContactForce1d, ContactForce3d, ContactForceTorque, coeffs=[0.25, 0.25, 0.5]) -> tuple[float, float, float]:
        """ Check the consistency of a contact force computed by three functions."""
        # average contact force
        ContactForceAvg = coeffs[0]*ContactForce1d + coeffs[1]*ContactForce3d + coeffs[2]*ContactForceTorque
        
        # norm delta
        Delta = (np.abs(coeffs[0]* (np.linalg.norm(ContactForce1d) - np.linalg.norm(ContactForceAvg))) +\
                 np.abs(coeffs[1]* (np.linalg.norm(ContactForce3d) - np.linalg.norm(ContactForceAvg))) +\
                 np.abs(coeffs[2]* (np.linalg.norm(ContactForceTorque) - np.linalg.norm(ContactForceAvg))) )/3
        
        # vector delta
        Delta3d = ( coeffs[0]* np.linalg.norm(ContactForce1d - ContactForceAvg) +\
                    coeffs[1]* np.linalg.norm(ContactForce3d - ContactForceAvg) +\
                    coeffs[2]* np.linalg.norm(ContactForceTorque - ContactForceAvg) )/3
        
        # print(ContactForce1d)
        # print(ContactForce3d)
        # print(ContactForceTorque)
        # vertical delta    
        DeltaVertical = (coeffs[0]* np.linalg.norm(ContactForce1d[2] - ContactForceAvg[2]) +\
                    coeffs[1]* np.linalg.norm(ContactForce3d[2] - ContactForceAvg[2]) +\
                    coeffs[2]* np.linalg.norm(ContactForceTorque[2] - ContactForceAvg[2]) )/3
        
        return Delta, Delta3d, DeltaVertical
    
    
    def ContactProbability_Force(self, ContactForce1d, ContactForce3d, ContactForceT, trigger=5, stiffness=3, coeffs=[0.3, 0.3, 0.4]) -> float:
        """ compute probability of contact based on three force estimations """
        c1, c2, c3 = coeffs
        if c1+c2+c3 != 1.0 and self.Talkative: self.logger.LogTheLog("Coeff sum != 1 in ContactProbability_Force", style="warn")

        # average vertical force
        VerticalForce = (c1*ContactForce1d[2] + c2*ContactForce3d[2] + c3*ContactForceT[2])/3
        # return probability
        return self.__SigmoidDiscriminator(VerticalForce, trigger, stiffness)


    def ContactProbability_Force_3D(self, LeftContactForce3d, RightContactForce3d, iter, center=0.01, stiffness=0.1) -> float:
        """ compute probability of contact """
        if iter==0:
            self.Previous3D_L = []
            self.Previous3D_R = []
        
        LeftVerticalForce = LeftContactForce3d[2]
        RightVerticalForce = RightContactForce3d[2]
        self.Previous3D_L.append(LeftVerticalForce)
        self.Previous3D_R.append(RightVerticalForce)

        # check if force is increasing (contact) or decreasing (not in contact)
        deltaL = center
        deltaR = center
        if iter>=1 and iter < 20 :
            deltaL = self.Previous3D_L[-1] - self.Previous3D_L[-2]
            deltaR = self.Previous3D_R[-1] - self.Previous3D_R[-2]
        
        
        elif iter >= 20 :
            PreviousAverageL = sum(self.Previous3D_L[-20:-10])/10
            CurrentAverageL = sum(self.Previous3D_L[-10:])/10
            deltaL = CurrentAverageL - PreviousAverageL

            PreviousAverageR = sum(self.Previous3D_R[-20:-10])/10
            CurrentAverageR = sum(self.Previous3D_R[-10:])/10
            deltaR = CurrentAverageR - PreviousAverageR

        return self.__SigmoidDiscriminator(deltaL, center, stiffness), self.__SigmoidDiscriminator(deltaR, center, stiffness)

       
        
    def ContactProbability_Torque(self, Vertical, KneeTorque, KneeID = 8, FootID=10, Center=4, Stiffness=5)-> tuple[np.ndarray, float]:
        """ compute force based on knee torques, and return the estimated force and contact probability"""

        pin.forwardKinematics(self.bolt.model, self.dataT, self.Q)
        pin.updateFramePlacements(self.bolt.model, self.dataT)
        # normalized vertical vector
        Vertical = Vertical / np.linalg.norm(Vertical)
        
        # knee to foot vector (both expressed in world frame)
        delta = self.dataT.oMf[KneeID].translation - self.dataT.oMf[FootID].translation
        # norm of horizontal components
        hdist = np.linalg.norm(delta -  utils.scalar(delta, Vertical)*Vertical)
        if hdist < 0.05 or hdist > 0.25 : 
            self.logger.LogTheLog(f"Computed distance from knee : {KneeID} to foot : {FootID} is anormal :: {hdist} m")
            hdist = 0.12

        # compute force
        ContactForce = np.array([0, 0, KneeTorque / hdist])
        ContactProb = self.__SigmoidDiscriminator(ContactForce[2], Center, Stiffness)  
        return ContactForce, ContactProb
          
    
    def Slips(self, ContactForce3d, FootFrameID, MuTrigger=0.2, FootAcc=np.zeros(3), AccTrigger=2., mingler=1.) -> float:
        """ Compute a coefficient about wether or not foot is slipping
        Args :
        Return : an average (weighted with mingler) of computed slipping probabilities
        """   
        # slipery level : horizontal norm over vertical DISTANCE
        # can be < 0
        if ContactForce3d[2] == 0:
            Mu = 10
        else :
            Mu = np.sqrt(ContactForce3d[0]**2 + ContactForce3d[1]**2) / ContactForce3d[2]
        Slip0 = self.__SigmoidDiscriminator(Mu, MuTrigger, 4)

        # uses kinematics to compute foot horizontal speed
        
        # uses Kinematics to compute foot acceleration and determin if the feet is slipping
        HorizontalAcc = np.linalg.norm(FootAcc[:2])
        Slip1 = self.__SigmoidDiscriminator(HorizontalAcc, AccTrigger, 5)
        # 0 : feet stable , 0.8 : feet potentially slipping
        SlipProb = mingler * Slip0 + (1-mingler) * Slip1
        #print(Slip0, Slip1)
        #print(SlipProb)
  
        return SlipProb 
        
        
    def TrustContact(self,Delta, Delta3d, DeltaVertical, Mu, coeffs = (0.1, 0.2, 0.1, 0.1)) -> float:
        c1, c2, c3, c4 = coeffs
        Trust = 1.0 - ( c1*Delta + c2*Delta3d + c3*DeltaVertical + c4*Mu)
        return Trust


    def ContactBool(self, Contact, ContactProb, Trust, ProbThresold, TrustThresold, side) -> bool:
        """ Compute a contact boolean based on a contact probability"""
        # pseudo-integrator of probability
        memorysize = self.MemorySize
        if self.iter < memorysize :
            if side=="left":
                self.PastProbL = ContactProb
            if side=="right":
                self.PastProbR = ContactProb
        else :
            if side=="left":
                self.PastProbL = ( self.PastProbL*(memorysize-1) + ContactProb )/memorysize
                ContactProb = self.PastProbL
            if side=="right":
                self.PastProbR = ( self.PastProbR*(memorysize-1) + ContactProb )/memorysize
                ContactProb = self.PastProbR
        
        

        if ContactProb > ProbThresold :
            if ContactProb > 1.2*ProbThresold:
                # high probability of contact
                Contact = True
            elif Trust > TrustThresold:
                # mid probability but high trust
                Contact = True
            else :
                # mid probability and low trust
                Contact = Contact # no change to variable (keeping previous state)
        else :
            # low contact probability
            if Trust < TrustThresold:
                # low trust
                Contact=False
                #pass # no change
            else :
                # high trust
                Contact = False
        return Contact


    def LegsOnGround(self, Q, Qd, Acc, Torques, Vertical, TorqueForceMingler=0.0, ProbThresold=0.5, TrustThresold=0.5 ) -> tuple[bool, bool]:
        """Return wether or not left and right legs are in contact with the ground
        Args = 
        Return = 
        """
        # updates encoders and torques
        self.Q = Q.copy()
        self.Qd = Qd.copy()
        self.Tau = Torques.copy()

        # compute probability of contact based on knee's torque
        self.LcF_T, self.ContactProbL_T = self.ContactProbability_Torque(Vertical=Vertical, 
                                                                         KneeTorque=self.Tau[self.LeftKneeTorqueID], 
                                                                         KneeID=self.LeftKneeFrameID, 
                                                                         FootID=self.LeftFootFrameID, 
                                                                         Center=4, Stiffness=2)
        self.RcF_T, self.ContactProbR_T = self.ContactProbability_Torque(Vertical=Vertical, 
                                                                         KneeTorque=self.Tau[self.RightKneeTorqueID], 
                                                                         KneeID=self.RightKneeFrameID, 
                                                                         FootID=self.RightFootFrameID, 
                                                                         Center=4, Stiffness=2)
        # get the contact forces # TODO : debug 1D
        self.LcF_1d, self.RcF_1d = np.zeros(3), np.zeros(3) #self.ContactForces1d(Torques=Torques, Q=Q, Qd=Qd, BaseAccelerationIMU=Acc , Dynamic=0.4, Resolution=7)
        # print("1D done", self.Q)
        self.LcF_3d, self.RcF_3d = self.ContactForces3d(frames=[self.LeftFootFrameID, self.RightFootFrameID])
        # print("3D done", self.Q)
        # get deltas
        self.DeltaL, self.DeltaL3d, self.DeltaLV = self.ConsistencyChecker(self.LcF_1d, self.LcF_3d, self.LcF_T, coeffs=self.coeffs)
        self.DeltaR, self.DeltaR3d, self.DeltaRV = self.ConsistencyChecker(self.RcF_1d, self.RcF_3d, self.RcF_T, coeffs=self.coeffs)

        # get probability of slip
        LFootAcc = np.zeros(3) # TODO : mod
        RFootAcc = np.zeros(3)
        self.SlipProbL = self.Slips(self.LcF_3d, self.LeftFootFrameID, MuTrigger=0.2, FootAcc=LFootAcc, AccTrigger=2., mingler=0.99)
        self.SlipProbR = self.Slips(self.RcF_3d, self.RightFootFrameID, MuTrigger=0.2, FootAcc=RFootAcc, AccTrigger=2., mingler=0.99)

        # compute probability of contact based on vertical contact forces from 3 DIFFERENT ESTIMATIONS
        
        #self.ContactProbL_F = self.ContactProbability_Force(self.LcF_1d, self.LcF_3d, self.LcF_T, trigger=4, stiffness=3, coeffs=[0., 1.0, 0.])
        #self.ContactProbR_F = self.ContactProbability_Force(self.RcF_1d, self.RcF_3d, self.RcF_T, trigger=4, stiffness=3, coeffs=[0., 1.0, 0.])
        
        self.ContactProbL_F, self.ContactProbR_F = self.ContactProbability_Force_3D(self.LcF_3d, self.RcF_3d, iter=self.iter, center=0.01, stiffness=0.03)
        
        # merge probability of contact from force and torque
        self.ContactProbL = self.ContactProbL_T * TorqueForceMingler + self.ContactProbL_F * (1-TorqueForceMingler)
        self.ContactProbR = self.ContactProbR_T * TorqueForceMingler + self.ContactProbR_F * (1-TorqueForceMingler)


        # compute trust in contact
        self.TrustL = self.TrustContact(self.DeltaL, self.DeltaL3d, self.DeltaLV, self.MuL, coeffs=(0.1, 0.1, 0.2, 0.1))
        self.TrustR = self.TrustContact(self.DeltaR, self.DeltaR3d, self.DeltaRV, self.MuR, coeffs=(0.1, 0.1, 0.2, 0.1))
        
        self.LeftContact = self.ContactBool(self.LeftContact, self.ContactProbL, self.TrustL, ProbThresold, TrustThresold, "left")
        self.RightContact = self.ContactBool(self.RightContact, self.ContactProbR, self.TrustR, ProbThresold, TrustThresold, "right")
         
        # log contact forces and deltas and contact
        if self.Logging : self.UpdateLogMatrixes()
        
        return self.LeftContact, self.RightContact





    def LegsOnGroundKin(self, Kinpos, vertical) -> tuple[bool, bool]:
        # return iwether or not left and right legs are in contact with the ground
        # uses the z distance from foot to base, with kinematics only and a vertical direction provided by IMU
        
        self.bolt.framesForwardKinematics(q=Kinpos)
        self.bolt.updateFramePlacements()

        LeftFootPos = self.bolt.oMf[self.LeftFootFrameID].translation
        RightFootPos = self.bolt.data.oMf[self.RightFootFrameID].translation
        
        VerticalDistLeft = utils.scalar(LeftFootPos, vertical)
        VerticalDistRight = utils.scalar(RightFootPos, vertical)

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
