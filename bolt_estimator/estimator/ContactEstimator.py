import numpy as np
import pinocchio as pin
from bolt_estimator.utils.Bolt_Utils import utils


class contactEstimator():
    def __init__(self, 
                robot, 
                left_foot_frame_id  : int=10, 
                right_foot_frame_id : int=18, 
                left_knee_frame_id  : int=4,
                right_knee_frame_id : int=8,
                left_knee_torque_id : int=2,
                right_knee_torque_id : int=5,
                iter_number       : int=1000,
                dt               : float=1e-3,
                memory_size       : int=5,
                logging          : bool=True,
                talkative        : bool=True,
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
        self.left_foot_frame_id = leftfoot_frame_id
        self.right_foot_frame_id = rightfoot_frame_id
        self.C_id = [leftfoot_frame_id, rightfoot_frame_id]
        # knees id
        self.left_knee_frame_id = LeftKneeFrameID
        self.right_knee_frame_id = RightKneeFrameID
        self.left_knee_torque_id = Leftknee_torqueID
        self.right_knee_torque_id = right_knee_torque_id
        
        # logging options
        if logger is None :
            self.talkative=False
        else :
            self.talkative=talkative
            self.logger = logger
            self.logger.LogTheLog("contact forces Estimator started", style="subtitle")
        self.logging = logging

        # compute bolt mass
        self.mass = pin.computeTotalMass(self.bolt.model)
        self.logger.LogTheLog("Mass of robot : " + str(self.mass), style="subinfo")
        
        # initialize all variables
        self.iter_number = iter_number
        self.InitVariables()
        if self.logging : self.InitLogMatrixes()
        self.iter = 0

        self.Torquecontacts = [False, False]

        # average contact probability over the last k=memory_size results
        self.memory_size = memory_size
        
        # coefficient for averaging contact forces
        self.coeffs = [0.0, 0.3, 0.7] #1D, 3D, torques-based
        if sum(self.coeffs) != 1 and self.talkative: self.logger.LogTheLog("Coeffs sum not equal to 1", "warn")
        self.c1, self.c2, self.c3 = self.coeffs
    
    
    
    
    def InitLogMatrixes(self) -> None:
        
        # contact forces logs
        self.log_lc_f_3d = np.zeros([3, self.iter_number])
        self.log_lc_f_1d = np.zeros([3, self.iter_number])
        self.log_rc_f_3d = np.zeros([3, self.iter_number])
        self.log_rc_f_1d = np.zeros([3, self.iter_number])
        self.log_lc_f_t = np.zeros([3, self.iter_number])
        self.log_rc_f_t = np.zeros([3, self.iter_number])
        # left and right trust in contacts logs
        self.log_delta1_d = np.zeros([2, self.iter_number])
        self.log_delta3_d = np.zeros([2, self.iter_number])
        self.log_delta_vertical = np.zeros([2, self.iter_number])
        # slipping logs
        self.log_mu = np.zeros([2, self.iter_number])
        self.log_slip_prob = np.zeros([2, self.iter_number])
        # boolean contact, probability of contact and trust in contact
        self.log_contact = np.zeros([2, self.iter_number], dtype=bool)
        self.log_contact_probability = np.zeros([2, self.iter_number])
        self.log_contact_probability_f = np.zeros([2, self.iter_number])
        self.log_contact_probability_T = np.zeros([2, self.iter_number])
        self.log_trust = np.zeros([2, self.iter_number])
        return None
    

    def UpdateLogMatrixes(self) -> None:
        log_iter = self.iter
        if self.iter >= self.iter_number:
            # Logs matrices' size will not be sufficient
            self.logger.LogTheLog("Excedind planned number of executions for contactEstimator, iter_number = " + str(self.iter_number), style="warn", ToPrint=self.talkative)
            log_iter = self.iter_number-1

        # contact forces logs
        self.log_lc_f_3d[:, log_iter] = self.LcF_3d[:]
        self.log_lc_f_1d[:, log_iter] = self.LcF_1d[:]
        self.log_rc_f_3d[:, log_iter] = self.RcF_3d[:]
        self.log_rc_f_1d[:, log_iter] = self.RcF_1d[:]
        self.log_lc_f_t[:, log_iter] = self.LcF_T[:]
        self.log_rc_f_t[:, log_iter] = self.RcF_T[:]
        # left and right trust in contacts logs
        self.log_delta1_d[:, log_iter] = [self.delta_l, self.delta_r]
        self.log_delta3_d[:, log_iter] = [self.delta_l3d, self.delta_r3d]
        self.log_delta_vertical[:, log_iter] = [self.delta_lV, self.delta_rV]
        # slipping logs
        self.log_mu[:, log_iter] = [self.mu_l, self.mu_r]
        self.log_slip_prob[:, log_iter] = [self.slip_probL, self.slip_probR]
        # boolean contact, probability of contact and trust in contact
        self.log_contact[:, log_iter] = [self.left_contact, self.right_contact]
        self.log_contact_probability[:, log_iter] = [self.contact_prob_l, self.contact_prob_r]
        self.log_contact_probability_f[:, log_iter] = [self.contact_prob_l_F, self.contact_prob_r_F]
        self.log_contact_probability_T[:, log_iter] = [self.contact_prob_l_T, self.contact_prob_r_T]
        self.log_trust[:, log_iter] = [self.trust_l, self.trust_r]
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
            return self.log_lc_f_1d, self.log_rc_f_1d
        elif data=='cf_3d':
            return self.log_lc_f_3d, self.log_rc_f_3d
        elif data=='cf_torques':
            return self.log_lc_f_t, self.log_rc_f_t
        
        # trusts and co
        elif data=='delta_1d':
            return self.log_delta1_d
        elif data=='delta_3d':
            return self.log_delta3_d
        elif data=='mu':
            return self.log_mu
        elif data=='slip_prob':
            return self.log_slip_prob
        elif data=="trust":
            return self.log_trust
        
        # bool log
        elif data=="contact_bool":
           return self.log_contact
        elif data=="contact_prob":
           return self.log_contact_probability
        elif data=="contact_prob_force":
           return self.log_contact_probability_f
        elif data=="contact_prob_torque":
           return self.log_contact_probability_T
                
        
        
        
        
    def InitVariables(self) -> None:
        # deltas and slips
        self.delta_l, self.delta_l3d, self.delta_lV, self.mu_l, self.trust_l = 0., 0., 0., 0., 0.
        self.delta_r, self.delta_r3d, self.delta_rV, self.mu_r, self.trust_r = 0., 0., 0., 0., 0.
        self.slip_probL = 0.
        self.slip_probR = 0.
        # contact and trust
        self.contact_prob_l = 0.
        self.contact_prob_r = 0.
        self.trust_l = 0.
        self.trust_r = 0.
        self.left_contact = False
        self.right_contact = False

        # contact forces
        self.LcF_1d = np.zeros(3)
        self.RcF_1d = np.zeros(3)
        self.LcF_3d = np.zeros(3)
        self.RcF_3d = np.zeros(3)
        self.LcF_T = np.zeros(3)
        self.RcF_T = np.zeros(3)

        # foot position
        self.right_foot_pos = np.zeros(3)
        self.left_foot_pos = np.zeros(3)

        # contact memory
        self.past_prob_l = 0.
        self.past_prob_r = 0.
     
        
    def __SigmoidDiscriminator(self, x, center, stiffness=5) -> np.ndarray:
        """Bring x data between 0 and 1, such that P(x=center)=0.5. The greater stiffness, the greater dP/dx (esp. around x=center)"""
        b0 = stiffness
        b = b0/center
        if b*x < -100 :
            return 1
        return 1/ (1 + np.exp(-b*x + b0))
    
    def __ApplyForce(self, contact_forces):
        """ return forces in the right fomat for pinocchio """
        # volé à victor
        ### Build force list for ABA
        forces = [ pin.Force.Zero() for _ in self.bolt.model.joints ]
        # I am supposing here that all contact frames are on separate joints. This is asserted below:
        #assert( len( set( [ cmodel.frames[idf].parentJoint for idf in contactIds ]) ) == len(contactIds) )
        
        for f,idf in zip(contact_forces,self.C_id):
            # contact forces introduced in ABA as spatial forces at joint frame.
            forces[self.bolt.model.frames[idf].parent] = self.bolt.model.frames[idf].placement * pin.Force(f, 0.*f)
        forces_out = pin.StdVec_Force()
        for f in forces:
            forces_out.append(f)
        return forces_out
    



    
    def contact_forces1d(self, torques, Q, Qd, base_acceleration_imu, dynamic=0.4, resolution=10) -> tuple[np.ndarray, np.ndarray]:
        """this function assumes that the weight is somehow distributed over both feet, and search an approximation of that distribution"""
        # NO EXTERNAL forces
        # contact forces ASSUMED vertical
        a_mes = base_acceleration_imu
        deltas = []
        delta_min = 1e6 #np.linalg.norm(a_mes)
        CF = (np.zeros(3), np.zeros(3))
        

        # update data from encoders
        self.Q[:] = Q[:]
        self.Qd[:] = Qd[:]
        tau_pin = np.zeros(self.bolt.model.nv)
        tau_pin[-6:] = torques[:]
        
        # dynamic states that contact forces sum is in [weight * (1-dynamic) , weight * (1+dynamic)]
        # minimal contact force is 0
        min_force = 0
        # maximal contact force is weight * (1+dynamic)
        max_force = self.mass*9.81 * (1+dynamic)
        # the lower and upper indices such that contact forces sum is in [weight * (1-dynamic) , weight * (1+dynamic)]
        upper = int(np.floor(resolution*(1+dynamic)))
        lower = int(np.ceil(resolution*(1-dynamic/2)))
        #print(f'upper : {upper}, lower : {lower}')
        # possible force range from min to max force
        possible_force_left = np.linspace(min_force, max_force, upper)
        possible_force_right = possible_force_left.copy()        
        
        for i in range(upper):
            # left contact force takes all possible values
            # for a given value of the left contact force, contact forces sum is in [weight * (1-dynamic) , weight * (1+dynamic)]
            reasonnable_min = max(0, lower-i)
            reasonnable_max = max(0, upper-i)
            for j in range( reasonnable_min, reasonnable_max):
                #print(f'Indices :  {i} ; {j}           Répartition :  {np.floor(possible_force_left[i])}/{np.floor(possible_force_right[j])}')
                
                # both our contact forces are within reasonnable range. Looking for the most likely forces repartition.
                # contact forces ASSUMED vertical
                LcF = np.array([0, 0, possible_force_left[i]])
                RcF = np.array([0, 0, possible_force_right[j]])
                forces = self.__ApplyForce([LcF, RcF])
                
                # compute forward dynamics
                self.Qdd = pin.aba(self.bolt.model, self.bolt.data, self.Q, self.Qd, tau_pin, forces)

                pin.forwardKinematics(self.bolt.model, self.bolt.data, self.Q, self.Qd, np.zeros(self.bolt.model.nv))
                a_pin = pin.getFrameacceleration(self.bolt.model, self.bolt.data, 1, pin.ReferenceFrame.LOCAL).linear

                
                # update speed and position
                self.Qd += self.Qdd*self.dt
                self.Q = pin.integrate(self.bolt.model, self.Q, self.Qd*self.dt)

                
                dxx = np.linalg.norm(a_pin-a_mes)
                #print(f"LCF : {LcF}\n RCF : {RcF}\n a_pin : {a_pin}\n delta  : {dxx}")
                deltas.append( np.linalg.norm(a_pin-a_mes))
                if deltas[-1] < delta_min:
                    # error is minimal
                    delta_min = deltas[-1]
                    CF = (LcF, RcF)
        return CF
    
    def contact_forces3d(self, frames=[10,18]) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(3), np.zeros(3)

    def contact_forces3d_(self, frames=[10,18]) -> tuple[np.ndarray, np.ndarray]:
        """ compute contact forces using jacobian and torques"""
        # set left foot and right foot data apart
        LF_id, RF_id = frames[0], frames[1]
        
        # adapt dimension of torques
        tau_pin = np.zeros(self.bolt.model.nv)
        tau_pin[-6:] = self.Tau[:]

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
        LcF = -np.linalg.pinv(JL.T) @ (tau_pin - g - b)
        RcF = -np.linalg.pinv(JR.T) @ (tau_pin - g - b)


        # compute contact wrench using method from ETH zurich, RD2017 p73 3.61
        
        """
        pin.computeJointJacobiansTimeVariation(self.bolt.model, self.bolt.data, self.Q, self.Qd)
        pin.forwardKinematics()
        JLd = pin.getJointJacobianTimeVariation(self.bolt.model, self.bolt.data, 3, pin.ReferenceFrame.WORLD)
        u = self.Qd
        
        t1 = JL @ np.linalg.pinv(M) @ JL.T
        t2 = JL @ np.linalg.pinv(M) @ (tau_pin - g)
        t3 = JLd@u
        
        LcF = np.linalg.pinv(t1) @ (t2 + t3)     
        """

        
        return LcF[:3], RcF[:3]


    def ConsistencyChecker_DEPRECATED(self, contact_force1d, contact_force3d) -> tuple[float, float, float]:
        """ Check the consistency of a contact force computed by two functions."""
        # difference between both contact forces norms      
        delta = np.abs(np.linalg.norm(contact_force1d) - np.linalg.norm(contact_force3d))
   
        delta3d = np.linalg.norm(contact_force1d - contact_force3d)
        delta_vertical = np.linalg.norm(contact_force1d[2] - contact_force3d[2])
        
        return delta, delta3d, delta_vertical
    
    
    def ConsistencyChecker(self, contact_force1d, contact_force3d, contact_force_torque, coeffs=[0.25, 0.25, 0.5]) -> tuple[float, float, float]:
        """ Check the consistency of a contact force computed by three functions."""
        # average contact force
        contact_force_avg = coeffs[0]*contact_force1d + coeffs[1]*contact_force3d + coeffs[2]*contact_force_torque
        
        # norm delta
        delta = (np.abs(coeffs[0]* (np.linalg.norm(contact_force1d) - np.linalg.norm(contact_force_avg))) +\
                 np.abs(coeffs[1]* (np.linalg.norm(contact_force3d) - np.linalg.norm(contact_force_avg))) +\
                 np.abs(coeffs[2]* (np.linalg.norm(contact_force_torque) - np.linalg.norm(contact_force_avg))) )/3
        
        # vector delta
        delta3d = ( coeffs[0]* np.linalg.norm(contact_force1d - contact_force_avg) +\
                    coeffs[1]* np.linalg.norm(contact_force3d - contact_force_avg) +\
                    coeffs[2]* np.linalg.norm(contact_force_torque - contact_force_avg) )/3
        
        # print(contact_force1d)
        # print(contact_force3d)
        # print(contact_force_torque)
        # vertical delta    
        delta_vertical = (coeffs[0]* np.linalg.norm(contact_force1d[2] - contact_force_avg[2]) +\
                    coeffs[1]* np.linalg.norm(contact_force3d[2] - contact_force_avg[2]) +\
                    coeffs[2]* np.linalg.norm(contact_force_torque[2] - contact_force_avg[2]) )/3
        
        return delta, delta3d, delta_vertical
    
    
    def contact_probability_Force(self, contact_force1d, contact_force3d, contact_force_t, trigger=5, stiffness=3, coeffs=[0.3, 0.3, 0.4]) -> float:
        """ compute probability of contact based on three force estimations """
        c1, c2, c3 = coeffs
        if c1+c2+c3 != 1.0 and self.talkative: self.logger.LogTheLog("Coeff sum != 1 in contact_probability_Force", style="warn")

        # average vertical force
        verticalForce = (c1*contact_force1d[2] + c2*contact_force3d[2] + c3*contact_force_t[2])/3
        # return probability
        return self.__SigmoidDiscriminator(verticalForce, trigger, stiffness)


    def contact_probability_Force_3D(self, left_contact_force3d, right_contact_force3d, iter, center=0.01, stiffness=0.1) -> float:
        """ compute probability of contact """
        if iter==0:
            self.previous3_d_l = []
            self.previous3_d_r = []
        
        left_vertical_force = left_contact_force3d[2]
        right_vertical_force = right_contact_force3d[2]
        self.previous3_d_l.append(left_vertical_force)
        self.previous3_d_r.append(right_vertical_force)

        # check if force is increasing (contact) or decreasing (not in contact)
        delta_l = center
        delta_r = center
        if iter>=1 and iter < 20 :
            delta_l = self.previous3_d_l[-1] - self.previous3_d_l[-2]
            delta_r = self.previous3_d_r[-1] - self.previous3_d_r[-2]
        
        
        elif iter >= 20 :
            previous_average_l = sum(self.previous3_d_l[-20:-10])/10
            current_average_l = sum(self.previous3_d_l[-10:])/10
            delta_l = current_average_l - previous_average_l

            previous_average_r = sum(self.previous3_d_r[-20:-10])/10
            current_average_r = sum(self.previous3_d_r[-10:])/10
            delta_r = current_average_r - previous_average_r

        return self.__SigmoidDiscriminator(delta_l, center, stiffness), self.__SigmoidDiscriminator(delta_r, center, stiffness)

       
        
    def contactProbability_Torque(self, vertical, knee_torque, knee_id = 8, foot_id=10, center=4, stiffness=5)-> tuple[np.ndarray, float]:
        """ compute force based on knee torques, and return the estimated force and contact probability"""

        pin.forwardKinematics(self.bolt.model, self.dataT, self.Q)
        pin.updateFramePlacements(self.bolt.model, self.dataT)
        # normalized vertical vector
        vertical = vertical / np.linalg.norm(vertical)
        
        # knee to foot vector (both expressed in world frame)
        delta = self.dataT.oMf[knee_id].translation - self.dataT.oMf[foot_id].translation
        # norm of horizontal components
        hdist = np.linalg.norm(delta -  utils.scalar(delta, vertical)*vertical)
        if hdist < 0.05 or hdist > 0.25 : 
            self.logger.LogTheLog(f"Computed distance from knee : {knee_id} to foot : {foot_id} is anormal :: {hdist} m")
            hdist = 0.12

        # compute force
        contact_force = np.array([0, 0, knee_torque / hdist])
        contact_prob = self.__SigmoidDiscriminator(contact_force[2], center, stiffness)  
        return contact_force, contact_prob
          
    
    def Slips(self, contact_force3d, foot_frame_id, mu_trigger=0.2, foot_acc=np.zeros(3), acc_trigger=2., mingler=1.) -> float:
        """ Compute a coefficient about wether or not foot is slipping
        Args :
        Return : an average (weighted with mingler) of computed slipping probabilities
        """   
        # slipery level : horizontal norm over vertical DISTANCE
        # can be < 0
        if contact_force3d[2] == 0:
            Mu = 10
        else :
            Mu = np.sqrt(contact_force3d[0]**2 + contact_force3d[1]**2) / contact_force3d[2]
        slip0 = self.__SigmoidDiscriminator(Mu, mu_trigger, 4)

        # uses kinematics to compute foot horizontal speed
        
        # uses Kinematics to compute foot acceleration and determin if the feet is slipping
        horizontal_acc = np.linalg.norm(foot_acc[:2])
        slip1 = self.__SigmoidDiscriminator(horizontal_acc, acc_trigger, 5)
        # 0 : feet stable , 0.8 : feet potentially slipping
        slip_prob = mingler * slip0 + (1-mingler) * slip1
        #print(slip0, slip1)
        #print(slip_prob)
  
        return slip_prob 
        
        
    def trustcontact(self,delta, delta3d, delta_vertical, Mu, coeffs = (0.1, 0.2, 0.1, 0.1)) -> float:
        c1, c2, c3, c4 = coeffs
        trust = 1.0 - ( c1*delta + c2*delta3d + c3*delta_vertical + c4*Mu)
        return trust


    def contactBool(self, contact, contact_prob, trust, prob_thresold, trust_thresold, side) -> bool:
        """ Compute a contact boolean based on a contact probability"""
        # pseudo-integrator of probability
        memory_size = self.memory_size
        if self.iter < memory_size :
            if side=="left":
                self.past_prob_l = contact_prob
            if side=="right":
                self.past_prob_r = contact_prob
        else :
            if side=="left":
                self.past_prob_l = ( self.past_prob_l*(memory_size-1) + contact_prob )/memory_size
                contact_prob = self.past_prob_l
            if side=="right":
                self.past_prob_r = ( self.past_prob_r*(memory_size-1) + contact_prob )/memory_size
                contact_prob = self.past_prob_r
        
        

        if contact_prob > prob_thresold :
            if contact_prob > 1.2*prob_thresold:
                # high probability of contact
                contact = True
            elif trust > trust_thresold:
                # mid probability but high trust
                contact = True
            else :
                # mid probability and low trust
                contact = contact # no change to variable (keeping previous state)
        else :
            # low contact probability
            if trust < trust_thresold:
                # low trust
                contact=False
                #pass # no change
            else :
                # high trust
                contact = False
        return contact


    def LegsOnGround(self, Q, Qd, acc, torques, vertical, torque_force_mingler=0.0, prob_thresold=0.5, trust_thresold=0.5 ) -> tuple[bool, bool]:
        """Return wether or not left and right legs are in contact with the ground
        Args = 
        Return = 
        """
        # updates encoders and torques
        self.Q = Q.copy()
        self.Qd = Qd.copy()
        self.Tau = torques.copy()

        # compute probability of contact based on knee's torque
        self.LcF_T, self.contact_prob_l_T = self.contact_probability_Torque(vertical=vertical, 
                                                                         knee_torque=self.Tau[self.Leftknee_torqueID], 
                                                                         knee_id=self.LeftKneeFrameID, 
                                                                         foot_id=self.leftfoot_frame_id, 
                                                                         center=4, stiffness=2)
        self.RcF_T, self.contact_prob_r_T = self.contact_probability_Torque(vertical=vertical, 
                                                                         knee_torque=self.Tau[self.Rightknee_torqueID], 
                                                                         knee_id=self.RightKneeFrameID, 
                                                                         foot_id=self.rightfoot_frame_id, 
                                                                         center=4, stiffness=2)
        # get the contact forces # TODO : debug 1D
        self.LcF_1d, self.RcF_1d = np.zeros(3), np.zeros(3) #self.contact_forces1d(torques=torques, Q=Q, Qd=Qd, base_acceleration_imu=acc , dynamic=0.4, resolution=7)
        # print("1D done", self.Q)
        self.LcF_3d, self.RcF_3d = self.contact_forces3d(frames=[self.leftfoot_frame_id, self.rightfoot_frame_id])
        # print("3D done", self.Q)
        # get deltas
        self.delta_l, self.delta_l3d, self.delta_lV = self.ConsistencyChecker(self.LcF_1d, self.LcF_3d, self.LcF_T, coeffs=self.coeffs)
        self.delta_r, self.delta_r3d, self.delta_rV = self.ConsistencyChecker(self.RcF_1d, self.RcF_3d, self.RcF_T, coeffs=self.coeffs)

        # get probability of slip
        Lfoot_acc = np.zeros(3) # TODO : mod
        Rfoot_acc = np.zeros(3)
        self.slip_probL = self.Slips(self.LcF_3d, self.leftfoot_frame_id, mu_trigger=0.2, foot_acc=Lfoot_acc, acc_trigger=2., mingler=0.99)
        self.slip_probR = self.Slips(self.RcF_3d, self.rightfoot_frame_id, mu_trigger=0.2, foot_acc=Rfoot_acc, acc_trigger=2., mingler=0.99)

        # compute probability of contact based on vertical contact forces from 3 DIFFERENT ESTIMATIONS
        
        #self.contact_prob_l_F = self.contact_probability_Force(self.LcF_1d, self.LcF_3d, self.LcF_T, trigger=4, stiffness=3, coeffs=[0., 1.0, 0.])
        #self.contact_prob_r_F = self.contact_probability_Force(self.RcF_1d, self.RcF_3d, self.RcF_T, trigger=4, stiffness=3, coeffs=[0., 1.0, 0.])
        
        self.contact_prob_l_F, self.contact_prob_r_F = self.contact_probability_Force_3D(self.LcF_3d, self.RcF_3d, iter=self.iter, center=0.01, stiffness=0.03)
        
        # merge probability of contact from force and torque
        self.contact_prob_l = self.contact_prob_l_T * torque_force_mingler + self.contact_prob_l_F * (1-torque_force_mingler)
        self.contact_prob_r = self.contact_prob_r_T * torque_force_mingler + self.contact_prob_r_F * (1-torque_force_mingler)


        # compute trust in contact
        self.trust_l = self.trustcontact(self.delta_l, self.delta_l3d, self.delta_lV, self.mu_l, coeffs=(0.1, 0.1, 0.2, 0.1))
        self.trust_r = self.trustcontact(self.delta_r, self.delta_r3d, self.delta_rV, self.mu_r, coeffs=(0.1, 0.1, 0.2, 0.1))
        
        self.left_contact = self.contactBool(self.left_contact, self.contact_prob_l, self.trust_l, prob_thresold, trust_thresold, "left")
        self.right_contact = self.contactBool(self.right_contact, self.contact_prob_r, self.trust_r, prob_thresold, trust_thresold, "right")
         
        # log contact forces and deltas and contact
        if self.logging : self.UpdateLogMatrixes()
        
        return self.left_contact, self.right_contact





    def LegsOnGroundKin(self, kinpos, vertical) -> tuple[bool, bool]:
        # return iwether or not left and right legs are in contact with the ground
        # uses the z distance from foot to base, with kinematics only and a vertical direction provided by IMU
        
        self.bolt.framesForwardKinematics(q=kinpos)
        self.bolt.updateFramePlacements()

        left_foot_pos = self.bolt.oMf[self.leftfoot_frame_id].translation
        right_foot_pos = self.bolt.data.oMf[self.rightfoot_frame_id].translation
        
        vertical_dist_left = utils.scalar(left_foot_pos, vertical)
        vertical_dist_right = utils.scalar(right_foot_pos, vertical)

        if pin.isapprox(vertical_dist_left, vertical_dist_right, eps=3e-3):
            # both feet might be in contact
            left_contact, right_contact = True, True
        elif vertical_dist_left > vertical_dist_right :
            # Left foot is much further away than right foot, and therefore should be in contact with the ground
            left_contact, right_contact = True, False
        else : 
            # Right foot is much further away than right foot, and therefore should be in contact with the ground
            left_contact, right_contact = False, True
        return left_contact, right_contact


    def FeetPositions(self) -> tuple[np.ndarray, np.ndarray]:
        # parce que Constant en a besoin
        return self.left_foot_pos, self.right_foot_pos
