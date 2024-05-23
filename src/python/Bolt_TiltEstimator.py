import numpy as np
import pinocchio as pin


"""
Created on Wed May 22 13:41:35 2024

@author: nalbrecht
"""


class TiltEstimator():
    
    def __init__(self,
                 robot,
                 Q0,
                 Qd0,
                 Niter)-> None:
        
        # iter 
        self.iter = 0
        self.Niter = Niter
        # pinocchio
        self.Q = Q0
        self.Qd = Qd0
        self.bolt = robot
        self.data = self.bolt.model.createData()
        pin.forwardKinematics(self.bolt.model, self.data, self.Q)
        pin.updateFramePlacements(self.bolt.model, self.data)
        
        # state variables
        self.x1_hat = np.zeros(3)       # TODO : value of init
        self.x1_hat_dot = np.zeros(3)
        self.x2_hat = np.array([0, 0, 1.])
        self.x2_hat_dot = np.array([0, 0, 1.])
        self.x2_prime = np.array([0, 0, 1.])
        self.x2_prime_dot = np.array([0, 0, 1.])
        
        # errors
        self.x1_tilde = np.zeros(3)
        self.x2_tilde = np.zeros(3)
        self.x2_tildeprime = np.zeros(3)
        
        # utilities
        self.c_R_l = np.zeros((3,3))
        
        
        # constants
        self.g0 = 9.81
        
        # logs
        self.InitLogs()

        return None
    
    
    def InitLogs(self)-> None :
        """ create log matrixes"""
        # state variables
        self.x1_hat_logs = np.zeros((self.Niter,3))
        self.x2_hat_logs = np.zeros((self.Niter, 3, 3))
        self.x2_prime_logs = np.zeros((self.Niter, 3, 3))
        
        # errors
        self.x1_tilde_logs = np.zeros((self.Niter,3))
        self.x2_tilde_logs = np.zeros((self.Niter, 3, 3))
        self.x2_tildeprime_logs = np.zeros((self.Niter, 3, 3))
        
        return None
    
    
    def UpdateLogs(self) -> None:
        """ update log matrixes"""
        return None
    
    
    def S(self, x:np.ndarray) -> np.ndarray:
        """ Skew-symetric operator """
        sx = np.array([[0,    -x[2],  x[1]],
                       [x[2],   0,   -x[0]],
                       [-x[1], x[0],    0 ]])
        return sx
    
    
    def GetYV_v1(self) -> np.ndarray:
        """ Estimate Yv and return it """
        
        yv = self.c_R_l.T @ self.c_Pdot_l + self.S(self.yg - self.c_Omega_l) @ self.c_R_l.T @ self.c_P_l
        return yv
    
    def GetYV_v2(self, LFootID, RFootID, BaseID, eta) -> np.ndarray:
        """ Estimate Yv and return it """
        c_P_anchor = eta *  self.data.oMf[LFootID].translation + (1-eta) * self.data.oMf[RFootID].translation
        v1 = pin.getFrameVelocity(self.bolt.model, self.data, LFootID)
        v2 = pin.getFrameVelocity(self.bolt.model, self.data, RFootID)
        c_Pdot_anchor = eta *  v1.translation + (1-eta) * v2.translation
        yv = -self.S(self.yg) @ c_P_anchor - c_Pdot_anchor
        return yv
    
    def PinocchioUpdate(self, BaseID, ContactFootID, dt) -> None:
        """ Update pinocchio data with forward kinematics and update FK variables"""
        
        # update pinocchio data
        pin.forwardKinematics(self.bolt.model, self.data, self.Q)
        pin.updateFramePlacements(self.bolt.model, self.data)
        pin.computeAllTerms(self.bolt.model, self.data, self.Q, self.Qd)
        
        # update relevant data
        # rotation matrix of base frame in contact foot frame
        self.prev_c_R_l = self.c_R_l.copy()
        self.c_R_l = np.array(self.data.oMf[BaseID].rotation.copy()) @ np.array(self.data.oMf[ContactFootID].rotation.copy()).T

        self.c_Rdot_l =  (self.c_R_l - self.prev_c_R_l) / dt # TODO : chk
        # position of base frame in contact foot frame
        self.c_P_l = np.array(self.data.oMf[BaseID].translation) - np.array(self.data.oMf[ContactFootID].translation)

        self.c_Pdot_l = pin.getFrameVelocity(self.bolt.model, self.data, ContactFootID).linear
        # rotation speed of base frame in contact foot frame
        self.c_Omega_l = pin.getFrameVelocity(self.bolt.model, self.data, BaseID).angular
        
        
        return None
    
    
    def ErrorUpdate(self, alpha1:float, alpha2:float, gamma:float, dt:float) -> None:
        """ Compute error in estimation and update error variables"""
        S2 = np.zeros(3) # TODO : upd

        x1_tilde_dot      = -self.S(self.yg) @ self.x1_tilde - alpha1 * self.x1_tilde - self.g0*self.x2_tildeprime
        x2_tildeprime_dot = -self.S(self.yg) @ self.x2_tilde - alpha2/self.g0 * self.x1_tilde
        x2_tilde_dot      = -self.S(self.yg) @ self.x2_tilde - gamma * S2 @ (self.x2_tilde - self.x2_tildeprime)
        
        self.x1_tilde += x1_tilde_dot * dt
        self.x2_tilde += x2_tilde_dot * dt
        self.x2_tildeprime += x2_tildeprime_dot * dt
        
        return None

    
    
    def Estimate(self, 
                  Q:np.ndarray,
                  Qd:np.ndarray,
                  BaseID:int,
                  ContactFootID:int,
                  ya:np.ndarray, 
                  yg:np.ndarray, 
                  dt:float, 
                  alpha1:float, alpha2:float, gamma:float) -> tuple[np.ndarray, np.ndarray] : # TODO : régler coeffs
        """ update state variables and return current tilt estimate """
        
        # store measurement
        self.ya = ya
        self.yg = yg
        # speed estimate
        self.Q = Q
        self.Qd = Qd
        self.PinocchioUpdate(BaseID, ContactFootID, dt)
        self.yv = self.GetYV_v1()
    
        
          
        # state variables derivatives update
        self.x1_hat_dot = - self.S(yg) @ self.x1_hat - self.g0 * ya + alpha1*( self.yv - self.x1_hat)
        
        self.x2_prime_dot = -self.S(yg) @ self.x2_prime - alpha2/self.g0 * ( self.yv - self.x1_hat)
        
        self.x2_hat_dot = -self.S(yg - gamma*self.S(self.x2_hat) @ self.x2_prime) @ self.x2_hat
        
        
        # state variable integration
        self.x1_hat += dt * self.x1_hat_dot
        self.x2_hat += dt * self.x2_hat_dot
        self.x2_prime += dt * self.x2_prime_dot
        
        
        # error update
        self.ErrorUpdate(alpha1, alpha2, gamma, dt)
        
        # logging
        self.UpdateLogs()
        
        # return estimated data
        return self.x1_hat, self.x2_hat
    
    














































