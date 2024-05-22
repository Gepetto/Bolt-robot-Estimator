import numpy as np


"""
Created on Wed May 22 13:41:35 2024

@author: nalbrecht
"""


class Bolt_TiltEstimator():
    
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
        pin.forwardKinematics(self.bolt.model, self.data, self.q)
        pin.updateFramePlacements(self.bolt.model, self.data)
        
        # state variables
        self.x1_hat = np.zeros(3)
        self.x1_hat_dot = np.zeros(3)
        self.x2_hat = np.zeros((3,3))
        self.x2_hat_dot = np.zeros((3,3))
        self.x2_prime = np.zeros((3,3))
        self.x2_prime_dot = np.zeros((3,3))
        
        # errors
        self.x1_tilde = 0
        self.x2_tilde = 0
        self.x2_tildeprime = 0
        
        
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
        return x
    
    
    def GetYV_v1(self) -> np.ndarray:
        """ Estimate Yv and return it """
        yv = self.c_R_l.T @ self.c_Pdot_l + self.S(self.yg - self.c_Omega_l) @ self.c_R_l.T @ self.c_P_l
        return yv
    
    def GetYV_v2(self, LFootID, RFootID, BaseID, eta) -> np.ndarray:
        """ Estimate Yv and return it """
        c_P_anchor = eta *  self.data.oMf[LFootID].translation + (1-eta) * self.data.oMf[RFootID].translation
        v1 = pin.getFrameVelocity(self.bolt.model, self.data, LFootID, BaseID)
        v2 = pin.getFrameVelocity(self.bolt.model, self.data, RFootID, BaseID)
        C_Pdot_anchor = eta *  v1.translation + (1-eta) * v2.translation
        yv = -S(self.yg) @ c_P_anchor - c_Pdot_anchor
        return yv
    
    def PinocchioUpdate(self, BaseID, ContactFootID, dt) -> None:
        """ Update pinocchio data with forward kinematics and update FK variables"""
        
        # update pinocchio data
        pin.forwardKinematics(self.bolt.model, self.data, self.Q)
        pin.updateFramePlacements(self.bolt.model, self.data)
        pin.computeAllTerms(self.bolt.model, self.data3D, self.Q, self.Qd)
        
        # update relevant data
        # rotation matrix of base frame in contact foot frame
        self.prev_c_R_l = self.c_R_l.copy()
        self.c_R_l = self.data.oMf[BaseID] @ self.data.oMf[ContactFootID].T
        self.c_Rdot_l =  (self.c_R_l -self.prev_c_R_l) / dt # TODO : chk
        # position of base frame in contact foot frame
        self.c_P_l = 
        self.c_Pdot_l =
        # rotation speed of base frame in contact foot frame
        self.c_Omega_l = pin.getFrameVelocity(self.bolt.model, self.data, BaseID, ContactFootID).rotation
        
        
        return None
    
    
    def ErrorUpdate(self, alpha1:float, alpha2:float, gamma:float, dt:float) -> None:
        """ Compute error in estimation and update error variables"""
        S2 = 0 # TODO : upd
        
        x1_tilde_dot      = -S(self.yg) @ self.x1_tilde - alpha1 * self.x1_tilde - g0*self.x2_tildeprime
        x2_tildeprime_dot = -S(self.yg) @ self.x2_tilde - alpha2/self.g0 * self.x1_tilde
        x2_tilde_dot      = -S(self.yg) @ self.x2_tilde - gamma * S2 @ (self.x2_tilde - self.x2_tildeprime)
        
        self.x1_tilde += x1_tilde_dot * dt
        self.x2_tilde += x2_tilde_dot * dt
        self.x2_tildeprime += x2_tildeprime_dot * dt
        
        return None

    
    
    def Estimator(self, 
                  Q:np.ndarray,
                  Qd:np.ndarray,
                  BaseID:int,
                  ContactFootID:int,
                  ya:np.ndarray, 
                  yg:np.ndarray, 
                  dt:float, 
                  alpha1:float, alpha2:float, gamma:float) -> tuple[np.ndarray, np.ndarray] :
        """ update state variables and return current tilt estimate """
        
        # speed estimate
        self.Q = Q
        self.Qd = Qd
        self.PinocchioUpdate(BaseID, ContactFootID, dt)
        yv = self.GetYV()
        
        # store measurement
        self.ya = ya
        self.yg = yg
        self.yv = yv
        
          
        # state variables derivatives update
        self.x1_hat_dot = - self.S(yg) @ self.x1_hat - self.g0 * ya + alpha1*( yv - self.x1_hat)
        
        self.x2_prime_dot = -self.S(yg) @ self.x2_prime - alpha2/self.g0 * (yv - x1_hat)
        
        self.x2_hat_dot = -S(yg - gamma*S(self.x2_hat) @ self.x2_prime) @ self.x2_hat
        
        
        # state variable integration
        self.x1_hat += dt * self.x1_hat_dot
        self.x2_hat += dt * self.x2_hat_dot
        self.x2_prime += dt * self.x2_prime_dot
        
        
        # error update
        self.ErrorUpdate(alpha1, alpha2, gamma, dt)
        
        # logging
        self.UpdateLogs()
        
        # measurement
        
        
        
        return self.x1_hat, self.x2_hat
    
    














































