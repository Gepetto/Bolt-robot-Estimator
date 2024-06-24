#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 12:16:08 2024

@author: niels
"""

from Bolt_Filter_Complementary import ComplementaryFilter
import numpy as np
import matplotlib.pyplot as plt

class MocapIMUFilter():
    def __init__(self,
                 parametersSF = [2],
                 parametersAF = [1.1],
                 parametersPF = [2],
                 FilterSpeed=True,
                 FilterAttitude=False,
                 FilterPosition=False,
                 dt = 0.001,
                 Logging=0,
                 Talkative=False
                 ):
        # params
        self.TimeStep = dt
        self.iter=0
        self.Logging = Logging
        self.FS = FilterSpeed
        self.FA = FilterAttitude
        self.FP = FilterPosition
        
        # filters params
        parametersSF = [self.TimeStep] + parametersSF
        parametersAF = [self.TimeStep] + parametersAF
        parametersPF = [self.TimeStep] + parametersPF

        self.SpeedFilter = ComplementaryFilter(parameters=parametersSF, 
                                                name="speed complementary filter", 
                                                talkative=Talkative, 
                                                logger=None, 
                                                ndim=3)
        self.PositionFilter = ComplementaryFilter(parameters=parametersPF, 
                                                name="speed complementary filter", 
                                                talkative=Talkative, 
                                                logger=None, 
                                                ndim=3)
        self.AttitudeFilter = ComplementaryFilter(parameters=parametersAF, 
                                                name="attitude complementary filter", 
                                                talkative=Talkative, 
                                                logger=None, 
                                                ndim=3)
        if Talkative: print("Mocap IMU filter initialized")
        
        if self.Logging != 0 :
            self.InitLogs()
        
        return None
    
    def InitLogs(self):
        self.p_logs = np.zeros((self.Logging, 3))
        self.v_logs = np.zeros((self.Logging, 3))
        self.q_logs = np.zeros((self.Logging, 4))
        self.w_logs = np.zeros((self.Logging, 3))
        
    def UpdateLogs(self, p, v, q, w):
        if self.iter>=self.Logging :
            return None
        self.p_logs[self.iter, :] = p[:]
        self.v_logs[self.iter, :] = v[:]
        self.q_logs[self.iter, :] = q[:]
        self.w_logs[self.iter, :] = w[:]
        return None
    
    def GetLogs(self, data="position"):
        if self.Logging==0:
            print("no logs stored")
            return None
        if data=="position":
            return self.p_logs
        elif data=="speed":
            return self.v_logs
        elif data=="theta" or data=="quat":
            return self.q_logs
        elif data=="omega":
            return self.w_logs
        else :
            print("wrong data getter")
    
    def PlotLogs(self):
        if self.Logging==0:
            print("no logs stored")
            return None
        plt.clf()
        
        plt.figure()
        plt.grid()
        plt.title("position")
        plt.plot(self.p_logs[:, 0], label="position X out")
        plt.plot(self.p_logs[:, 1], label="position Y out")
        plt.plot(self.p_logs[:, 2], label="position Z out")
        plt.legend()
        
        plt.figure()
        plt.grid()
        plt.title("speed")
        plt.plot(self.v_logs[:, 0], label="speed X out")
        plt.plot(self.v_logs[:, 1], label="speed Y out")
        plt.plot(self.v_logs[:, 2], label="speed Z out")
        plt.legend
        
        plt.figure()
        plt.grid()
        plt.title("attitude")
        plt.plot(self.q_logs, label="quaternion attitude")
        plt.legend()
        
        plt.figure()
        plt.grid()
        plt.title("angular speed")
        plt.plot(self.w_logs[:, 0], label="angular speed X out")
        plt.plot(self.w_logs[:, 1], label="angular speed Y out")
        plt.plot(self.w_logs[:, 2], label="angular speed Z out")
        plt.legend
        
        plt.show()

    
    
    def Run(self, p_mocap, v_mocap, quat_mocap, omega_imu, a_imu):
        # initialize data
        p_out = np.copy(p_mocap)
        v_out = np.copy(v_mocap)
        quat_out = np.copy(quat_mocap)
        omega_out = np.copy(omega_imu)
        
        
        
        # filter speed
        v_filter = self.SpeedFilter.RunFilter(v_mocap, a_imu)
        # filter attitude
        quat_filter = self.AttitudeFilter.RunFilterQuaternion(quat_mocap, omega_imu)
        # filter position
        p_filter = self.PositionFilter.RunFilter(p_mocap, v_filter)
        
        if self.FP:
            p_out = p_filter
        if self.FS:
            v_out = v_filter
        if self.FA :
            quat_out = quat_filter
        
        if self.Logging != 0 :
            self.UpdateLogs(p_out, v_out, quat_out, omega_out)
        
        self.iter += 1
        
        return p_out, v_out, quat_out, omega_out
    
    
    
    
    