#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 12:16:08 2024

@author: niels
"""

from application.Bolt_Filter_Complementary import ComplementaryFilter
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
        self.iter = 0
        self.Logging = Logging
        self.FS = FilterSpeed
        self.FA = FilterAttitude
        self.FP = FilterPosition

        self.LearningIter = 0
        self.omega_bias = np.zeros((3,))
        self.a_bias = np.zeros((3,))

        
        # filters params
        parametersSF = [self.TimeStep] + parametersSF
        parametersAF = [self.TimeStep] + parametersAF
        parametersPF = [self.TimeStep] + parametersPF

        self.SpeedFilter = ComplementaryFilter(parameters=parametersSF, 
                                                name="speed complementary filter", 
                                                talkative=Talkative, 
                                                logger=None, 
                                                ndim=3,
                                                MemorySize=100,
                                                OffsetGain=0.005)
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

        self.p_logs_mocap = np.zeros((self.Logging, 3))
        self.v_logs_mocap = np.zeros((self.Logging, 3))
        self.q_logs_mocap = np.zeros((self.Logging, 4))

        self.a_logs_imu = np.zeros((self.Logging, 3))

        
    def UpdateLogs(self, p, v, q, w, p_mocap, v_mocap, q_mocap, a_imu):
        if self.iter>=self.Logging :
            return None
        self.p_logs[self.iter, :] = p[:]
        self.v_logs[self.iter, :] = v[:]
        self.q_logs[self.iter, :] = q[:]
        self.w_logs[self.iter, :] = w[:]

        self.p_logs_mocap[self.iter, :] = p_mocap[:]
        self.v_logs_mocap[self.iter, :] = v_mocap[:]
        self.q_logs_mocap[self.iter, :] = q_mocap[:]

        self.a_logs_imu[self.iter, :] = a_imu[:]
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

    def LearnIMUBias(self, a_imu, omega_imu):
        self.a_bias = (self.LearningIter * self.a_bias + a_imu)/(self.LearningIter + 1)
        self.omega_bias = (self.LearningIter * self.omega_bias + omega_imu)/(self.LearningIter + 1)
    
    def PlotLogs(self):
        if self.Logging==0:
            print("no logs stored")
            return None
        plt.clf()
        
        plt.figure()
        plt.grid()
        plt.title("position out")
        plt.plot(self.p_logs[:, 0], label="position X out")
        plt.plot(self.p_logs[:, 1], label="position Y out")
        plt.plot(self.p_logs[:, 2], label="position Z out")
        plt.legend()
        
        plt.figure()
        plt.grid()
        plt.title("speed out")
        plt.plot(self.v_logs[:, 0], label="speed X out")
        plt.plot(self.v_logs[:, 1], label="speed Y out")
        plt.plot(self.v_logs[:, 2], label="speed Z out")
        plt.legend()

        # plt.figure()
        # plt.grid()
        # plt.title("speed derived from position out")
        # plt.plot(self.p_logs[1:, 0] - self.p_logs[:-1, 0]*1.6, label="computed speed X for 1600Hz") # HARD CODED
        # plt.legend()


        plt.figure()
        plt.grid()
        plt.title("speed X comparison")
        plt.plot(self.v_logs_mocap[:, 0], label="speed X mocap")
        plt.plot(self.v_logs[:, 0], label="speed X out")
        plt.legend()
        
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
        plt.legend()

        plt.figure()
        plt.grid()
        plt.title("acceleration imu")
        plt.plot(self.a_logs_imu[:, 0], label="acc X imu")
        plt.plot(self.a_logs_imu[:, 1], label="acc Y imu")
        plt.plot(self.a_logs_imu[:, 2], label="acc Z imu")
        plt.legend()
        
        plt.show()

    
    
    def Run(self, p_mocap, v_mocap, quat_mocap, omega_imu, a_imu, dt=None):
        # correct learned bias
        a_imu_corr = a_imu - self.a_bias
        omega_imu_corr = omega_imu - self.omega_bias
        # initialize data
        p_out = np.copy(p_mocap)
        v_out = np.copy(v_mocap)
        quat_out = np.copy(quat_mocap)
        omega_out = np.copy(omega_imu_corr)
        
        
        
        # filter speed
        v_filter = self.SpeedFilter.RunFilter(v_mocap, a_imu_corr, dt)
        # filter attitude
        quat_filter = self.AttitudeFilter.RunFilterQuaternion(quat_mocap, omega_imu_corr, dt)
        # filter position
        p_filter = self.PositionFilter.RunFilter(p_mocap, v_filter, dt)
        
        if self.FP:
            p_out = p_filter
        if self.FS:
            v_out = v_filter
        if self.FA :
            quat_out = quat_filter
        
        if self.Logging != 0 :
            self.UpdateLogs(p_out, v_out, quat_out, omega_out, p_mocap, v_mocap, quat_mocap, a_imu)
        
        self.iter += 1
        
        return p_out, v_out, quat_out, omega_out
    
    
    
    
    