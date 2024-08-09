#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:02:02 2024

@author: niels
"""

from bolt_estimator.utils.MocapIMUFilter import MocapIMUFilter
import numpy as np

Filter = MocapIMUFilter(parameters_sf = [2],
                        parameters_af = [1.1],
                        parameters_pf = [2],
                        filter_speed=True,
                        filter_attitude=True,
                        filter_position=True,
                        dt = 0.001,
                        logging=5,
                        talkative=False)

N=100


for j in range(N):
    p = np.random.rand(3)
    v = np.random.rand(3)
    q = np.random.rand(4)
    w = np.random.rand(3)
    a = np.random.rand(3)
    Filter.Run(p, v, q, w, a)

x = Filter.GetLogs("theta")
Filter.PlotLogs()
