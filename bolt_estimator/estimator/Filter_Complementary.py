import numpy as np

class ComplementaryFilter():
    def __init__(self,
                parameters=[0.001, 2],
                name="[Complementary]",
                ndim=1,
                talkative=False,
                logger=None,
                memory_size=100,
                offset_gain=0.005) -> None:
        self.name=name
        self.parameters = parameters
        self.talkative = talkative
        self.ndim = ndim
        # number of filter run
        self.k = 0
        # sampling interval, cut-off frequency
        self.T, self.a = self.parameters[0], self.parameters[1]
        self.b = self.a / (self.T + self.a)
        # initializing current output
        self.estimate = np.zeros(self.ndim)
        # averaging system for offset correction
        self.memory_size = memory_size
        self.error_history = np.zeros((self.memory_size, self.ndim))
        self.offset = np.zeros(self.ndim)
        self.offset_gain = offset_gain
        # logs
        if logger is not None : 
            self.logger = logger
            self.talkative = False
        if self.talkative : self.logger.LogTheLog("Filter '" + self.name + "' initialized with parameters " + str(self.parameters))


    def Setdt(self, dt) -> None:
        if dt is not None :
            self.T = dt
            self.b = self.a / (self.T + self.a)
        return None

    # standard filter
    def RunFilter(self, x, xdot, dt=None) -> np.ndarray:
        # complementary filter x and its temporal derivative xdot. Updates previous estimates and returns current estimate.
        # check data
        self.Setdt(dt)
        if self.ndim==1 :
            if  not isinstance(x, float) or not isinstance(xdot, float):
                self.logger.LogTheLog(f"giving unadapted argument to filter '{self.name}' of dim. 1 : expected float, got {type(x)}" , style="danger")
        elif (not isinstance(x, np.ndarray) or not isinstance(xdot, np.ndarray)):
            self.logger.LogTheLog(f"giving unadapted argument to filter '{self.name}' : expected np.array" , style="warn")
            return None
        elif (x.shape != (self.ndim, ) or xdot.shape != (self.ndim, ) ):  
            self.logger.LogTheLog(f"giving unadapted argument to filter '{self.name}' : expected dim. ({self.ndim},), got {x.shape} and {xdot.shape}" , style="danger")           
            return None
        if self.k==0:
            # filter runs for the first time
            self.estimate = x
        self.estimate = self.b*self.estimate + self.T*self.b*xdot + (1-self.b)*x
        self.k += 1
        # if (self.estimate.shape != (self.ndim, ) ):  
        #     self.logger.LogTheLog(f"anormal output shape {self.estimate.shape}" , style="warn")           
        return self.estimate
    
    
    def RunFilterQuaternion(self, q, w, dt=None) -> np.ndarray:
        # complementary filter for q [scalar-last format] and angular speed w
        # check data
        self.Setdt(dt)
        if (not isinstance(q, np.ndarray)) or (not isinstance(w, np.ndarray)) or (q.shape!=(4,) or w.shape!=(3,) ):  
            if self.talkative : self.logger.LogTheLog(f"giving unadapted argument to filter '{self.name}' : expected np.array of dim 4 and 3" , style="danger")           
        if self.k==0:
            # filter runs for the first time
            self.estimate = q
        # compute quaternion derivative
        wx, wy, wz = w
        qdot = 0.5 * np.array([wx*q[3] + wz*q[1] - wy*q[2],
                               wy*q[3] - wz*q[0] + wx*q[2],
                               wz*q[3] + wy*q[0] - wx*q[1],
                              -wx*q[0] - wy*q[1] - wz*q[2]])
        # update estimate
        self.estimate = self.b*self.estimate + self.T*self.b*qdot + (1-self.b)*q
        self.estimate = self.estimate / np.linalg.norm(self.estimate)
        self.k += 1
        #if self.k < 5 : self.logger.LogTheLog("Running Filter " + self.name + " (on run " + str(self.k) + " out of 4 prints)")
        return self.estimate

    
    # standard filter with non-idiotic offset compensation
    def RunFilteroffset(self, x, xdot, dt=None) -> np.ndarray:
        # better averaging technique
        # complementary filter x and its temporal derivative xdot. Updates previous estimates and returns current estimate.
        # check data
        self.Setdt(dt)
        if (not isinstance(x, np.ndarray) or not isinstance(xdot, np.ndarray)):
            self.logger.LogTheLog(f"giving unadapted argument to filter '{self.name}' : expected np.array" , style="danger")
            return None
        elif (x.shape != (self.ndim, ) or xdot.shape != (self.ndim, ) ):  
            self.logger.LogTheLog(f"giving unadapted argument to filter '{self.name}' : expected dim. ({self.ndim},), got {x.shape}" , style="danger")           
            return None
        if self.k==0:
            # filter runs for the first time
            self.estimate = x
        self.estimate = self.b*self.estimate + self.T*self.b*xdot + (1-self.b)*x
        # prepare offset correction
        self.error_history[self.k%self.memory_size, :] = x-self.estimate
        self.offset = np.true_divide(self.error_history.sum(axis=0), np.count_nonzero(self.error_history, axis=0)) 
        # offset correction
        self.estimate += self.offset * self.offset_gain
        self.k += 1
        #if self.k < 5 : self.logger.LogTheLog("Running Filter " + self.name + " (on run " + str(self.k) + " out of 4 prints)")
        return self.estimate

    # standard filter with offset commpensation and adaptative gain for quicker convergence
    def RunFilteroffsetAdaptive(self, x, xdot, dt=None) -> np.ndarray:
        # complementary filter x and its temporal derivative xdot. Updates previous estimates and returns current estimate.
        # check data
        self.Setdt(dt)
        if (not isinstance(x, np.ndarray) or not isinstance(xdot, np.ndarray)):
            self.logger.LogTheLog(f"giving unadapted argument to filter '{self.name}' : expected np.array" , style="danger")
            return None
        elif (x.shape != (self.ndim, ) or xdot.shape != (self.ndim, ) ):  
            self.logger.LogTheLog(f"giving unadapted argument to filter '{self.name}' : expected dim. ({self.ndim},), got {x.shape}" , style="danger")           
            return None
        if self.k==0:
            # filter runs for the first time
            self.estimate = x
        self.estimate = self.b*self.estimate + self.T*self.b*xdot + (1-self.b)*x
        # prepare offset correction
        self.error_history[self.k%self.memory_size, :] = x-self.estimate
        
        converger = (self.memory_size/4 - self.k) * self.offset_gain * 0.17
        if converger < self.offset_gain:
            converger = self.offset_gain
        self.offset = np.true_divide(self.error_history.sum(axis=0), np.count_nonzero(self.error_history, axis=0)) * converger
        # offset correction
        self.estimate += self.offset
        self.k += 1
        #if self.k < 5 : self.logger.LogTheLog("Running Filter " + self.name + " (on run " + str(self.k) + " out of 4 prints)")
        return self.estimate




    
