import numpy as np

class ComplementaryFilter():
    def __init__(self,
                parameters=[0.001, 2],
                name="[Complementary]",
                ndim=1,
                talkative=False,
                logger=None,
                MemorySize=100,
                OffsetGain=0.005) -> None:
        self.name=name
        self.parameters = parameters
        self.Talkative = talkative
        self.ndim = ndim
        # number of filter run
        self.k = 0
        # sampling interval, cut-off frequency
        self.T, self.a = self.parameters[0], self.parameters[1]
        self.b = self.a / (self.T + self.a)
        # initializing current output
        self.Estimate = np.zeros(self.ndim)
        # averaging system for offset correction
        self.MemorySize = MemorySize
        self.ErrorHistory = np.zeros((self.MemorySize, self.ndim))
        self.Offset = np.zeros(self.ndim)
        self.OffsetGain=OffsetGain
        # logs
        if logger is not None : 
            self.logger = logger
            self.Talkative = False
        if self.Talkative : self.logger.LogTheLog("Filter '" + self.name + "' initialized with parameters " + str(self.parameters))




    # standard filter
    def RunFilter(self, x, xdot) -> np.ndarray:
        # complementary filter x and its temporal derivative xdot. Updates previous estimates and returns current estimate.
        # check data
        if self.ndim==1 :
            if  not isinstance(x, float) or not isinstance(xdot, float):
                self.logger.LogTheLog(f"giving unadapted argument to filter '{self.name}' of dim. 1 : expected float, got {type(x)}" , style="warn")
        elif (not isinstance(x, np.ndarray) or not isinstance(xdot, np.ndarray)):
            self.logger.LogTheLog(f"giving unadapted argument to filter '{self.name}' : expected np.array" , style="warn")
            return None
        elif (x.shape != (self.ndim, ) or xdot.shape != (self.ndim, ) ):  
            self.logger.LogTheLog(f"giving unadapted argument to filter '{self.name}' : expected dim. ({self.ndim},), got {x.shape} and {xdot.shape}" , style="warn")           
            return None
        if self.k==0:
            # filter runs for the first time
            self.Estimate = x
        self.Estimate = self.b*self.Estimate + self.T*self.b*xdot + (1-self.b)*x
        self.k += 1
        # if (self.Estimate.shape != (self.ndim, ) ):  
        #     self.logger.LogTheLog(f"anormal output shape {self.Estimate.shape}" , style="warn")           
        return self.Estimate
    
    
    def RunFilterQuaternion(self, q, w) -> np.ndarray:
        # complementary filter for q [scalar-last format] and angular speed w
        # check data
        if (not isinstance(q, np.ndarray)) or (not isinstance(w, np.ndarray)) or (q.shape!=(4,) or w.shape!=(3,) ):  
            if self.Talkative : self.logger.LogTheLog(f"giving unadapted argument to filter '{self.name}' : expected np.array of dim 4 and 3" , style="warn")           
        if self.k==0:
            # filter runs for the first time
            self.Estimate = q
        # compute quaternion derivative
        wx, wy, wz = w
        qdot = 0.5 * np.array([wx*q[3] + wz*q[1] - wy*q[2],
                               wy*q[3] - wz*q[0] + wx*q[2],
                               wz*q[3] + wy*q[0] - wx*q[1],
                              -wx*q[0] - wy*q[1] - wz*q[2]])
        # update estimate
        self.Estimate = self.b*self.Estimate + self.T*self.b*qdot + (1-self.b)*q
        self.Estimate = self.Estimate / np.linalg.norm(self.Estimate)
        self.k += 1
        #if self.k < 5 : self.logger.LogTheLog("Running Filter " + self.name + " (on run " + str(self.k) + " out of 4 prints)")
        return self.Estimate

    
    # standard filter with non-idiotic offset compensation
    def RunFilterOffset(self, x, xdot) -> np.ndarray:
        # better averaging technique
        # complementary filter x and its temporal derivative xdot. Updates previous estimates and returns current estimate.
        # check data
        if (not isinstance(x, np.ndarray) or not isinstance(xdot, np.ndarray)):
            self.logger.LogTheLog(f"giving unadapted argument to filter '{self.name}' : expected np.array" , style="warn")
            return None
        elif (x.shape != (self.ndim, ) or xdot.shape != (self.ndim, ) ):  
            self.logger.LogTheLog(f"giving unadapted argument to filter '{self.name}' : expected dim. ({self.ndim},), got {x.shape}" , style="warn")           
            return None
        if self.k==0:
            # filter runs for the first time
            self.Estimate = x
        self.Estimate = self.b*self.Estimate + self.T*self.b*xdot + (1-self.b)*x
        # prepare offset correction
        self.ErrorHistory[self.k%self.MemorySize, :] = x-self.Estimate
        self.Offset = np.true_divide(self.ErrorHistory.sum(axis=0), np.count_nonzero(self.ErrorHistory, axis=0)) 
        # offset correction
        self.Estimate += self.Offset * self.OffsetGain
        self.k += 1
        #if self.k < 5 : self.logger.LogTheLog("Running Filter " + self.name + " (on run " + str(self.k) + " out of 4 prints)")
        return self.Estimate

    # standard filter with offset commpensation and adaptative gain for quicker convergence
    def RunFilterOffset3(self, x, xdot) -> np.ndarray:
        # complementary filter x and its temporal derivative xdot. Updates previous estimates and returns current estimate.
        # check data
        if (not isinstance(x, np.ndarray) or not isinstance(xdot, np.ndarray)):
            self.logger.LogTheLog(f"giving unadapted argument to filter '{self.name}' : expected np.array" , style="warn")
            return None
        elif (x.shape != (self.ndim, ) or xdot.shape != (self.ndim, ) ):  
            self.logger.LogTheLog(f"giving unadapted argument to filter '{self.name}' : expected dim. ({self.ndim},), got {x.shape}" , style="warn")           
            return None
        if self.k==0:
            # filter runs for the first time
            self.Estimate = x
        self.Estimate = self.b*self.Estimate + self.T*self.b*xdot + (1-self.b)*x
        # prepare offset correction
        self.ErrorHistory[self.k%self.MemorySize, :] = x-self.Estimate
        
        Converger = (self.MemorySize/4 - self.k) * self.OffsetGain * 0.17
        if Converger < self.OffsetGain:
            Converger = self.OffsetGain
        self.Offset = np.true_divide(self.ErrorHistory.sum(axis=0), np.count_nonzero(self.ErrorHistory, axis=0)) * Converger
        # offset correction
        self.Estimate += self.Offset
        self.k += 1
        #if self.k < 5 : self.logger.LogTheLog("Running Filter " + self.name + " (on run " + str(self.k) + " out of 4 prints)")
        return self.Estimate




    
