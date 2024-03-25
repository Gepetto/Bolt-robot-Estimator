import numpy as np

class ComplementaryFilter():
    def __init__(self,
                parameters=(0.001, 2),
                name="[Complementary]",
                ndim=3,
                talkative=False,
                logger=None) -> None:
        self.name=name
        self.parameters = parameters
        self.Talkative = talkative
        # sampling interval, cut-off frequency
        self.T, self.a = self.parameters
        self.b = self.a / (self.T + self.a)
        self.PreviousOutput = np.zeros(ndim)
        if logger is not None : self.logger = logger
        if self.Talkative : logger.LogTheLog("Filter " + self.name + " initialized with parameters " + str(self.parameters))


    def FilterAttitude(self, xdot, x) -> np.ndarray: 
        # complementary filter x and its temporal derivative xdot. Updates previous estimates and returns current estimate.
        self.PreviousOutput = self.b*self.PreviousOutput + self.T*self.b*xdot + (1-self.b)*x
        return self.PreviousOutput


    
