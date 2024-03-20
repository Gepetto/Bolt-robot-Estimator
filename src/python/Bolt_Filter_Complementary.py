import numpy as numpy

class ComplementaryFilter():
    def __init__(self,
                parameters=(0.001, 2),
                name="[Complementary]",
                talkative=False) -> None:
        self.name=name
        self.parameters = parameters
        self.Talkative = talkative
        # sampling interval, cut-off frequency
        self.T, self.a = self.parameters
        self.b = self.a / (self.T + self.a)
        if self.Talkative : print("  -> Filter " + self.name + " initialized with parameters " + str(self.parameters))


    def FilterAttitude(self, RotAcc, RotGyro, PreviousRot) -> numpy.ndarray: 
        # complementary filter of rotation matrixes. Updates previous estimates and returns current estimate.
        self.PreviousOutput = self.b*self.PreviousOutput + self.T*self.b*RotGyro + (1-self.b)*RotAcc
        return self.PreviousOutput


    
