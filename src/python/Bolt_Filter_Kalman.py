import numpy as np
from pykalman import KalmanFilter



class KalmanFilter():
    def __init__(self,
                parameters=None,
                name="[Kalman]",
                talkative=False) -> None:
        self.name=name
        self.n = 3 + 3 + 6   # base position, base velocity, feet position
        self.m = 4 + 12      # ??? IMU 
        if parameters==None:
            parameters = self.DefaultParametersInitializer()
        self.parameters = parameters
        self.Talkative = talkative
        # matrixes and covariances
        # A : TransitionMatrix
        # B : CommandMatrix
        # H : ObservationMatrix
        # Q : TransitionCovariance
        # R : ObservationCovariance
        # K : Kalman gain
        self.A, self.B, self.H, self.Q, self.R, self.InitialValue = self.parameters
        self.X = self.InitialValue
        kf = KalmanFilter(transition_matrices=self.A,
                        observation_matrices=self.H,
                        command_matrices = self.B,
                        initial_state_mean=self.InitialValue,
                        observation_covariance=self.Q,
                        transition_covariance=self.R)
        if self.Talkative : print("  -> Filter " + self.name + " initialized with parameters " + str(self.parameters))

    def DefaultParametersInitializer(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # initializes filter parameters to default parameters
        A = np.eye(self.n)
        A[0:3, 3:6] = self.TimeStep * np.eye(3)

        ControlMatrix = np.zeros((self.n, 3))
        ControlMatrix[:, :3] += 0.5*self.TimeStep**2*np.eye(3)
        ControlMatrix[:, 3:6] += self.TimeStep*np.eye(3)

        H = np.zeros((self.m, self.n))


        Q = np.zeros((self.n, self.n))
        R = np.zeros((self.m, self.m))
        InitialValue = np.zeros()
        return A, B, H, Q, R, InitialValue

    def FilterAttitude(self, Rot) -> np.ndarray:
        # Runs the Kalman filter
        self.FilteredStateMean, self.FilteredStateCovariance = kf.filter(Rot)
        return self.FilteredStateMean








class KalmanFilterReimplemented(KalmanFilter):
    def __init__(self,
                parameters=None,
                name="[Kalman]",
                talkative=False):
        super.__init__(parameters, name, talkative)
    
    def Initialize(self):
        pass
    def Predict(self):
        pass
    def Correct(self):
        pass
    def Update(self):
        pass