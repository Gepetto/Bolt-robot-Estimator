import numpy as np
from pykalman import KalmanFilter



class KalmanFilter():
    def __init__(self,
                parameters=None,
                name="[Kalman]",
                talkative=False,
                logger=None,
                n=None,
                m=None) -> None:
        self.name=name
        if n is not None:
            self.n = n
        else:
            self.n = 3 + 3 + 6   # base position, base velocity, feet position
        if m is not None:
            self.m = m
        else:
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
        self.A, self.B, self.H, self.Q, self.R, self.initial_value = self.parameters
        self.X = self.initial_value
        self.U = 0.#???
        kf = KalmanFilter(transition_matrices=self.A,
                        observation_matrices=self.H,
                        command_matrices = self.B,
                        initial_state_mean=self.initial_value,
                        observation_covariance=self.Q,
                        transition_covariance=self.R)
        if logger is not None : self.logger = logger
        if self.Talkative : logger.LogTheLog("Filter " + self.name + " initialized with parameters " + str(self.parameters))

    def DefaultParametersInitializer(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # initializes filter parameters to default parameters
        A = np.eye(self.n)
        A[0:3, 3:6] = self.TimeStep * np.eye(3)

        control_matrix = np.zeros((self.n, 3))
        control_matrix[:, :3] += 0.5*self.TimeStep**2*np.eye(3)
        control_matrix[:, 3:6] += self.TimeStep*np.eye(3)

        H = np.zeros((self.m, self.n))

        Q = np.zeros((self.n, self.n))
        R = np.zeros((self.m, self.m))
        initial_value = np.zeros()
        return A, B, H, Q, R, initial_value

    def FilterAttitude(self, RobotState) -> np.ndarray:
        # Runs the Kalman filter
        self.filtered_state_mean, self.filtered_state_covariance = kf.filter(RobotState)
        return self.filtered_state_mean








class KalmanFilterReimplemented(KalmanFilter):
    def __init__(self,
                parameters=None,
                name="[Kalman, Reimplemented:Unreliable]",
                talkative=False):
        super.__init__(parameters, name, talkative)
    
    def Initialize(self) -> None :
        return None

    def Predict(self, U) -> tuple[np.ndarray, np.ndarray]:
        self.U = U
        self.X = self.A @ self.X + self.B @ self.U
        self.P = self.A @ (self.P @ self.A.T)  + self. Q
        return self.X, self.P

    def Correct(self, Z):
        # Z measurement
        self.K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        self.X = self.X + self.K @ (Z - self.H @ self.X)
        self.P = self.P - self.K @ self.H @ self.P
        pass

    def Update(self):
        # Y measurement
        # mean of Y prediction (?)
        IM = self.H @ self.X
        # Covariance of Y (?)
        IS = self.R + self.H @ (self.P @ np.transpose(self.H))
        # Kalman gain
        self.K = P @ (np.transpose(self.H) @ np.linalg.inv(IS))
        self.X = self.X + self.K @ (self.Y-IM)
        self.P = self.P - self.K @ (IS @ np.transpose(self.K))
        LH = 0. # TO UPDATE





