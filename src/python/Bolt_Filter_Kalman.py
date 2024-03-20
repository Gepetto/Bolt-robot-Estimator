import numpy as numpy
from pykalman import KalmanFilter

class KalmanFilter():
    def __init__(self,
                parameters=None,
                name="[Kalman]",
                talkative=False) -> None:
        self.name=name
        if parameters==None:
            parameters = self.DefaultParametersInitializer()
        self.parameters = parameters
        self.Talkative = talkative
        # matrixes and covariances
        self.TransitionMatrix, self.ObservationMatrix, self.TransitionCovariance, self.ObservationCovariance, self.InitialValue = self.parameters

        kf = KalmanFilter(transition_matrices=self.TransitionMatrix,
                        observation_matrices=self.ObservationMatrix,
                        initial_state_mean=self.InitialValue,
                        observation_covariance=self.ObservationCovariance,
                        transition_covariance=self.TransitionCovariance)
        if self.Talkative : print("  -> Filter " + self.name + " initialized with parameters " + str(self.parameters))

    def DefaultParametersInitializer(self) -> tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        # initializes filter parameters to default parameters
        TransitionMatrix = np.array([[]])
        ObservationMatrix = np.array([[]])
        TransitionCovariance = np.array([[]])
        ObservationCovariance = np.array([[]])
        InitialValue = np.zeros()

        return TransitionMatrix, ObservationMatrix, TransitionCovariance, ObservationCovariance, InitialValue

    def FilterAttitude(self, Rot) -> np.ndarray:
        # Runs the Kalman filter
        self.FilteredStateMean, self.FilteredStateCovariance = kf.filter(Rot)
        return self.FilteredStateMean


    
