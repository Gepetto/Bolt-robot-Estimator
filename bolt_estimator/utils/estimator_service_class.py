
import numpy as np


"""
        Initialize estimator class.
        Args :  device              (object)        the odri interface from which Estimator will get sensor's data
                model_path           (string)        if none, will use example-robot-data
                urdf_path            (string)
                talkative           (boolean)       if True, Estimator will log errors and key information and print them
                logger              (object)        the logger object to store the logs in
                attitude_filter_type  (string)        "complementary"
                parameters_af        (list of float) parameters of the attitude filter. If complementary, list of one float.
                speed_filter_type     (string)        "complementary"
                parameters_sf        (list of float) parameters of the attitude filter. If complementary, list of one float.
                parameters_pf        (list of float) parameters of the height of base filter. If complementary, list of one float.
                parameters_ti        (list of float) parameters of the tilt estimator, list of three float (alpha1, alpha2, gamma)
                t0pos_drift_comp      (float)         At time PDC, estimator will start compensating position drift using footstep integration 
                time_step            (float)         dt
                iter_number          (int)           the estimated number of times Estimator will run. Logs will only include the n=iter_number first data 
                estimator_logging    (boolean)       whether estimator should store data in log matrixes
                contact_logging      (boolean)       whether contact estimator should store data in log matrixes
                tilt_logging         (boolean)       whether tilt estimator should store data in log matrixes
        """

class EstimatorServiceClass():
    def __init__(self):
        self.model_path = ""
        self.urdf_path = ""
        self.talkative = True


        self.attitude_filter_type = "complementary"
        self.parameters_af = [2]
        self.speed_filter_type  = "complementary"
        self.parameters_sf = [1.1]
        self.parameters_pf = [0.15]
        self.parameters_ti = [10, 60, 2]

        self.t0pos_drift_comp = 2.5

        self.time_step = 0.001
        self.iter_number = 1000

        self.estimator_logging = True
        self.contact_logging = True
        self.tilt_logging = True

        return None


    def set(self,
                model_path           : str = "",
                urdf_path            : str = "",
                talkative           : bool = True,
                attitude_filter_type  : str = "complementary",
                parameters_af        : list = [2],
                speed_filter_type     : str = "complementary",
                parameters_sf        : list = [1.1],
                parameters_pf        : list = [0.15],
                parameters_ti        : list = [10, 60, 2],
                t0pos_drift_comp      : float = 2.5,
                time_step            : float = 0.01,
                iter_number          : int = 1000,
                estimator_logging    : bool = True,
                contact_logging      : bool = True,
                tilt_logging         : bool = True,
                ) -> None:
        pass
        