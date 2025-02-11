import numpy as np
import pinocchio as pin
import time as t
from scipy.spatial.transform import Rotation as R
import example_robot_data


from bolt_estimator.utils.Bolt_Utils import utils
from bolt_estimator.utils.Bolt_Utils import Log

from bolt_estimator.estimator.Bolt_ContactEstimator import ContactEstimator
from bolt_estimator.estimator.Bolt_TiltEstimator import TiltEstimator

from bolt_estimator.estimator.Bolt_Filter import Filter
from bolt_estimator.estimator.Bolt_Filter_Complementary import ComplementaryFilter
from bolt_estimator.estimator.Bolt_FootAttitudeEstimator import FootAttitudeEstimator


"""
An estimator for attitude, speed, and position using only an IMU

    This code uses Pinocchio, encoders and IMU data to provide an estimate of the robot
    base attitude, base speed and position, and center of mass' speed.

    The Estimator Class contains the different filters and sub-estimators 
    needed to merge the relevant data. Its main method is Estimator.Estimate() .

    The estimator called by Estimator is TiltEstimator.
    
    Documentation can be found at  $ $ $  G I T  $ $ $ 

License, authors, LAAS


NOTA BENE :
    * quaternions are in scalar-last format [xyz w]
    * logs and print cas be de-activated
    * complementary filters have 1/dt as first parameters automatically
    * g_*** are estimation of gravity in robot frame, normalized to 1 
    * euler angles are in [xyz] format # TODO : add as parameter
    

"""




class Estimator():
    def __init__(self,
                device,
                ModelPath           : str = "",
                UrdfPath            : str = "",
                Talkative           : bool = True,
                logger              : Log = None,
                AttitudeFilterType  : str = "complementary",
                parametersAF        : list = [2],
                SpeedFilterType     : str = "complementary",
                parametersSF        : list = [1.1],
                parametersPF        : list = [0.15],
                parametersTI        : list = [10, 60, 2],
                T0posDriftComp      : float = 2.5,
                TimeStep            : float = 0.01,
                IterNumber          : int = 1000,
                EstimatorLogging    : bool = True,
                ContactLogging      : bool = True,
                TiltLogging         : bool = True,
                ) -> None:
        """
        Initialize estimator class.
        Args :  device              (object)        the odri interface from which Estimator will get sensor's data
                ModelPath           (string)        if none, will use example-robot-data
                UrdfPath            (string)
                Talkative           (boolean)       if True, Estimator will log errors and key information and print them
                logger              (object)        the logger object to store the logs in
                AttitudeFilterType  (string)        "complementary"
                parametersAF        (list of float) parameters of the attitude filter. If complementary, list of one float.
                SpeedFilterType     (string)        "complementary"
                parametersSF        (list of float) parameters of the attitude filter. If complementary, list of one float.
                parametersPF        (list of float) parameters of the height of base filter. If complementary, list of one float.
                parametersTI        (list of float) parameters of the tilt estimator, list of three float (alpha1, alpha2, gamma)
                T0posDriftComp      (float)         At time PDC, estimator will start compensating position drift using footstep integration 
                TimeStep            (float)         dt
                IterNumber          (int)           the estimated number of times Estimator will run. Logs will only include the n=IterNumber first data 
                EstimatorLogging    (boolean)       whether estimator should store data in log matrixes
                ContactLogging      (boolean)       whether contact estimator should store data in log matrixes
                TiltLogging         (boolean)       whether tilt estimator should store data in log matrixes
        """
        
        self.MsgName = "Bolt Estimator v0.8"

        # logging options 
        self.Talkative=Talkative
        self.EstimatorLogging = EstimatorLogging
        self.ContactLogging = ContactLogging
        self.TiltLogging = TiltLogging
        # adding logger
        if logger is not None :
            self.logger = logger
        else:
            self.logger = Log("default " + self.MsgName+ " log")
        self.logger.LogTheLog(" Starting log of" + self.MsgName, ToPrint=False)
        if self.Talkative : self.logger.LogTheLog("Initializing " + self.MsgName + "...", style="title", ToPrint=Talkative)
        
        # iteration and timing
        self.IterNumber = IterNumber
        self.iter = 0
        self.TimeRunning = 0.
        # 1 kHz by default
        self.TimeStep = TimeStep
        
        # loading data from file
        if UrdfPath=="" or ModelPath=="":
            self.logger.LogTheLog("No URDF path or ModelPath addeds", style="warn", ToPrint=True)
            self.robot = example_robot_data.load("bolt")
        else :
            #model, collision_model, visual_model = pin.buildModelsFromUrdf(UrdfPath, ModelPath, pin.JointModelFreeFlyer())
            self.logger.LogTheLog("Bypassing URDF path or ModelPath", style="warn", ToPrint=True)
            self.robot = example_robot_data.load("bolt")

            self.robot = pin.RobotWrapper.BuildFromURDF(UrdfPath, ModelPath)
            self.logger.LogTheLog("URDF built", ToPrint=Talkative)
        # number of frames and movement
        self.nq = self.robot.model.nq
        self.nv = self.robot.model.nv
        # useful frame indexes
        self.FeetIndexes = [self.robot.model.getFrameId("FL_FOOT"), self.robot.model.getFrameId("FR_FOOT")] # Left, Right
        self.BaseID = 1

        # interfacing with masterboard (?)
        if device is not None :
            self.device = device
        else :
            self.logger.LogTheLog("No device added", style="warn", ToPrint=True)
            self.device=None
        
        # initializes data & logs with np.zeros arrays
        self.InitImuData()
        self.InitKinematicData()
        self.InitOutData()
        self.InitContactData()
        if self.EstimatorLogging : self.InitLogMatrixes()

        # check that sensors can be read
        if self.device is not None :
            self.ReadSensor()
            if self.Talkative : self.logger.LogTheLog("Sensors read, initial data acquired", ToPrint=Talkative)
        

        # update height of CoM value, assuming Bolt is vertical
        pin.forwardKinematics(self.robot.model, self.robot.data, self.q)
        pin.updateFramePlacements(self.robot.model, self.robot.data)
        c = np.array((self.robot.data.oMf[self.FeetIndexes[0]].inverse()*self.robot.data.oMf[self.BaseID]).translation).copy()
        self.p_out[2] = np.linalg.norm(c)
        self.p_out[2] += 0.02 # TODO : radius of bolt foot
        self.p_out[2] = 0.34 # TODO stop doing this
        
        if self.EstimatorLogging : self.UpdateLogMatrixes()
        self.iter += 1
        if self.Talkative : self.logger.LogTheLog("Initial data stored in logs", ToPrint=Talkative)
        
        # position drift compensation using switch and step measurement
        self.T0posDriftComp = T0posDriftComp
        self.PosDriftCompensation = 0
        self.StepStart = 0
        self.StepDuration = 0
        
        # filter parameters
        parametersAF = [self.TimeStep] + parametersAF
        parametersSF = [self.TimeStep] + parametersSF
        parametersPF = [self.TimeStep] + parametersPF
        # set desired filters types for attitude and speed
        # for the time being, complementary only
        if AttitudeFilterType=="complementary":
            self.AttitudeFilter = ComplementaryFilter(parameters=parametersAF, 
                                                    name="attitude complementary filter", 
                                                    talkative=Talkative, 
                                                    logger=self.logger, 
                                                    ndim=3)
        if self.Talkative : self.logger.LogTheLog("Attitude Filter of type '" + AttitudeFilterType + "' added.", ToPrint=Talkative)
        
        if SpeedFilterType=="complementary":
            self.SpeedFilter = ComplementaryFilter(parameters=parametersSF, 
                                                    name="speed complementary filter", 
                                                    talkative=Talkative, 
                                                    logger=self.logger, 
                                                    ndim=3,
                                                    MemorySize=80,
                                                    OffsetGain=0.02)
        if self.Talkative : self.logger.LogTheLog("Speed Filter of type '" + SpeedFilterType + "' added.", ToPrint=Talkative)
        
        self.HeightFilter = ComplementaryFilter(parameters=parametersPF, 
                                                name="base height complementary filter", 
                                                talkative=Talkative, 
                                                logger=self.logger, 
                                                ndim=1)
      
        # returns info on Slips, Contact Forces, Contact with the ground
        self.TiltandSpeedEstimator = TiltEstimator(robot=self.robot,
                                                   Q0=self.q,
                                                   Qd0=self.qdot,
                                                   Niter=self.IterNumber,
                                                   Logging=self.TiltLogging,
                                                   params=parametersTI)
        
        if self.Talkative : self.logger.LogTheLog("Tilt Estimator added with parameters " + str(parametersTI), ToPrint=Talkative)
        
        
        # returns info on foot attitude
        self.FootAttitudeEstimator = FootAttitudeEstimator(parameters=[self.TimeStep, 2],
                                                         dt=self.TimeStep,
                                                         name="Foot Attitude Estimator",
                                                         talkative=Talkative,
                                                         logger=self.logger)
        if self.Talkative : self.logger.LogTheLog("Foot Attitude Estimator added with parameters " + str(0), ToPrint=Talkative)
        

        self.logger.LogTheLog(self.MsgName +" initialized successfully.", ToPrint=Talkative)
        return None
    
    def SetInitValues(self, BaseSpeed, BaseAccG, UnitGravity, UnitGravityDerivative, ContactFootID, Q, Qd):
        """ modify init values """
        self.q[:] = Q
        self.qdot[:]  = Qd
        _, rot = self.ComputeFramePose(ContactFootID)
        BaseWRTFootOrientationAsMatrix = rot.T
        UnitGravity[2] *=-1
        UnitGravityDerivative[2] *=-1
        print("inited")
        self.TiltandSpeedEstimator.SetInitValues(BaseSpeed, BaseAccG, UnitGravity, UnitGravityDerivative, BaseWRTFootOrientationAsMatrix)


    def ExternalDataCaster(self, DataType:str, ReceivedData) -> None:
        # In case data from elsewhere needs to be converted to another format, or truncated
        if DataType == "acceleration":
            self.a = np.array(ReceivedData)
        #...
        return None


    def InitImuData(self) -> None :
        # initialize data to the right format
        self.a_imu = np.zeros((3,))   
        self.ag_imu = np.array([0, 0, -9.81])            
        self.w_imu = np.zeros((3,)) 
        self.theta_imu = np.array([0, 0, 0, 1])#R.from_euler('xyz', np.zeros(3))
        # angles ? quaternion ?
        self.DeltaTheta = R.from_euler('xyz', np.zeros(3))
        self.DeltaV = np.zeros((3,))
        self.v_imu = np.zeros((3,))


        self.ReferenceOrientation = np.zeros((4,))
        return None

    def InitOutData(self) -> None:
        # initialize estimator out data
        self.a_out = np.zeros((3,)) 
        self.theta_out = np.array([0, 0, 0, 1])
        self.w_out = np.zeros((3,)) 
        self.g_out = np.array([0, 0, -1])

        self.p_out = np.zeros((3,))
        self.v_out = np.zeros((3,)) 

        self.ContactFoot = "none"
        return None

    def InitKinematicData(self) -> None :
        # initialize data to the right format
        # base kinematics
        self.v_kin = np.zeros((3,))
        self.z_kin = np.zeros((1,))
        # motors positions & velocities & torques
        self.q = np.zeros((self.nq, ))
        self.qdot = np.zeros((self.nv, ))
        self.tau = np.zeros((6, ))
        # attitude from Kin
        self.w_kin = np.zeros((3,))
        self.theta_kin = np.array([0, 0, 0, 1])
        # tilt
        self.v_tilt = np.zeros((3,))
        self.g_tilt = np.array([0, 0, -1])
        self.theta_tilt = np.array([0, 0, 0, 1])
        # contact foot switch
        self.SwitchDelta = np.zeros((3,))
        self.AllTimeSwitchDeltas = np.zeros((3,))
        self.PreviousContactFoot = 0
        self.PreviousLeftContact, self.PreviousRightContact = False, False
        self.Switch = False
        self.SwitchLen = 0
        return None

    def InitContactData(self) -> None:
        self.LeftContact = False
        self.RightContact = False
        self.FLContact = np.zeros(3)
        self.FRContact = np.zeros(3)
        return None

    def InitLogMatrixes(self) -> None :
        # initialize data to the right format
        # base velocitie & co, post-filtering logs
        self.log_v_out = np.zeros([3, self.IterNumber])
        self.log_w_out = np.zeros([3, self.IterNumber])
        self.log_a_out = np.zeros([3, self.IterNumber])
        self.log_theta_out = np.zeros([4, self.IterNumber])
        self.log_theta_out[3, :] = np.ones((self.IterNumber, ))
        self.log_g_out = np.zeros([3, self.IterNumber])
        # imu data log
        self.log_v_imu = np.zeros([3, self.IterNumber])
        self.log_w_imu = np.zeros([3, self.IterNumber])
        self.log_a_imu = np.zeros([3, self.IterNumber])
        self.log_theta_imu = np.zeros([4, self.IterNumber])
        self.log_theta_imu[3, :] = np.ones((self.IterNumber, ))
        # forward kinematics data log
        self.log_v_kin = np.zeros([3, self.IterNumber])
        self.log_z_kin = np.zeros([1, self.IterNumber])
        self.log_q = np.zeros([self.nq, self.IterNumber])
        self.log_qdot = np.zeros([self.nv, self.IterNumber])
        self.log_theta_kin = np.zeros([4, self.IterNumber])
        self.log_theta_kin[3, :] = np.ones((self.IterNumber, ))
        self.log_w_kin = np.zeros([3, self.IterNumber])
        
        # tilt log 
        self.log_v_tilt = np.zeros([3, self.IterNumber])
        self.log_g_tilt = np.zeros([3, self.IterNumber])
        self.log_theta_tilt = np.zeros([4, self.IterNumber])
        self.log_theta_tilt[3, :] = np.ones((self.IterNumber, ))
        
        # other logs
        self.log_p_out = np.zeros([3, self.IterNumber])
        self.log_contactforces = np.zeros([6, self.IterNumber])
        
        # Contact switch log
        self.log_switch = np.zeros([3, self.IterNumber])
        
        # time
        self.TimeStamp = np.zeros((self.IterNumber, ))
        
        return None

    def UpdateLogMatrixes(self) -> None :
        LogIter = self.iter
        if self.iter >= self.IterNumber:
            # Logs matrices' size will not be sufficient
            if self.Talkative : self.logger.LogTheLog("Excedind planned number of executions, IterNumber = " + str(self.IterNumber), style="warn", ToPrint=self.Talkative)
            LogIter = self.IterNumber-1

        # update logs with latest data
        # base velocitie & co, post-filtering logs
        self.log_v_out[:, LogIter] = self.v_out[:]
        self.log_w_out[:, LogIter] = self.w_out[:]#self.w_out.as_quat()[:]
        self.log_a_out[:, LogIter] = self.a_out[:]
        self.log_theta_out[:, LogIter] = self.theta_out[:]#.as_quat()[:]
        self.log_g_out[:, LogIter] = self.g_out[:]
        # imu data log
        self.log_v_imu[:, LogIter] = self.v_imu[:]
        self.log_w_imu[:, LogIter] = self.w_imu[:]#self.w_imu.as_quat()[:]
        self.log_a_imu[:, LogIter] = self.a_imu[:]
        self.log_theta_imu[:, LogIter] = self.theta_imu[:]#.as_quat()[:]
        # forward kinematics data log
        self.log_v_kin[:, LogIter] = self.v_kin[:]
        self.log_z_kin[:, LogIter] = self.z_kin[:]
        self.log_q[:, LogIter] = self.q[:]
        self.log_qdot[:, LogIter] = self.qdot[:]
        self.log_theta_kin[:, LogIter] = self.theta_kin[:] #.as_quat()[:]
        self.log_w_kin[:, LogIter] = self.w_kin[:]
        # tilt log 
        self.log_v_tilt[:, LogIter] = self.v_tilt[:]
        self.log_g_tilt[:, LogIter] = self.g_tilt[:]
        self.log_theta_tilt[:, LogIter] = self.theta_tilt[:]
        # other
        self.log_p_out[:, LogIter] = self.p_out[:]
        self.log_contactforces[:3, LogIter] = self.FLContact[:]
        self.log_contactforces[3:, LogIter] = self.FRContact[:]
        # switch
        self.log_switch[:, LogIter] = self.AllTimeSwitchDeltas[:]
        # time
        self.TimeStamp[LogIter] = self.TimeRunning
        return None


    def Get(self, data="acceleration") -> np.ndarray:
        # getter for all internal pertinent data

        # out data getter
        if data=="acceleration" or data=="a":
            return np.copy(self.a_out)
        elif data=="rotation_speed" or data=="w" or data=="omega":
            return np.copy(self.w_out)
        elif data=="attitude" or data=="theta":
            return np.copy(self.theta_out)
        elif data=="attitude_euler" or data=="theta_euler":
            return R.from_quat(self.theta_out.T).as_euler("xyz")
        elif data=="base_position" or data=="p":
            return np.copy(self.p_out)
        elif data=="base_speed" or data=="v":
            return np.copy(self.v_out)
        elif data=="contact_forces" or data=="f":
            ContactForces = np.zeros(6)
            ContactForces[:3] = self.FLContact
            ContactForces[3:] = self.FRContact
            return ContactForces
        elif data=="q":
            return np.copy(self.q)
        elif data=="qdot":
            return np.copy(self.qdot)
        elif data=="tau":
            return np.copy(self.tau)
        elif data=="left_foot_pos":
            # left foot position in base frame
            pos, _ = self.ComputeFramePose(self.FeetIndexes[0])
            return pos
        elif data=="right_foot_pos":
            pos, _ = self.ComputeFramePose(self.FeetIndexes[1])
            return pos
        elif data=="timestamp" or data=="t":
            return self.TimeRunning
        
        # other instantaneous getter
        elif data=="attitude_tilt" or data=="theta_tilt":
            return np.copy(self.theta_tilt)
        

        # logs data getter

        elif data=="acceleration_logs" or data=="a_logs":
            return self.log_a_out
        elif data=="rotation_speed_logs" or data=="w_logs" or data=="omega_logs":
            return self.log_w_out
        elif data=="attitude_logs" or data=="theta_logs":
            return self.log_theta_out
        elif data=="attitude_logs_euler" or data=="theta_logs_euler":
            return R.from_quat(self.log_theta_out.T).as_euler("xyz").T
        elif data=="base_position_logs" or data=="p_logs":
            return self.log_p_out
        elif data=="base_speed_logs" or data=="v_logs":
            return self.log_v_out
        elif data=="contact_forces_logs" or data=="f_logs":
            return self.log_contactforces
        elif data=="q_logs":
            return self.log_q, 
        elif data=="qdot_logs":
            return self.log_qdot
        elif data=="g_out_logs"or data=="g_logs":
            return self.log_g_out

        elif data=="g_tilt_logs":
            return self.log_g_tilt
        elif data=="v_tilt_logs":
            return self.log_v_tilt
        elif data=="theta_tilt_logs":
            return self.log_theta_tilt
        elif data=="theta_tilt_logs_euler":
            return R.from_quat(self.log_theta_tilt.T).as_euler("xyz").T
        
        elif data=="p_switch_logs":
            return self.log_switch
        elif data=="timestamp_logs" or data=="t_logs":
            return self.TimeStamp
        
        # IMU logs data getter
        elif data=="acceleration_logs_imu" or data=="a_logs_imu":
            return self.log_a_imu
        elif data=="rotation_speed_logs_imu" or data=="w_logs_imu" or data=="omega_logs_imu":
            return self.log_w_imu
        elif data=="theta_logs_imu" or data=="attitude_logs_imu":
            return self.log_theta_imu
        elif data=="theta_logs_imu_euler" or data=="attitude_logs_imu_euler":
            return R.from_quat(self.log_theta_imu.T).as_euler("xyz")
        elif data=="base_speed_logs_imu" or data=="v_logs_imu":
            return self.log_v_imu
        # kin logs data getter
        elif data=="rotation_speed_logs_kin" or data=="w_logs_kin" or data=="omega_logs_kin":
            return self.log_w_kin
        elif data=="theta_logs_kin" or data=="attitude_logs_kin":
            return self.log_theta_kin
        elif data=="theta_logs_kin_euler" or data=="attitude_logs_kin_euler":
            return R.from_quat(self.log_theta_kin.T).as_euler("xyz").T
        elif data=="base_speed_logs_kin" or data=="v_logs_kin":
            return self.log_v_kin
        elif data=="v_out_logs" or data=="base_speed_logs_out":
            return self.log_v_out
        # ...
        else :
            self.logger.LogTheLog("Could not get data '" + data + "'. Unrecognised data getter.", style="danger", ToPrint=self.Talkative)
            return None



    def ReadSensor(self) -> None:
        # rotation are updated supposing the value returned by device is xyz euler angles, in radians
        self.device.Read() # FOR TESTING ONLY #PPP
        # base acceleration, acceleration with gravity and rotation speed from IMU
        self.a_imu[:] = self.device.baseLinearAcceleration[:] # COPIED FROM SOLO CODE, CHECK CONSISTENCY WITH BOLT MASTERBOARD
        self.ag_imu[:] = self.device.baseLinearAccelerationGravity[:] # uncertain
        self.w_imu[:] = self.device.baseAngularVelocity[:]
        # integrated data from IMU
        self.DeltaTheta = R.from_euler('xyz', self.device.baseOrientation - self.device.offset_yaw_IMU) # bs, to be found
        self.DeltaV[:] = self.device.baseSpeed[:] - self.device.offset_speed_IMU[:] # bs
        # Kinematic data from encoders
        self.q[:] = self.device.q_mes[:]
        self.qdot[:] = self.device.v_mes[:]
        # torques from motors
        self.tau[:] = self.device.tau_mes[:]

        #self.ExternalDataCaster("acceleration", self.a)


        return None
    
    def ReadExternalSensor(self, 
                            baseLinearAcceleration,
                            baseLinearAccelerationGravity,
                            baseAngularVelocity,
                            q_mes,
                            v_mes,
                            tau_mes) -> None:
                            # acc with g is absolutely needed
        # rotation are updated supposing the value returned by device is xyz euler angles, in radians
        # base acceleration, acceleration with gravity and rotation speed from IMU
        self.a_imu[:] = baseLinearAcceleration[:]# COPIED FROM SOLO CODE, CHECK CONSISTENCY WITH BOLT MASTERBOARD
        self.ag_imu[:] = baseLinearAccelerationGravity[:] # uncertain
        self.w_imu[:] = baseAngularVelocity[:]
        # integrated data from IMU

        # Kinematic data from encoders
        self.q[:] = q_mes[:]
        self.qdot[:] = v_mes[:]
        # torques from motors
        self.tau[:] = tau_mes[:]


        return None

            
    
        

    
    
    def KinematicPosition(self):
        """
        Compute height of base compared to foot in contact
        -------
        p : position of base in world frame, robot-aligned
        """
        if self.LeftContact and self.RightContact:
            pleft, _  = self.ComputeFramePose(self.FeetIndexes[0])
            pright, _ = self.ComputeFramePose(self.FeetIndexes[1])
            p = 0.5  *(pleft + pright)
        elif self.LeftContact :
            p, _ = self.ComputeFramePose(self.FeetIndexes[0])
        elif self.RightContact :
            p, _ = self.ComputeFramePose(self.FeetIndexes[1])
        else :
            p = None
        
        return p
    
    def ComputeFramePose(self, nframe) -> tuple((np.ndarray, np.ndarray)):
        """ get frame 'nframe' pose in base frame 
            ie. {baseframe -> nframe} 
        """
        pin.forwardKinematics(self.robot.model, self.robot.data, self.q)
        pin.updateFramePlacements(self.robot.model, self.robot.data)
        pin.computeAllTerms(self.robot.model, self.robot.data, self.q, self.qdot)
        
        BaseFramePose = self.robot.data.oMf[self.BaseID].inverse()*self.robot.data.oMf[nframe]
        BaseFrameAttitude = np.array(BaseFramePose.rotation).copy()
        BaseFramePosition = np.array(BaseFramePose.translation).copy()
        return BaseFramePosition, BaseFrameAttitude



    def IMUAttitude(self) -> np.ndarray :
        # IMU gives us acceleration and acceleration without gravity
        # measured gravity
        g = self.ag_imu - self.a_imu
        g0 = np.array([0, 0, 9.81])
        # compute the quaternion to pass from g0 to g
        gg0 = utils.cross(g, g0)
        q0 = np.array( [np.linalg.norm(g) * np.linalg.norm(g0) + utils.scalar(g, g0)] )
        #q = R.from_quat( np.concatenate((gg0, q0), axis=0) )
        self.theta_imu = np.concatenate((gg0, q0), axis=0)
        return self.theta_imu #.as_euler('xyz')

    
    def GyroAttitude(self) -> np.ndarray:
        # Uses integrated angular velocity to derive rotation angles 
        # 3DM-CX5-AHRS sensor returns Δθ
        return self.DeltaTheta.as_euler('xyz')


    def IMUSpeed(self) -> np.ndarray:
        # direclty uses IMU data to approximate speed
        return self.ReferenceSpeed + self.DeltaV



    def SpeedFusion(self, mitigate=[0.1, 0.2, 0.7]) -> None:
        """
        
        """
        # uses Kinematic-derived speed estimate and IMU to estimate speed
        self.KinematicSpeed()

        # runs speed and tilt estimator
        if self.LeftContact :
            ContactFootID = self.FeetIndexes[0]
        else :
            ContactFootID = self.FeetIndexes[1]
        # run tilt estimator
        self.v_tilt, self.g_tilt = self.TiltandSpeedEstimator.Estimate(Q=self.q.copy(),
                                            Qd=self.qdot.copy(),
                                            BaseID=1,
                                            ContactFootID=ContactFootID,
                                            ya=self.ag_imu.copy(),
                                            yg=self.w_imu.copy(), 
                                            dt=self.TimeStep)
        
        # filter results
        self.v_out[:] = self.SpeedFilter.RunFilter(self.v_tilt.copy(), self.a_imu.copy())
        # filter speed with data from imu
        if np.linalg.norm(self.ag_imu - self.a_imu)<9:
            if self.Talkative : self.logger.LogTheLog(f"anormal gravity input : {self.ag_imu - self.a_imu} on iter {self.iter}", "warn")

        return None
    


    
    
    
    def TiltfromG(self, g0) -> np.ndarray :
        """
        From estimated g in base frame, get the quaternion between world frame and base frame
        """
        g = g0[:]
        if self.Talkative and np.linalg.norm(g0) > 10:
            self.logger.LogTheLog(f"gravity computed on iter {self.iter} is anormal : {g0}", "warn")
        g = g/np.linalg.norm(g)
        gworld = np.array([0, 0, -1])
        
        # compute the quaternion to pass from gworld to g0
        gg0 = utils.cross(g, gworld)
        q0 = np.array( [np.linalg.norm(g) * np.linalg.norm(gworld) + utils.scalar(g, gworld)] )

        q = np.concatenate((gg0, q0), axis=0)
        if np.linalg.norm(q) == 0 and self.Talkative :
            self.logger.LogTheLog(f"Null norm quaternion computed for gworld -> grobot at iter {self.iter} with gworld {gworld} and g measured {g0}", "danger")
            return np.array([0, 0, 0, 1])    
        return q / np.linalg.norm(q)

        
    
    def CheckQuat(self, q, name="") -> np.ndarray:
        """
        Check if a quat is not of norm 1

        """
        if np.linalg.norm(q) < 1e-6:
            self.logger.LogTheLog(f"Norm of quaternion {name} is NULL : {q} on iter {self.iter}", "danger")
            return np.array([0, 0, 0, 1])
        if np.linalg.norm(q)< 0.99 or  np.linalg.norm(q)> 1.01:
            self.logger.LogTheLog(f"Norm of quaternion {name} is NOT ONE : {q} on iter {self.iter}", "danger")
            return q/np.linalg.norm(q)
        return q
        
    
    
    def Estimate(self, ContactFrame, BaseFrame=1, TimeStep=None):
        """ this is the main function"""
        if TimeStep is not None :
            self.TimeStep = TimeStep
        self.ContactFrame = ContactFrame
        self.BaseFrame = BaseFrame
        
        # update all variables with latest available measurements
        if self.device is not None :
            self.ReadSensor()
        
        # run contact estimator
        self.UpdateContactInformation()
        if self.iter < 3 :
            self.PreviousLeftContact = self.LeftContact
            self.PreviousRightContact = self.RightContact
            self.Switch = False

        # estimate speed
        self.SpeedFusion(mitigate=[0., 0., 1.])
        
        # integrate speed to get position
        self.p_out += self.v_out[:]*self.TimeStep

        # derive data & runs filter

        #self.theta_kin = self.KinematicAttitude()
        self.theta_tilt = self.TiltfromG(self.g_tilt)
        self.theta_out = self.AttitudeFilter.RunFilterQuaternion(self.theta_tilt.copy(), self.w_imu.copy())
        # check that quaternions are not of norm 0
        self.theta_out  =  self.CheckQuat(self.theta_out, "theta_out")
        self.theta_tilt =  self.CheckQuat(self.theta_tilt, "theta_tilt")

        self.g_out = R.from_quat(self.theta_out).apply(np.array([0, 0, -1]))

        self.a_out[:] = self.a_imu[:]
        self.w_out[:] = self.w_imu[:]
        
        # correct z position using kin
        BaseKinPos = self.KinematicPosition()

        if BaseKinPos is not None :
            # at least one foot is touching the ground, correcting base height
            self.p_out[2] =  -BaseKinPos[2] + 0.01 # foot radius
        self.p_out[2] = self.HeightFilter.RunFilter(self.p_out[2], self.v_out[2])

        # correcting x position using kin
        self.FootStepLen(self.LeftContact, self.RightContact, MaxSwitchLen=15)
        if self.TimeRunning > self.T0posDriftComp :
            if self.EndingSwitch and self.StepDuration > 70 :
                # switch just ended, position have been updated & step has dured long enough
                # compute increment to add over next step to compensate drift
                PosDrift = self.AllTimeSwitchDeltas[0] - self.p_out[0]
                sat = 0.05
                if PosDrift < -sat:
                    PosDrift = -sat
                elif PosDrift > sat:
                    PosDrift = sat
                self.PosDriftCompensation = PosDrift/self.StepDuration
            self.p_out[0] += self.PosDriftCompensation
                
                
        

        # update all logs & past variables
        if self.EstimatorLogging : self.UpdateLogMatrixes()
        # count iteration
        if self.iter % 100 == 0 :
            print(f" iter {self.iter} \t dt {self.TimeStep}")
        if self.iter==1 :
            if self.Talkative : self.logger.LogTheLog("executed Estimator for the first time", "subinfo")
        self.iter += 1
        self.TimeRunning += self.TimeStep
        

        return None




















