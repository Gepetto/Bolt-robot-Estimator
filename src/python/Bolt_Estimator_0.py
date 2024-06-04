import numpy as np
import pinocchio as pin
import time as t
from scipy.spatial.transform import Rotation as R
import example_robot_data


from Bolt_Utils import utils
from Bolt_Utils import Log

from Bolt_ContactEstimator import ContactEstimator
from Bolt_TiltEstimator import TiltEstimator

from Bolt_Filter import Filter
from Bolt_Filter_Complementary import ComplementaryFilter
from Bolt_Filter_Benallegue import BenallegueEstimator


"""
An estimator for Bolt Bipedal Robot

    This code uses Pinocchio, encoders and IMU data to provide an estimate of Bolt's
    base attitude, base speed and position, and center of mass' speed.

    The Estimator Class contains the different filters and sub-estimators 
    needed to merge the relevant data. Its main method is Estimator.Estimate() .

    The estimators called by Estimator are : ContactEstimator and TiltEstimator.
    
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
                parametersTI        : list = [10, 60, 2],
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
                parametersTI        (list of float) parameters of the tilt estimator, list of three float (alpha1, alpha2, gamma)
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
        # iteration number
        self.IterNumber = IterNumber
        self.iter = 0
        
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

        # 1 kHz by default
        self.TimeStep = TimeStep
        
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
        self.c_out[2] = np.linalg.norm(c)
        self.c_out[2] += 0.02 # TODO : radius of bolt foot

        if self.EstimatorLogging : self.UpdateLogMatrixes()
        self.iter += 1
        if self.Talkative : self.logger.LogTheLog("Initial data stored in logs", ToPrint=Talkative)
        

        # filter parameters
        parametersAF = [self.TimeStep] + parametersAF
        parametersSF = [self.TimeStep] + parametersSF
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


        # returns info on Slips, Contact Forces, Contact with the ground
        self.ContactEstimator = ContactEstimator(robot=self.robot, 
                                                 LeftFootFrameID=self.FeetIndexes[0], 
                                                 RightFootFrameID=self.FeetIndexes[1], 
                                                 LeftKneeFrameID=7, # self.robot.model.getFrameId("FL_KNEE"),
                                                 RightKneeFrameID=15, # self.robot.model.getFrameId("FR_KNEE"),
                                                 LeftKneeTorqueID=2,
                                                 RightKneeTorqueID=5,
                                                 IterNumber=self.IterNumber,
                                                 dt=self.TimeStep,
                                                 MemorySize=5,
                                                 Logging=self.ContactLogging,
                                                 Talkative=self.Talkative,
                                                 logger=self.logger)
        if self.Talkative : self.logger.LogTheLog("Contact Estimator added.", ToPrint=Talkative)
        
        # returns info on Slips, Contact Forces, Contact with the ground
        self.TiltandSpeedEstimator = TiltEstimator(robot=self.robot,
                                                   Q0=self.q,
                                                   Qd0=self.qdot,
                                                   Niter=self.IterNumber,
                                                   Logging=self.TiltLogging,
                                                   params=parametersTI)
        
        if self.Talkative : self.logger.LogTheLog("Tilt Estimator added with parameters " + str(parametersTI), ToPrint=Talkative)
        
        # returns info on foot attitude
        self.FootAttitudeEstimator = BenallegueEstimator(parameters=[0.001, 2],
                                                         dt=self.TimeStep,
                                                         name="Foot Attitude Estimator",
                                                         talkative=Talkative,
                                                         logger=self.logger)
        if self.Talkative : self.logger.LogTheLog("Foot Attitude Estimator added with parameters " + str(0), ToPrint=Talkative)
        

        self.logger.LogTheLog(self.MsgName +" initialized successfully.", ToPrint=Talkative)
        return None


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
        self.v_out = np.zeros((3,)) 
        self.a_out = np.zeros((3,)) 
        #self.theta_out = R.from_euler('xyz', np.zeros(3))
        self.theta_out = np.array([0, 0, 0, 1])
        self.w_out = np.zeros((3,)) 
        self.g_out = np.array([0, 0, -1])

        self.c_out = np.zeros((3,))
        self.cdot_out = np.zeros((3,)) 
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
        self.log_g_out = np.zeros([3, self.IterNumber])
        # imu data log
        self.log_v_imu = np.zeros([3, self.IterNumber])
        self.log_w_imu = np.zeros([3, self.IterNumber])
        self.log_a_imu = np.zeros([3, self.IterNumber])
        self.log_theta_imu = np.zeros([4, self.IterNumber])
        # forward kinematics data log
        self.log_v_kin = np.zeros([3, self.IterNumber])
        self.log_z_kin = np.zeros([1, self.IterNumber])
        self.log_q = np.zeros([self.nq, self.IterNumber])
        self.log_qdot = np.zeros([self.nv, self.IterNumber])
        self.log_theta_kin = np.zeros([4, self.IterNumber])
        self.log_w_kin = np.zeros([3, self.IterNumber])
        
        # tilt log 
        self.log_v_tilt = np.zeros([3, self.IterNumber])
        self.log_g_tilt = np.zeros([3, self.IterNumber])
        self.log_theta_tilt = np.zeros([4, self.IterNumber])
        
        # other logs
        self.log_c_out = np.zeros([3, self.IterNumber])
        self.log_cdot_out = np.zeros([3, self.IterNumber])
        self.log_contactforces = np.zeros([6, self.IterNumber])
        
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
        self.log_c_out[:, LogIter] = self.c_out[:]
        self.log_cdot_out[:, LogIter] = self.cdot_out[:]
        self.log_contactforces[:3, LogIter] = self.FLContact[:]
        self.log_contactforces[3:, LogIter] = self.FRContact[:]
        return None


    def Get(self, data="acceleration") -> np.ndarray:
        # getter for all internal pertinent data

        # out data getter
        if data=="acceleration" or data=="a":
            return self.a_out
        elif data=="rotation_speed" or data=="w" or data=="omega":
            return self.w_out
        elif data=="attitude" or data=="theta":
            return self.theta_out
        elif data=="attitude_euler" or data=="theta_euler":
            return R.from_quat(self.theta_out.T).as_euler("xyz")
        elif data=="com_position" or data=="c":
            return self.c_out
        elif data=="com_speed" or data=="cdot":
            return self.cdot_out
        elif data=="base_speed" or data=="v":
            return self.v_out
        elif data=="contact_forces" or data=="f":
            ContactForces = np.zeros(6)
            ContactForces[:3] = self.FLContact
            ContactForces[3:] = self.FRContact
            return ContactForces
        elif data=="q":
            return self.q, 
        elif data=="qdot":
            return self.qdot
        elif data=="tau":
            return self.tau
        

        # logs data getter

        elif data=="acceleration_logs" or data=="a_logs":
            return self.log_a_out
        elif data=="rotation_speed_logs" or data=="w_logs" or data=="omega_logs":
            return self.log_w_out
        elif data=="attitude_logs" or data=="theta_logs":
            return self.log_theta_out
        elif data=="attitude_logs_euler" or data=="theta_logs_euler":
            return R.from_quat(self.log_theta_out.T).as_euler("xyz").T
        elif data=="com_position_logs" or data=="c_logs":
            return self.log_c_out
        elif data=="com_speed_logs" or data=="cdot_logs":
            return self.log_cdot_out
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
            self.logger.LogTheLog("Could not get data '" + data + "'. Unrecognised data getter.", style="warn", ToPrint=self.Talkative)
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
    

    def UpdateContactInformation(self):
        """ get contact information from Contact Estimator"""
        # boolean contact
        self.LeftContact, self.RightContact = self.ContactEstimator.LegsOnGround(self.q, 
                                                                                 self.qdot,
                                                                                 self.a_imu, 
                                                                                 self.tau,
                                                                                 self.g_tilt,
                                                                                 TorqueForceMingler=1.0, 
                                                                                 ProbThresold=0.45, 
                                                                                 TrustThresold=0.5
                                                                                 )
        # contact forces
        self.FLContact, self.RLContact = self.ContactEstimator.Get("current_cf_averaged")


    
    # TODO : update with benallegue
    def KinematicAttitude(self) -> np.ndarray:
        # uses robot model and rotation speed to provide attitude estimate based on encoder data
        
        # consider the right contact frames, depending on which foot is in contact with the ground
        if self.LeftContact and self.RightContact :
            if self.Talkative : self.logger.LogTheLog("Both feet are touching the ground", style="warn", ToPrint=self.Talkative)
            ContactFrames = [0, 1]
        elif self.LeftContact :
            if self.Talkative : self.logger.LogTheLog("left foot touching the ground", ToPrint=self.Talkative)
            ContactFrames = [0]
        elif self.RightContact :
            if self.Talkative : self.logger.LogTheLog("right foot touching the ground", ToPrint=self.Talkative)
            ContactFrames = [1]
        else :
            self.logger.LogTheLog("No feet are touching the ground", style="warn", ToPrint=self.Talkative)
            ContactFrames = []

        # Compute the base's attitude for each foot in contact
        FrameAttitude = []
        
        pin.forwardKinematics(self.robot.model, self.robot.data, self.q)
        pin.updateFramePlacements(self.robot.model, self.robot.data)
        pin.computeAllTerms(self.robot.model, self.robot.data, self.q, self.qdot)
        
        if self.LeftContact :
            ContactFootID = self.FeetIndexes[0]
        else :
            ContactFootID = self.FeetIndexes[1]
        BaseID = 1
        
        for foot in ContactFrames:
            
            # attitude from foot to base
            FootBasePose = self.robot.data.oMf[ContactFootID].inverse()*self.robot.data.oMf[BaseID]
            FootBaseAttitude = np.array(FootBasePose.rotation).copy()
            FootBasePosition = np.array(FootBasePose.translation).copy()
            
            # attitude of the foot
            WorldFootAttitude = self.FootAttitudeEstimator.RunFilter(IMUKinPos=FootBasePosition, IMUKinRot=FootBaseAttitude, ya=self.ag_imu, yg=self.w_imu)
            
            
            # combined attitude
            WorldBaseAttitude = WorldFootAttitude + FootBaseAttitude
        
        self.theta_kin = R.from_euler(WorldBaseAttitude)

        #return self.theta_kin.as_euler('xyz')
        return WorldBaseAttitude



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

    
    def KinematicSpeed(self) -> tuple((np.ndarray, np.ndarray)):
        # uses Kinematic data
        # along with contact and rotation speed information to approximate speed

        # consider the right contact frames, depending on which foot is in contact with the ground
        if self.LeftContact and self.RightContact :
            if self.Talkative : self.logger.LogTheLog("Both feet are touching the ground on iter " + str(self.iter), style="warn", ToPrint=self.Talkative)
            ContactFrames = [0,1]
        elif self.LeftContact :
            if self.Talkative : self.logger.LogTheLog("left foot touching the ground", ToPrint=False)
            ContactFrames = [0]
        elif self.RightContact :
            if self.Talkative : self.logger.LogTheLog("right foot touching the ground", ToPrint=False)
            ContactFrames = [1]
        else :
            self.logger.LogTheLog("No feet are touching the ground on iter " + str(self.iter), style="warn", ToPrint=self.Talkative)
            ContactFrames = []

        # Compute the base's speed for each foot in contact
        FrameSpeed = []
        FrameRotSpeed = []

        pin.forwardKinematics(self.robot.model, self.robot.data, self.q)
        pin.updateFramePlacements(self.robot.model, self.robot.data)
        pin.computeAllTerms(self.robot.model, self.robot.data, self.q, self.qdot)

        for ContactFootID in ContactFrames:
            # speed of Base wrt its immobile foot
            oMf = self.robot.data.oMf[ContactFootID]
            c_speed_l = oMf.inverse().action @ pin.getFrameVelocity(self.robot.model, self.robot.data, self.BaseID, pin.WORLD)
            speed = np.array(c_speed_l[:3]).copy()
            # rotation speed of base frame in contact foot frame
            omega = np.array(c_speed_l[3:]).copy()

            FrameSpeed.append(speed)
            FrameRotSpeed.append(omega)
        
        if self.LeftContact and self.RightContact :
            # averages results
            self.v_kin = np.mean(np.array(FrameSpeed), axis=0)
            self.w_kin = np.mean(np.array(FrameRotSpeed), axis=0)
        elif self.LeftContact or self.RightContact :
            # one foot in contact
            self.v_kin = np.array(FrameSpeed)
            self.w_kin = np.array(FrameRotSpeed)
        else :
            # no foot touching the ground, keeping old speed data
            if self.EstimatorLogging : 
                v_avg = np.mean(self.log_v_kin[:, max(0, self.iter-10):self.iter-1], axis=1)
                w_avg = np.mean(self.log_w_kin[:, max(0, self.iter-10):self.iter-1], axis=1)
            else :
                v_avg, w_avg = self.v_kin, self.w_kin
            self.w_kin = w_avg
            self.v_kin = v_avg
        
        # filter speed
        #self.v_kin = self.SpeedFilter.RunFilter(self.v_kin, self.a_imu)

        return self.v_kin, self.w_kin


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
        self.v_tilt, self.g_tilt = self.TiltandSpeedEstimator.Estimate(Q=self.q,
                                            Qd=self.qdot,
                                            BaseID=1,
                                            ContactFootID=ContactFootID,
                                            ya=self.ag_imu,
                                            yg=self.w_imu, 
                                            dt=self.TimeStep)

        v_out = self.v_tilt
        # filter speed with data from imu

        self.v_out  = np.reshape(self.v_tilt, (1, 3)) #self.SpeedFilter.RunFilter(v_out, self.a_imu)
        if np.linalg.norm(self.ag_imu - self.a_imu)<9:
            if self.Talkative : self.logger.LogTheLog(f"anormal gravity input : {self.ag_imu - self.a_imu} on iter {self.iter}", "warn")

        return None
    


    def TiltfromG_(self, g0) -> np.ndarray:
        """
        From estimated g in base frame, get the euler angles between world frame and base frame
        """
        g = g0[:]
        g = g/np.linalg.norm(g)
        gworld = np.array([0, 0, 1])

        v = np.cross(gworld, g)
        s = np.linalg.norm(v)
        c = utils.scalar(gworld, g)
        if c != -1 :
            RotM = np.eye(3) + utils.S(v) + utils.S(v**2)*(1/(1+c))
        else :
            RotM = -np.eye(3) # TODO : check
            if self.Talkative : self.logger.LogTheLog("Could not compute g rot matrix on iter "+self.iter, "warn")

        euler = R.from_matrix(RotM).as_euler("xyz")

        return euler
    
    def TiltfromG(self, g0) -> np.ndarray :
        """
        From estimated g in base frame, get the quaternion between world frame and base frame
        """
        g = g0[:]
        g = g/np.linalg.norm(g)
        gworld = np.array([0, 0, -1])
        
        # compute the quaternion to pass from gworld to g0
        gg0 = utils.cross(g, gworld)
        q0 = np.array( [np.linalg.norm(g) * np.linalg.norm(gworld) + utils.scalar(g, gworld)] )
        
        if q0==0 or np.linalg.norm(gg0) == 0:
            print("norme nulle à iter " + str(self.iter))
            print(g)
            print(gg0)
            print(gworld)
            print(q0)
        q = np.concatenate((gg0, q0), axis=0)        
        return q / np.linalg.norm(q)

        
    
    def CheckQuat(self, q, name=""):
        """
        Check if a quat is not of norm 1

        """
        if np.sum(q) == 0:
            self.logger.LogTheLog(f"Norm of quaternion {name} is NULL : {q} on iter {self.iter}", "warn")
            return False
        if np.linalg.norm(q)< 0.99 or  np.linalg.norm(q)> 1.01:
            self.logger.LogTheLog(f"Norm of quaternion {name} is NOT ONE : {q} on iter {self.iter}", "warn")
            return False
        return True
        
    
    
    def Estimate(self, dt=None):
        """ this is the main function"""
        if dt is not None :
            self.TimeStep = dt
        
        # update all variables with latest available measurements
        if self.device is not None :
            self.ReadSensor()
        
        # run contact estimator
        self.UpdateContactInformation()
        # estimate speed
        self.SpeedFusion(mitigate=[0., 0., 1.])
        
        # integrate speed to get position
        if self.v_out.shape == (3,):
            self.v_out.shape = (1, 3)
        self.c_out += self.v_out[0, :]*self.TimeStep # TODO bizarre les dimensions

        # derive data & runs filter

        #self.theta_kin = self.KinematicAttitude()
        self.theta_tilt = self.TiltfromG(self.g_tilt)
        #theta_tilt = R.from_quat(self.theta_tilt).as_euler("xyz")
        self.theta_out = self.AttitudeFilter.RunFilterQuaternion(self.theta_tilt.copy(), self.w_imu.copy())
        self.g_out = R.from_quat(self.theta_out).apply(np.array([0, 0, -1]))

        self.a_out = self.a_imu
        
        
        self.CheckQuat(self.theta_out, "theta_out")
        self.CheckQuat(self.theta_tilt, "theta_tilt")

        # update all logs & past variables
        if self.EstimatorLogging : self.UpdateLogMatrixes()
        # count iteration
        if self.iter % 50 == 0 :
            print(str(self.iter))
        if self.iter==1 :
            self.logger.LogTheLog("executed Estimator for the first time", "subinfo")
        self.iter += 1
        

        return None




















