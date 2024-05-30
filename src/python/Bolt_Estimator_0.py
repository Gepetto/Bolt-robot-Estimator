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

    This code uses Pinocchio and IMU data to provide an estimate of Bolt's
    base attitude and center of mass' speed.

    The Estimator Class contains the different filters and sub-estimators 
    needed to merge the relevant data. Its main method is Estimator.Estimate() .
    
    Documentation can be found at  $ $ $  G I T  $ $ $ 

License, authors, LAAS

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
                parametersSF        : list = [2],
                parametersTI        : list = [1, 1, 1],
                TimeStep            : float = 0.01,
                IterNumber          : int = 1000,
                EstimatorLogging    : bool = True,
                ContactLogging      : bool = True,
                TiltLogging         : bool = True,
                ) -> None:
        
        self.MsgName = "Bolt Estimator v0.6"

        # logging options 
        self.Talkative=Talkative
        self.EstimatorLogging = EstimatorLogging
        self.ContactLogging = ContactLogging
        self.TiltLogging = TiltLogging
        if logger is not None :
            self.logger = logger
        else:
            self.logger = Log("default " + self.MsgName+ " log")
        self.logger.LogTheLog(" Starting log of" + self.MsgName, ToPrint=False)
        if self.Talkative : self.logger.LogTheLog("Initializing " + self.MsgName + "...", style="title", ToPrint=Talkative)
        self.IterNumber = IterNumber
        self.iter = 0
        
        # loading data from file
        if UrdfPath=="" or ModelPath=="":
            self.logger.LogTheLog("No URDF path or ModelPath added !", style="warn", ToPrint=True)
            self.robot = example_robot_data.load("bolt")
        else :
            #model, collision_model, visual_model = pin.buildModelsFromUrdf(UrdfPath, ModelPath, pin.JointModelFreeFlyer())
            self.logger.LogTheLog("Bypassing URDF path or ModelPath", style="warn", ToPrint=True)
            self.robot = example_robot_data.load("bolt")

            self.robot = pin.RobotWrapper.BuildFromURDF(UrdfPath, ModelPath)
            self.logger.LogTheLog("URDF built", ToPrint=Talkative)
        self.nq = self.robot.model.nq
        self.nv = self.robot.model.nv
        self.FeetIndexes = [self.robot.model.getFrameId("FL_FOOT"), self.robot.model.getFrameId("FR_FOOT")] # Left, Right
        self.BaseID = 1

        # interfacing with masterboard (?)
        self.device = device

        # 1 kHz by default
        self.TimeStep = TimeStep
        
        # initializes data & logs with np.zeros arrays
        self.InitImuData()
        self.InitKinematicData()
        self.InitOutData()
        self.InitContactData()
        self.InitLogMatrixes()

        # check that sensors can be read 
        self.ReadSensor()
        if self.Talkative : self.logger.LogTheLog("Sensors read, initial data acquired", ToPrint=Talkative)

        # update height of CoM value, assuming Bolt is vertical
        pin.forwardKinematics(self.robot.model, self.robot.data, self.q)
        pin.updateFramePlacements(self.robot.model, self.robot.data)
        c = np.array((self.robot.data.oMf[self.FeetIndexes[0]].inverse()*self.robot.data.oMf[self.BaseID]).translation).copy()
        self.c_out[2] = np.linalg.norm(c)
        self.c_out[2] += 0.02 # TODO : radius of bolt foot

        self.UpdateLogMatrixes()
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
        self.w_imu = np.zeros((3,)) #R.from_euler('xyz', np.zeros(3))
        self.theta_imu = R.from_euler('xyz', np.zeros(3))
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
        self.theta_out = np.zeros((3,))
        self.w_out = np.zeros((3,)) #R.from_euler('xyz', np.zeros(3))
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
        self.theta_kin = np.zeros(3)#R.from_euler('xyz', np.zeros(3))
        # tilt
        self.v_tilt = np.zeros(3)
        self.g_tilt = np.array([0, 0, -1])
        self.theta_tilt = np.zeros(3)
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
        self.log_theta_out = np.zeros([3, self.IterNumber])
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
        self.log_theta_kin = np.zeros([3, self.IterNumber])
        self.log_w_kin = np.zeros([3, self.IterNumber])
        
        # tilt log 
        self.log_v_tilt = np.zeros([3, self.IterNumber])
        self.log_g_tilt = np.zeros([3, self.IterNumber])
        self.log_theta_tilt = np.zeros([3, self.IterNumber])
        
        # other logs
        self.log_c_out = np.zeros([3, self.IterNumber])
        self.log_cdot_out = np.zeros([3, self.IterNumber])
        self.log_contactforces = np.zeros([6, self.IterNumber])
        
        return None

    def UpdateLogMatrixes(self) -> None :
        if self.iter >= self.IterNumber:
            # Logs matrices' size will not be sufficient
            if self.Talkative : self.logger.LogTheLog("Excedind planned number of executions, IterNumber = " + str(self.IterNumber), style="warn", ToPrint=self.Talkative)

        # update logs with latest data
        # base velocitie & co, post-filtering logs
        self.log_v_out[:, self.iter] = self.v_out[:]
        self.log_w_out[:, self.iter] = self.w_out[:]#self.w_out.as_quat()[:]
        self.log_a_out[:, self.iter] = self.a_out[:]
        self.log_theta_out[:, self.iter] = self.theta_out#.as_quat()[:]
        self.log_g_out[:, self.iter] = self.g_out[:]
        # imu data log
        self.log_v_imu[:, self.iter] = self.v_imu[:]
        self.log_w_imu[:, self.iter] = self.w_imu[:]#self.w_imu.as_quat()[:]
        self.log_a_imu[:, self.iter] = self.a_imu[:]
        self.log_theta_imu[:, self.iter] = self.theta_imu.as_quat()[:]
        # forward kinematics data log
        self.log_v_kin[:, self.iter] = self.v_kin[:]
        self.log_z_kin[:, self.iter] = self.z_kin[:]
        self.log_q[:, self.iter] = self.q[:]
        self.log_qdot[:, self.iter] = self.qdot[:]
        self.log_theta_kin[:, self.iter] = self.theta_kin[:] #.as_quat()[:]
        self.log_w_kin[:, self.iter] = self.w_kin[:]
        # tilt log 
        self.log_v_tilt[:, self.iter] = self.v_tilt[:]
        self.log_g_tilt[:, self.iter] = self.g_tilt[:]
        # other
        self.log_c_out[:, self.iter] = self.c_out[:]
        self.log_cdot_out[:, self.iter] = self.cdot_out[:]
        self.log_contactforces[:3, self.iter] = self.FLContact[:]
        self.log_contactforces[3:, self.iter] = self.FRContact[:]
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
        
        # IMU logs data getter
        elif data=="acceleration_logs_imu" or data=="a_logs_imu":
            return self.log_a_imu
        elif data=="rotation_speed_logs_imu" or data=="w_logs_imu" or data=="omega_logs_imu":
            return self.log_w_imu
        elif data=="theta_logs_imu" or data=="attitude_logs_imu":
            return self.log_theta_imu
        elif data=="base_speed_logs_imu" or data=="v_logs_imu":
            return self.log_v_imu
        # kin logs data getter
        elif data=="rotation_speed_logs_kin" or data=="w_logs_kin" or data=="omega_logs_kin":
            return self.log_w_kin
        elif data=="theta_logs_kin" or data=="attitude_logs_kin":
            return self.log_theta_kin
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
        # data from forward kinematics
        self.v_kin[:] = np.zeros(3)[:]
        self.z_kin[:] = np.zeros(1)[:]
        # torques from motors
        self.tau[:] = self.device.tau_mes[:]

        #self.ExternalDataCaster("acceleration", self.a)

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
        q = R.from_quat( np.concatenate((gg0, q0), axis=0) )
        self.theta_imu = q
        return self.theta_imu.as_euler('xyz')

    
    def GyroAttitude(self) -> np.ndarray:
        # Uses integrated angular velocity to derive rotation angles 
        # 3DM-CX5-AHRS sensor returns Δθ
        return self.DeltaTheta.as_euler('xyz')

    
    # TODO : mod 
    def AttitudeFusion(self, alpha=1) -> None :
        # uses AttitudeFusion_AG and AttitudeFusion_KG to provide attitude estimate
        
        # uses attitude from direction of gravity estimate and gyro data to provide attitude estimate
        AttitudeFromIMU = self.IMUAttitude()
        self.theta_out = R.from_euler('xyz', self.AttitudeFilter.RunFilter(AttitudeFromIMU, self.w_imu.copy()))
            
        # uses attitude Kinematic estimate and gyro data to provide attitude estimate
        #AttitudeFromKin = self.KinematicAttitude()
        self.theta_out_kg = self.theta_out_ag#R.from_euler('xyz', self.AttitudeFilter.RunFilter(AttitudeFromKin, self.w_imu))
        
        # average both
        self.theta_out = R.from_euler('xyz', alpha*self.theta_out_ag.as_euler('xyz') + (1-alpha)*self.theta_out_kg.as_euler('xyz'))
        return None


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
            v_avg = np.mean(self.log_v_kin[:, max(0, self.iter-10):self.iter-1], axis=1)
            w_avg = np.mean(self.log_w_kin[:, max(0, self.iter-10):self.iter-1], axis=1)
            self.w_kin = w_avg
            self.v_kin = v_avg
        
        # filter speed
        #self.v_kin = self.SpeedFilter.RunFilter(self.v_kin, self.a_imu)

        return self.v_kin, self.w_kin


    def SpeedFusion(self, mitigate=[0.1, 0.2, 0.7]) -> None:
        # uses Kinematic-derived speed estimate and IMU to estimate speed
        self.KinematicSpeed()

        # runs speed and tilt estimator
        if self.LeftContact :
            ContactFootID = self.FeetIndexes[0]
        else :
            ContactFootID = self.FeetIndexes[1]
        self.v_tilt, self.g_tilt = self.TiltandSpeedEstimator.Estimate(Q=self.q,
                                            Qd=self.qdot,
                                            BaseID=1,
                                            ContactFootID=ContactFootID,
                                            ya=self.ag_imu,
                                            yg=self.w_imu, 
                                            dt=self.TimeStep)

        v_out = mitigate[0]*self.v_imu + mitigate[1]*self.v_kin + mitigate[2]*self.v_tilt
        self.v_out  = self.SpeedFilter.RunFilter(v_out, self.a_imu)
        if np.linalg.norm(self.ag_imu - self.a_imu)<9:
            if self.Talkative : self.logger.LogTheLog(f"anormal gravity input : {self.ag_imu - self.a_imu} on iter {self.iter}", "warn")

        return None
    


    def TiltfromG(self, g0) -> np.ndarray:
        """
        From estimated g in base frame, get the euler angles between world frame and base frame
        """
        g = g0[:]#*np.array([9.81, 1, -9.81])
        g = g/np.linalg.norm(g)
        gworld = np.array([0, 0, 1])

        v = np.cross(gworld, g)
        s = np.linalg.norm(v)
        c = utils.scalar(gworld, g0)
        if c != -1 :
            RotM = np.eye(3) + utils.S(v) + utils.S(v**2)*(1/(1+c))
        else :
            RotM = -np.eye(3) # TODO : check
            if self.Talkative : self.logger.LogTheLog("Could not compute g rot matrix on iter "+self.iter, "warn")

        euler = R.from_matrix(RotM).as_euler("xyz")

        return euler
    
    
    def Estimate(self, dt=None):
        """ this is the main function"""
        if dt is not None :
            self.TimeStep = dt
        
        # update all variables with latest available measurements
        self.ReadSensor()
        # run contact estimator
        self.UpdateContactInformation()
        # estimate speed
        self.SpeedFusion(mitigate=[0., 0., 1.])
        
        # integrate speed to get position
        self.c_out += self.v_out[0, :]*self.TimeStep

        # derive data & runs filter

        #self.theta_kin = self.KinematicAttitude()

        self.theta_tilt = self.TiltfromG(self.g_tilt)
        self.theta_out = self.AttitudeFilter.RunFilter(self.theta_tilt.copy(), self.w_imu.copy())
        rotmat = R.from_euler("xyz", self.theta_out.copy()).as_matrix()
        self.g_out = rotmat @ np.array([0, 0, -1])

        # update all logs & past variables
        self.UpdateLogMatrixes()
        # count iteration
        if self.iter==1 :
            self.logger.LogTheLog("executed Estimator for the first time", "subinfo")
        self.iter += 1
        

        return None



