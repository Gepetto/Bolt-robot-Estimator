import numpy as np
import pinocchio as pin
import time as t
from scipy.spatial.transform import Rotation as R
import example_robot_data


from Bolt_Utils import utils
from Bolt_Utils import Log

from Bolt_ContactEstimator import ContactEstimator
from Bolt_Filter import Filter
from Bolt_Filter_Complementary import ComplementaryFilter


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
                TimeStep            : float = 0.01,
                IterNumber          : int = 1000) -> None:

        self.MsgName = "Bolt Estimator v0.5"
        self.Talkative=Talkative
        if logger is not None :
            self.logger = logger
        else:
            self.logger = Log("default " + self.MsgName+ " log")
        self.logger.LogTheLog(" Starting log of" + self.MsgName, ToPrint=False)
        self.logger.LogTheLog("Initializing " + self.MsgName + "...", style="title", ToPrint=Talkative)
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
        self.logger.LogTheLog("Sensors read, initial data acquired", ToPrint=Talkative)
        self.UpdateLogMatrixes()
        self.iter += 1
        self.logger.LogTheLog("Initial data stored in logs", ToPrint=Talkative)
        

        # filter parameters
        parametersAF = [self.TimeStep] + parametersAF
        parametersSF = [self.TimeStep] + parametersSF
        # set desired filters types for attitude and speed
        # for the time being, complementary only
        if AttitudeFilterType=="complementary":
            self.AttitudeFilter = ComplementaryFilter(parameters=parametersAF, name="attitude complementary filter", talkative=Talkative, logger=self.logger, ndim=3)
        self.logger.LogTheLog("Attitude Filter of type '" + AttitudeFilterType + "' added.", ToPrint=Talkative)
        
        if SpeedFilterType=="complementary":
            self.SpeedFilter = ComplementaryFilter(parameters=parametersSF, name="speed complementary filter", talkative=Talkative, logger=self.logger, ndim=3)
        self.logger.LogTheLog("Speed Filter of type '" + SpeedFilterType + "' added.", ToPrint=Talkative)

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
                                                 logger=self.logger)
        self.logger.LogTheLog("Contact Estimator added.", ToPrint=Talkative)

        
        #self.AllTimeAcceleration, self.AllTimeq = np.zeros((3, self.MemorySize)), np.zeros((3, self.MemorySize))
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
        self.ag_imu = np.array([0, 0, 10])            
        self.w_imu = np.zeros((3,)) #R.from_euler('xyz', np.zeros(3))
        self.theta_imu = R.from_euler('xyz', np.zeros(3))
        # angles ? quaternion ?
        self.DeltaTheta = R.from_euler('xyz', np.zeros(3))
        self.DeltaV = np.zeros((3,))
        self.v_imu = np.zeros((3,))


        self.ReferenceOrientation = np.zeros((4,))
        return None

    def InitOutData(self) -> None:
        # initialize filter out data
        self.v_out = np.zeros((3,)) 
        self.a_out = np.zeros((3,)) 
        self.theta_out = R.from_euler('xyz', np.zeros(3)) 
        self.w_out = np.zeros((3,)) #R.from_euler('xyz', np.zeros(3))

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
        self.w_kin = R.from_euler('xyz', np.zeros(3))
        self.theta_kin = R.from_euler('xyz', np.zeros(3))
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
        self.log_w_kin = np.zeros([4, self.IterNumber])
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
        self.log_theta_out[:, self.iter] = self.theta_out.as_quat()[:]
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
        self.log_theta_kin[:, self.iter] = self.theta_kin.as_quat()[:]
        self.log_w_kin[:, self.iter] = self.w_kin.as_quat()[:]
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
        # ...
        else :
            self.logger.LogTheLog("Could not get data '" + data + "'. Unrecognised data getter.", style="warn", ToPrint=self.Talkative)
            return None



    def ReadSensor(self) -> None:
        # rotation are updated supposing the value returned by device is xyz euler angles, in radians
        self.device.Read() # FOR TESTING ONLY #PPP
        # base acceleration, acceleration with gravity and rotation speed from IMU
        self.a_imu[:] = self.device.baseLinearAcceleration[:] # COPIED FROM SOLO CODE, CHECK CONSISTENCY WITH BOLT MASTERBOARD
        self.ag_imu[:] = self.device.baseLinearAccelerationGravity[:] # bs
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
    


    def UpdateContactInformation(self, TypeOfContactEstimator="default"):

        if TypeOfContactEstimator=="default":
            self.LeftContcat, self.RightContact = self.ContactEstimator.LegsOnGround(self.q, 
                                                                                     self.qdot,
                                                                                     self.a_imu, 
                                                                                     self.tau,
                                                                                     self.a_imu - self.ag_imu + np.array([0, 0, 10]),
                                                                                     TorqueForceMingler=0.75, 
                                                                                     ProbThresold=0.5, 
                                                                                     TrustThresold=0.5
                                                                                     )
            
        elif TypeOfContactEstimator=="kin":
            self.LeftContcat, self.RightContact = self.ContactEstimator.LegsOnGroundKin(self.q, self.a_imu - self.ag_imu)

    
    def KinematicAttitude(self) -> np.ndarray:
        # uses robot model and rotation speed to provide attitude estimate based on encoder data
        
        # consider the right contact frames, depending on which foot is in contact with the ground
        if self.LeftContact and self.RightContact :
            self.logger.LogTheLog("Both feet are touching the ground", style="warn", ToPrint=self.Talkative)
            ContactFrames = [0,1]
        elif self.LeftContact :
            self.logger.LogTheLog("left foot touching the ground", ToPrint=self.Talkative)
            ContactFrames = [0]
        elif self.RightContact :
            self.logger.LogTheLog("right foot touching the ground", ToPrint=self.Talkative)
            ContactFrames = [1]
        else :
            self.logger.LogTheLog("No feet are touching the ground", style="warn", ToPrint=self.Talkative)
            ContactFrames = []

        # Compute the base's attitude for each foot in contact
        FrameAttitude = []
        for foot in ContactFrames:
            # attitude f the foot
            
            # attitude from foot to base
            pin.forwardKinematics(self.q, [self.v,[self.a]])
            FrameAttitude.append( - pin.updateFramePlacement(self.model, self.data, self.FeetIndexes[foot]).rotation)
            
            # combined attitude
        
        if self.LeftContact or self.RightContact :
            # averages results
            self.theta_kin = np.mean(np.array(FrameAttitude))
        else :
            # no foot touching the ground, keeping old attitude data
            self.theta_kin = self.theta_kin

        return self.theta_kin.as_euler('xyz') 



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

    
    
    def AttitudeFusion(self, alpha=1) -> None :
        # uses AttitudeFusion_AG and AttitudeFusion_KG to provide attitude estimate
        
        # uses attitude from direction of gravity estimate and gyro data to provide attitude estimate
        AttitudeFromIMU = self.IMUAttitude()
        self.theta_out_ag = R.from_euler('xyz', self.AttitudeFilter.RunFilter(AttitudeFromIMU, self.w_imu))
            
        # uses attitude Kinematic estimate and gyro data to provide attitude estimate
        AttitudeFromKin = self.KinematicAttitude()
        self.theta_out_kg = R.from_euler('xyz', self.AttitudeFilter.RunFilter(AttitudeFromKin, self.w_imu))
        
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
            self.logger.LogTheLog("Both feet are touching the ground", style="warn", ToPrint=self.Talkative)
            ContactFrames = [0,1]
        elif self.LeftContact :
            self.logger.LogTheLog("left foot touching the ground", ToPrint=self.Talkative)
            ContactFrames = [0]
        elif self.RightContact :
            self.logger.LogTheLog("right foot touching the ground", ToPrint=self.Talkative)
            ContactFrames = [1]
        else :
            self.logger.LogTheLog("No feet are touching the ground", style="warn", ToPrint=self.Talkative)
            ContactFrames = []

        # Compute the base's speed for each foot in contact
        FrameSpeed = []
        FrameRotSpeed = []
        for foot in ContactFrames:
            pin.forwardKinematics(self.q, [self.v,[self.a]])
            # rotation and translation speed of Base wrt its immobile foot
            FrameSpeed.append(pin.getFrameVelocity(self.model, self.data, self.BaseIndex, self.FeetIndexes[foot]).translation)
            FrameRotSpeed.append(pin.getFrameVelocity(self.model, self.data, self.BaseIndex, self.FeetIndexes[foot]).rotation)
        
        if self.LeftContact or self.RightContact :
            # averages results
            self.v_kin = np.mean(np.array(FrameSpeed))
            self.w_kin = np.mean(np.array(FrameRotSpeed))
        else :
            # no foot touching the ground, keeping old speed data
            self.w_kin = self.w_kin
            self.v_kin = self.v_kin

        return self.v_kin, self.w_kin


    def SpeedFusion(self) -> None:
        # uses Kinematic-derived speed estimate and gyro (?) to estimate speed
        return None
    
    
    def Estimate(self):
        # this is the main function
        # updates all variables with latest available measurements
        self.ReadSensor()
        self.UpdateContactInformation()


        # derive data & runs filter
        #PPP self.SpeedFusion()

        #self.AttitudeFusion()

        # update all logs & past variables
        self.UpdateLogMatrixes()
        # count iteration
        self.iter += 1

        return None



