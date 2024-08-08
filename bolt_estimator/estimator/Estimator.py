import numpy as np
import pinocchio as pin
import time as t
from scipy.spatial.transform import Rotation as R
import example_robot_data


from bolt_estimator.utils.Utils import utils
from bolt_estimator.utils.Utils import Log

from bolt_estimator.estimator.contact_estimator import contact_estimator
from bolt_estimator.estimator.TiltEstimator import TiltEstimator

from bolt_estimator.estimator.Filter import Filter
from bolt_estimator.estimator.Filter_Complementary import ComplementaryFilter
from bolt_estimator.estimator.foot_attitude_estimator import foot_attitude_estimator


"""
An estimator for Bolt Bipedal Robot

    This code uses Pinocchio, encoders and IMU data to provide an estimate of Bolt's
    base attitude, base speed and position, and center of mass' speed.

    The Estimator Class contains the different filters and sub-estimators 
    needed to merge the relevant data. Its main method is Estimator.Estimate() .

    The estimators called by Estimator are : contact_estimator and TiltEstimator.
    
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
                model_path           : str = "",
                urdf_path            : str = "",
                talkative           : bool = True,
                logger              : Log = None,
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
        
        self.msg_name = "Bolt Estimator"

        # logging options 
        self.talkative=talkative
        self.estimator_logging = estimator_logging
        self.contact_logging = contact_logging
        self.tilt_logging = tilt_logging
        # adding logger
        if logger is not None :
            self.logger = logger
            self.logger.print_on_flight = self.talkative
        else:
            self.logger = Log("default " + self.msg_name+ " log", print_on_flight=self.talkative)
        self.logger.LogTheLog(" Starting log of" + self.msg_name, to_print=False)
        self.logger.LogTheLog("Initializing " + self.msg_name + "...", style="title", to_print=talkative)
        
        # iteration and timing
        self.iter_number = iter_number
        self.iter = 0
        self.time_running = 0.
        # 1 kHz by default
        self.time_step = time_step
        
        # loading data from file
        if  model_path=="":#or urdf_path=="" :
            self.logger.LogTheLog("No URDF path or model_path addeds", style="warn", to_print=self.talkative)
            self.robot = example_robot_data.load("bolt")
        else :
            #model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path, model_path, pin.JointModelFreeFlyer())
            self.logger.LogTheLog("Bypassing URDF path or model_path", style="warn", to_print=self.talkative)
            self.robot = example_robot_data.load(model_path)

            #self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path, model_path)
            #self.logger.LogTheLog("URDF built", to_print=talkative)
        # adding IMU frame with default data
        self.imu_base_transform = pin.SE3.Identity()
        self.imu_base_quat = np.array([0, 0, 0, 1])

        # number of frames and movement
        self.nq = self.robot.model.nq
        self.nv = self.robot.model.nv
        # useful frame indexes
        self.feet_indexes = [self.robot.model.getFrameId("FL_FOOT"), self.robot.model.getFrameId("FR_FOOT")] # Left, Right
        self.base_id = 1
        self.base_joint_id = 1

        # interfacing with masterboard (?)
        if device is not None :
            self.device = device
        else :
            self.logger.LogTheLog("No device added", style="warn", to_print=self.talkative)
            self.device=None
        
        # initializes data & logs with np.zeros arrays
        self.InitImuData()
        self.InitKinematicData()
        self.InitOutData()
        self.InitContactData()
        if self.estimator_logging : self.InitLogMatrixes()

        # check that sensors can be read
        if self.device is not None :
            self.ReadSensor()
            self.logger.LogTheLog("Sensors read, initial data acquired", to_print=talkative)
        

        # update height of CoM value, assuming Bolt is vertical
        pin.forwardKinematics(self.robot.model, self.robot.data, self.q)
        pin.updateFramePlacements(self.robot.model, self.robot.data)
        c = np.array((self.robot.data.oMf[self.feet_indexes[0]].inverse()*self.robot.data.oMf[self.base_id]).translation).copy()
        self.p_out[2] = np.linalg.norm(c)
        self.p_out[2] += 0.02 # TODO : radius of bolt foot
        self.p_out[2] = 0.34 # TODO stop doing this
        self.all_time_switch_deltas[2] = self.p_out[2]
        
        if self.estimator_logging : self.UpdateLogMatrixes()
        self.iter += 1
        self.logger.LogTheLog("Initial data stored in logs", to_print=talkative)
        
        # position drift compensation using switch and step measurement
        self.t0pos_drift_comp = t0pos_drift_comp
        self.pos_drift_compensation = 0
        self.step_start = 0
        self.step_duration = 0
        
        # filter parameters
        parameters_af = [self.time_step] + parameters_af
        parameters_sf = [self.time_step] + parameters_sf
        parameters_pf = [self.time_step] + parameters_pf
        # set desired filters types for attitude and speed
        # for the time being, complementary only
        if attitude_filter_type=="complementary":
            self.attitude_filter = ComplementaryFilter(parameters=parameters_af, 
                                                    name="attitude complementary filter", 
                                                    talkative=talkative, 
                                                    logger=self.logger, 
                                                    ndim=3)
        self.logger.LogTheLog("Attitude Filter of type '" + attitude_filter_type + "' added.", to_print=talkative)
        
        if speed_filter_type=="complementary":
            self.speed_filter = ComplementaryFilter(parameters=parameters_sf, 
                                                    name="speed complementary filter", 
                                                    talkative=talkative, 
                                                    logger=self.logger, 
                                                    ndim=3,
                                                    MemorySize=80,
                                                    OffsetGain=0.02)
        self.logger.LogTheLog("Speed Filter of type '" + speed_filter_type + "' added.", to_print=talkative)
        
        self.height_filter = ComplementaryFilter(parameters=parameters_pf, 
                                                name="base height complementary filter", 
                                                talkative=talkative, 
                                                logger=self.logger, 
                                                ndim=1)
        

        # returns info on Slips, Contact Forces, Contact with the ground
        self.contact_estimator = ContactEstimator(robot=self.robot, 
                                                 LeftFootFrameID=self.feet_indexes[0], 
                                                 RightFootFrameID=self.feet_indexes[1], 
                                                 LeftKneeFrameID=7, # self.robot.model.getFrameId("FL_KNEE"),
                                                 RightKneeFrameID=15, # self.robot.model.getFrameId("FR_KNEE"),
                                                 LeftKneeTorqueID=2,
                                                 RightKneeTorqueID=5,
                                                 iter_number=self.iter_number,
                                                 dt=self.time_step,
                                                 MemorySize=5,
                                                 Logging=self.contact_logging,
                                                 talkative=self.talkative,
                                                 logger=self.logger)
        self.logger.LogTheLog("Contact Estimator added.", to_print=talkative)
        
        # returns info on Slips, Contact Forces, Contact with the ground
        self.tiltand_speed_estimator = TiltEstimator(robot=self.robot,
                                                   Q0=self.q,
                                                   Qd0=self.qdot,
                                                   Niter=self.iter_number,
                                                   Logging=self.tilt_logging,
                                                   params=parameters_ti)
        
        self.logger.LogTheLog("Tilt Estimator added with parameters " + str(parameters_ti), to_print=talkative)
        
        
        # returns info on foot attitude
        self.foot_attitude_estimator = FootAttitudeEstimator(parameters=[self.time_step, 2],
                                                         dt=self.time_step,
                                                         name="Foot Attitude Estimator",
                                                         talkative=talkative,
                                                         logger=self.logger)
        self.logger.LogTheLog("Foot Attitude Estimator added with parameters " + str(0), to_print=talkative)
        

        self.logger.LogTheLog(self.msg_name + " initialized successfully.", to_print=talkative)
        return None
    

    
    def SetInitValues(self, base_speed, base_acc_g, unit_gravity, unit_gravity_derivative, contact_foot_id, Q, Qd):
        """ modify init values """
        self.q[:] = Q
        self.qdot[:]  = Qd
        _, rot = self.ComputeFramePose(contact_foot_id)
        base_wrt_foot_orientation_as_matrix = rot.T
        unit_gravity[2] *=-1
        unit_gravity_derivative[2] *=-1
        self.tiltand_speed_estimator.SetInitValues(base_speed, base_acc_g, unit_gravity, unit_gravity_derivative, base_wrt_foot_orientation_as_matrix)


    def ExternalDataCaster(self, data_type:str, received_data) -> np.ndarray:
        # In case data from elsewhere needs to be converted to another format, or truncated
        if data_type == "acceleration_imu":
            # acceleration in IMU frame -> in base frame
            # ANGULAR SPEED MUST BE ADAPTED FIRST
            #received_data = utils.RotByQuat(received_data, self.imu_base_quat)
            received_data  = self.imu_base_transform.rotation @ received_data
            omegadot = (self.w_imu - self.prev_w_imu) / self.time_step
            x = received_data + utils.cross(omegadot, self.imu_base_transform.translation) + utils.cross(self.w_imu, utils.cross(self.w_imu, self.imu_base_transform.translation))
        elif data_type == "angular_speed_imu":
            # angular speed in IMU frame -> in base frame
            x = self.imu_base_transform.rotation @ received_data
            #x = utils.RotByQuat(received_data, self.imu_base_quat)

        elif data_type == "attitude_imu":
            # orientation in IMU frame -> in base frame
            #q = pin.SE3ToXYZQUAT(self.imu_base_transform)[3:]
            q = self.imu_base_quat
            x = utils.RotateQuat(received_data, q)
        elif data_type == "speed_imu":
            # speed in IMU frame -> in base frame
            received_data = self.imu_base_transform.rotation @ received_data
            #received_data = utils.RotByQuat(received_data, self.imu_base_quat)
            x = received_data + utils.cross(self.w_imu, self.imu_base_transform.translation)
        #...
        else :
            self.logger.LogTheLog("unrecognized data caster : " + data_type, "warn", to_print=self.talkative)
            x = None
        return x
    
    def SetIMUToBaseTransformation(self, XYZ=[0, 0, 0], EulerXYZ=[np.pi, 0, 0]) -> None :
        """ Add an IMU frame to bolt model, linked to the base
            XYZ : IMU -> Base distance
            Euler : IMU -> Base rotation
            Used to turn IMU acceleration and rotation into base's acceleration and rotation
            By default, XYZ and Euler are those of an IMU attached to Bolt's hat """
        # check if frame has already been added
        if self.robot.model.getFrameId("IMU") != self.robot.model.nframes :
            self.logger.LogTheLog("IMU frame already added to model", "warn", to_print=self.talkative)
        # convert data
        XYZ = np.array(XYZ)
        Euler = np.array(EulerXYZ)
        ROT = R.from_euler("xyz", Euler).as_matrix()
        # create imu frame attached to base

        M = pin.SE3(ROT, XYZ)
        imu_frame = pin.Frame("IMU", self.base_joint_id, self.base_id, M, pin.FrameType.OP_FRAME)
        # add frame
        """
        self.robot.model.addFrame(imu_frame)
        self.robot.data = self.robot.model.createData()
        """
        self.imu_base_transform = M
        self.imu_base_quat = R.from_matrix(M.rotation).as_quat()
        self.logger.LogTheLog("added IMU frame", "subinfo", to_print=self.talkative)
        """
        # recreate contact estimator data to make up for the added frame
        self.contact_estimator.dataT = self.robot.model.createData()
        self.contact_estimator.data3D = self.robot.model.createData()
        self.contact_estimator.data1D = self.robot.model.createData()
        """
        return None


    def InitImuData(self) -> None :
        # initialize data to the right format
        self.a_imu = np.zeros((3,))   
        self.ag_imu = np.array([0, 0, -9.81])            
        self.w_imu = np.zeros((3,))
        self.theta_imu = np.array([0, 0, 0, 1])
        self.prev_w_imu = np.zeros((3,))
        # imu pre integrated speed
        self.v_imu = np.zeros((3,))


        self.reference_orientation = np.zeros((4,))
        return None

    def InitOutData(self) -> None:
        # initialize estimator out data
        self.a_out = np.zeros((3,)) 
        self.theta_out = np.array([0, 0, 0, 1])
        self.w_out = np.zeros((3,)) 
        self.g_out = np.array([0, 0, -1])

        self.p_out = np.zeros((3,))
        self.v_out = np.zeros((3,)) 

        self.contact_foot = "none"
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
        self.switch_delta = np.zeros((3,))
        self.all_time_switch_deltas = np.zeros((3,))
        self.previouscontact_foot = 0
        self.previous_left_contact, self.previous_right_contact = False, False
        self.switch = False
        self.switch_len = 0
        return None

    def InitContactData(self) -> None:
        self.left_contact = False
        self.right_contact = False
        self.fl_contact = np.zeros(3)
        self.fr_contact = np.zeros(3)
        return None

    def InitLogMatrixes(self) -> None :
        # initialize data to the right format
        # base velocitie & co, post-filtering logs
        self.log_v_out = np.zeros([3, self.iter_number])
        self.log_w_out = np.zeros([3, self.iter_number])
        self.log_a_out = np.zeros([3, self.iter_number])
        self.log_theta_out = np.zeros([4, self.iter_number])
        self.log_theta_out[3, :] = np.ones((self.iter_number, ))
        self.log_g_out = np.zeros([3, self.iter_number])
        self.log_p_out = np.zeros([3, self.iter_number])

        # imu data log
        self.log_v_imu = np.zeros([3, self.iter_number])
        self.log_w_imu = np.zeros([3, self.iter_number])
        self.log_a_imu = np.zeros([3, self.iter_number])
        self.log_theta_imu = np.zeros([4, self.iter_number])
        self.log_theta_imu[3, :] = np.ones((self.iter_number, ))
        # forward kinematics data log
        self.log_v_kin = np.zeros([3, self.iter_number])
        self.log_z_kin = np.zeros([1, self.iter_number])
        self.log_q = np.zeros([self.nq, self.iter_number])
        self.log_qdot = np.zeros([self.nv, self.iter_number])
        self.log_theta_kin = np.zeros([4, self.iter_number])
        self.log_theta_kin[3, :] = np.ones((self.iter_number, ))
        self.log_w_kin = np.zeros([3, self.iter_number])
        
        # tilt log 
        self.log_v_tilt = np.zeros([3, self.iter_number])
        self.log_g_tilt = np.zeros([3, self.iter_number])
        self.log_theta_tilt = np.zeros([4, self.iter_number])
        self.log_theta_tilt[3, :] = np.ones((self.iter_number, ))
        
        # other logs
        self.log_contactforces = np.zeros([6, self.iter_number])
        
        # Contact switch log
        self.log_p_switch = np.zeros([3, self.iter_number])
        
        # time
        self.time_stamp = np.zeros((self.iter_number, ))
        
        return None

    def UpdateLogMatrixes(self) -> None :
        log_iter = self.iter
        if self.iter >= self.iter_number:
            # Logs matrices' size will not be sufficient
            self.logger.LogTheLog("Excedind planned number of executions, iter_number = " + str(self.iter_number), style="warn", to_print=self.talkative)
            log_iter = self.iter_number-1

        # update logs with latest data
        # base velocitie & co, post-filtering logs
        self.log_v_out[:, log_iter] = self.v_out[:]
        self.log_w_out[:, log_iter] = self.w_out[:]#self.w_out.as_quat()[:]
        self.log_a_out[:, log_iter] = self.a_out[:]
        self.log_theta_out[:, log_iter] = self.theta_out[:]#.as_quat()[:]
        self.log_g_out[:, log_iter] = self.g_out[:]
        self.log_p_out[:, log_iter] = self.p_out[:]

        # imu data log
        self.log_v_imu[:, log_iter] = self.v_imu[:]
        self.log_w_imu[:, log_iter] = self.w_imu[:]#self.w_imu.as_quat()[:]
        self.log_a_imu[:, log_iter] = self.a_imu[:]
        self.log_theta_imu[:, log_iter] = self.theta_imu[:]#.as_quat()[:]
        # forward kinematics data log
        self.log_v_kin[:, log_iter] = self.v_kin[:]
        self.log_z_kin[:, log_iter] = self.z_kin[:]
        self.log_q[:, log_iter] = self.q[:]
        self.log_qdot[:, log_iter] = self.qdot[:]
        self.log_theta_kin[:, log_iter] = self.theta_kin[:] #.as_quat()[:]
        self.log_w_kin[:, log_iter] = self.w_kin[:]
        # tilt log 
        self.log_v_tilt[:, log_iter] = self.v_tilt[:]
        self.log_g_tilt[:, log_iter] = self.g_tilt[:]
        self.log_theta_tilt[:, log_iter] = self.theta_tilt[:]
        # other
        self.log_contactforces[:3, log_iter] = self.fl_contact[:]
        self.log_contactforces[3:, log_iter] = self.fr_contact[:]
        # switch
        self.log_p_switch[:, log_iter] = self.all_time_switch_deltas[:]
        # time
        self.time_stamp[log_iter] = self.time_running
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
            contact_forces = np.zeros(6)
            contact_forces[:3] = self.fl_contact
            contact_forces[3:] = self.fr_contact
            return contact_forces
        elif data=="q":
            return np.copy(self.q)
        elif data=="qdot":
            return np.copy(self.qdot)
        elif data=="tau":
            return np.copy(self.tau)
        elif data=="left_foot_pos":
            # left foot position in base frame
            pos, _ = self.ComputeFramePose(self.feet_indexes[0])
            return pos
        elif data=="right_foot_pos":
            pos, _ = self.ComputeFramePose(self.feet_indexes[1])
            return pos
        elif data=="timestamp" or data=="t":
            return self.time_running
        
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
            return self.log_p_switch
        elif data=="timestamp_logs" or data=="t_logs":
            return self.time_stamp
        
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
            self.logger.LogTheLog("Could not get data '" + data + "'. Unrecognised data getter.", style="danger", to_print=self.talkative)
            return None

    def SaveLogs(self, prefix=None, out=True, tilt=False, contact=True, imu=False, kin=False) -> None :
        # save logs
        if prefix is None :
            prefix = "./data/"
        else :
            prefix=prefix
        if out :
            # base velocitie & co, post-filtering logs
            np.save(prefix + "estimator_logs_v_out", self.log_v_out)
            np.save(prefix + "estimator_logs_w_out", self.log_w_out)
            np.save(prefix + "estimator_logs_a_out", self.log_a_out)
            np.save(prefix + "estimator_logs_theta_out", self.log_theta_out)
            np.save(prefix + "estimator_logs_g_out", self.log_g_out)
            np.save(prefix + "estimator_logs_p_out", self.log_p_out)
        
        if imu :
            # imu input data log
            np.save(prefix + "estimator_logs_v_imu", self.log_v_imu)
            np.save(prefix + "estimator_logs_w_imu", self.log_w_imu)
            np.save(prefix + "estimator_logs_a_imu", self.log_a_imu)
            np.save(prefix + "estimator_logs_theta_imu", self.log_theta_imu)
        if kin :
            # forward kinematics data log
            np.save(prefix + "estimator_logs_v_kin", self.log_v_kin)
            np.save(prefix + "estimator_logs_z_kin", self.log_z_kin)
            np.save(prefix + "estimator_logs_q", self.log_q)
            np.save(prefix + "estimator_logs_qdot", self.log_qdot)
            np.save(prefix + "estimator_logs_theta_kin", self.log_theta_kin)
            np.save(prefix + "estimator_logs_w_kin", self.log_w_kin)
        if tilt : 
            # tilt log 
            np.save(prefix + "estimator_logs_v_tilt", self.log_v_tilt)
            np.save(prefix + "estimator_logs_g_tilt", self.log_g_tilt)
            np.save(prefix + "estimator_logs_theta_tilt", self.log_theta_tilt)
        if contact : 
            # contact logs
            np.save(prefix + "estimator_logs_contact_forces", self.log_contactforces)
            np.save(prefix + "estimator_logs_contact_bool", self.contact_estimator.Get("contact_bool"))
            np.save(prefix + "estimator_logs_contact_prob", self.contact_estimator.Get("contact_prob"))
            # Contact switch log
            np.save(prefix + "estimator_logs_p_switch", self.log_p_switch)

        # time
        np.save(prefix + "estimator_logs_t", self.time_stamp)
        # logs
        np.save(prefix + "estimator_logs_logs", self.logger.GetLog())
        
        return None

    def ReadSensor(self) -> None:
        # rotation are updated supposing the value returned by device is xyz euler angles, in radians
        self.device.Read() # FOR TESTING ONLY #PPP
        # base acceleration, acceleration with gravity and rotation speed from IMU
        self.a_imu[:] = self.device.baseLinearAcceleration[:] # COPIED FROM SOLO CODE, CHECK CONSISTENCY WITH BOLT MASTERBOARD
        self.ag_imu[:] = self.device.baseLinearAccelerationGravity[:]
        
        self.w_imu[:] = self.device.baseAngularVelocity[:]
        # integrated data from IMU
        self.theta_imu[:] = self.device.baseAttitude[:]
        self.v_imu[:] = self.device.base_speed[:]
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
                            base_speed,
                            baseAttitude,
                            baseAngularVelocity,
                            q_mes,
                            v_mes,
                            tau_mes) -> None:
                            # acc with g is absolutely needed
        # rotation are updated supposing the value returned by device is xyz euler angles, in radians
        # base acceleration, acceleration with gravity and rotation speed from IMU
        self.a_imu[:] = baseLinearAcceleration[:]
        self.ag_imu[:] = baseLinearAccelerationGravity[:]
        self.theta_imu[:] = baseAttitude[:]
        self.w_imu[:] = baseAngularVelocity[:]
        self.v_imu[:] = base_speed[:]
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
        self.left_contact, self.right_contact = self.contact_estimator.LegsOnGround(self.q, 
                                                                                 self.qdot,
                                                                                 self.a_imu, 
                                                                                 self.tau,
                                                                                 self.g_tilt,
                                                                                 TorqueForceMingler=1.0, 
                                                                                 ProbThresold=0.45, 
                                                                                 TrustThresold=0.5
                                                                                 )
        # contact forces
        self.fl_contact, self.RLContact = self.contact_estimator.Get("current_cf_averaged")
        
        # contact foot
        if self.left_contact and self.right_contact :
            self.contact_foot = "both"
        elif self.left_contact :
            self.contact_foot = "left"
        elif self.right_contact :
            self.contact_foot = "right"
        else :
            self.contact_foot = "none"
            
    
        

        
        
    def FootStepLen(self, left_contact, right_contact, Maxswitch_len=20):
        """
        Measure length of a footstep
        Bolt switch contact foot when left_contact = right_contact
        ----------
        left_contact : Bool
        right_contact : Bool
        ----------
        Returns None
        """
        self.endingswitch = False
        
        # entering switch phase
        if (left_contact != self.previous_left_contact or right_contact != self.previous_right_contact) and not self.switch :
            #self.logger.LogTheLog(f"switch started on iter {self.iter}", "subinfo")
            # switch start
            self.switch = True
            self.switch_len = 0
            # foot in contact at the beginning of the switch
            if self.previous_left_contact :
                self.previouscontact_foot = 0
            else :
                self.previouscontact_foot = 1
                      
        # switching
        if self.switch :
            self.switch_len += 1
            # compute foot to foot distance
            base_to_back_foot, _ = self.ComputeFramePose(self.feet_indexes[self.previouscontact_foot])
            base_to_front_foot, _ = self.ComputeFramePose(self.feet_indexes[1-self.previouscontact_foot])
            # average distance over the current switch
            self.switch_delta = ( self.switch_delta*(self.switch_len - 1) + (base_to_front_foot - base_to_back_foot) ) / self.switch_len
            #self.switch_delta = base_to_back_foot - base_to_front_foot
            
        # switch finished
        if (left_contact!=self.previous_left_contact and right_contact!=self.previous_right_contact):
            # ending switch
            self.endingswitch = True
            
        # did not initialize correctly, or got lost somehow
        if self.switch and self.switch_len > Maxswitch_len and not (left_contact and right_contact):
            # ending switch
            self.endingswitch= True
            self.logger.LogTheLog(f"switch stopped on iter {self.iter} : exceding max switch duration {Maxswitch_len} iter ", "warn", to_print=self.talkative)
            
        if self.endingswitch :
            # ending switch
            self.switch = False
            self.step_duration = self.iter - self.step_start
            self.step_start = self.iter
            # updating all-time distance
            self.all_time_switch_deltas[:] += self.switch_delta[:]
            # updating contact info
            self.previous_left_contact = left_contact
            self.previous_right_contact = right_contact
            self.logger.LogTheLog(f"switch ended on iter {self.iter}, step of length {self.switch_delta} lasting {self.switch_len} iter.", "subinfo", to_print=self.talkative)

        


    
    # TODO : update with benallegue
    def KinematicAttitude(self) -> np.ndarray:
        # uses robot model and rotation speed to provide attitude estimate based on encoder data
        
        # consider the right contact frames, depending on which foot is in contact with the ground
        if self.left_contact and self.right_contact :
            self.logger.LogTheLog("Both feet are touching the ground", style="warn", to_print=self.talkative)
            contact_frames = [0, 1]
        elif self.left_contact :
            self.logger.LogTheLog("left foot touching the ground", to_print=self.talkative)
            contact_frames = [0]
        elif self.right_contact :
            self.logger.LogTheLog("right foot touching the ground", to_print=self.talkative)
            contact_frames = [1]
        else :
            self.logger.LogTheLog("No feet are touching the ground", style="warn", to_print=self.talkative)
            contact_frames = []

        # Compute the base's attitude for each foot in contact
        frame_attitude = []
        
        pin.forwardKinematics(self.robot.model, self.robot.data, self.q)
        pin.updateFramePlacements(self.robot.model, self.robot.data)
        pin.computeAllTerms(self.robot.model, self.robot.data, self.q, self.qdot)
        
        if self.left_contact :
            contact_foot_id = self.feet_indexes[0]
        else :
            contact_foot_id = self.feet_indexes[1]
        base_id = 1
        
        for foot in contact_frames:
            
            # attitude from foot to base
            foot_base_pose = self.robot.data.oMf[contact_foot_id].inverse()*self.robot.data.oMf[base_id]
            foot_base_attitude = np.array(foot_base_pose.rotation).copy()
            foot_base_position = np.array(foot_base_pose.translation).copy()
            
            # attitude of the foot
            world_foot_attitude = self.foot_attitude_estimator.RunFilter(IMUKinPos=foot_base_position, IMUKinRot=foot_base_attitude, ya=self.ag_imu, yg=self.w_imu)
            
            
            # combined attitude
            world_base_attitude = world_foot_attitude + foot_base_attitude
        
        self.theta_kin = R.from_euler(world_base_attitude)

        #return self.theta_kin.as_euler('xyz')
        return world_base_attitude
    
    def KinematicPosition(self):
        """
        Compute height of base compared to foot in contact
        -------
        p : position of base in world frame, robot-aligned
        """
        if self.left_contact and self.right_contact:
            pleft, _  = self.ComputeFramePose(self.feet_indexes[0])
            pright, _ = self.ComputeFramePose(self.feet_indexes[1])
            p = 0.5  *(pleft + pright)
        elif self.left_contact :
            p, _ = self.ComputeFramePose(self.feet_indexes[0])
        elif self.right_contact :
            p, _ = self.ComputeFramePose(self.feet_indexes[1])
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
        
        BaseFramePose = self.robot.data.oMf[self.base_id].inverse()*self.robot.data.oMf[nframe]
        Baseframe_attitude = np.array(BaseFramePose.rotation).copy()
        BaseFramePosition = np.array(BaseFramePose.translation).copy()
        return BaseFramePosition, Baseframe_attitude



    def IMUAttitude(self) -> np.ndarray :
        # IMU gives us acceleration and acceleration without gravity
        # measured gravity
        g0 = self.ag_imu - self.a_imu
        self.theta_g = self.TiltfromG(g0)
        return self.theta_g
    

    
    def KinematicSpeed(self) -> tuple((np.ndarray, np.ndarray)):
        # uses Kinematic data
        # along with contact and rotation speed information to approximate speed

        # consider the right contact frames, depending on which foot is in contact with the ground
        if self.left_contact and self.right_contact :
            self.logger.LogTheLog("Both feet are touching the ground on iter " + str(self.iter), style="warn", to_print=self.talkative)
            contact_frames = [0,1]
        elif self.left_contact :
            self.logger.LogTheLog("left foot touching the ground", to_print=False)
            contact_frames = [0]
        elif self.right_contact :
            self.logger.LogTheLog("right foot touching the ground", to_print=False)
            contact_frames = [1]
        else :
            self.logger.LogTheLog("No feet are touching the ground on iter " + str(self.iter), style="warn", to_print=self.talkative)
            contact_frames = []

        # Compute the base's speed for each foot in contact
        frame_speed = []
        frame_rot_speed = []

        pin.forwardKinematics(self.robot.model, self.robot.data, self.q)
        pin.updateFramePlacements(self.robot.model, self.robot.data)
        pin.computeAllTerms(self.robot.model, self.robot.data, self.q, self.qdot)

        for contact_foot_id in contact_frames:
            # speed of Base wrt its immobile foot
            oMf = self.robot.data.oMf[contact_foot_id]
            p_speed_l = oMf.inverse().action @ pin.getFrameVelocity(self.robot.model, self.robot.data, self.base_id, pin.WORLD)
            speed = np.array(p_speed_l[:3]).copy()
            # rotation speed of base frame in contact foot frame
            omega = np.array(p_speed_l[3:]).copy()

            frame_speed.append(speed)
            frame_rot_speed.append(omega)
        
        if self.left_contact and self.right_contact :
            # averages results
            self.v_kin = np.mean(np.array(frame_speed), axis=0)
            self.w_kin = np.mean(np.array(frame_rot_speed), axis=0)
        elif self.left_contact or self.right_contact :
            # one foot in contact
            self.v_kin = np.array(frame_speed)
            self.w_kin = np.array(frame_rot_speed)
        else :
            # no foot touching the ground, keeping old speed data
            if self.estimator_logging : 
                v_avg = np.mean(self.log_v_kin[:, max(0, self.iter-10):self.iter-1], axis=1)
                w_avg = np.mean(self.log_w_kin[:, max(0, self.iter-10):self.iter-1], axis=1)
            else :
                v_avg, w_avg = self.v_kin, self.w_kin
            self.w_kin = w_avg
            self.v_kin = v_avg
        
        # filter speed
        #self.v_kin = self.speed_filter.RunFilter(self.v_kin, self.a_imu)

        return self.v_kin, self.w_kin


    def SpeedFusion(self, mitigate=[0.1, 0.2, 0.7]) -> None:
        """
        
        """
        # uses Kinematic-derived speed estimate and IMU to estimate speed
        #self.KinematicSpeed()

        # runs speed and tilt estimator
        if self.left_contact :
            contact_foot_id = self.feet_indexes[0]
        else :
            contact_foot_id = self.feet_indexes[1]
        # run tilt estimator
        self.v_tilt, self.g_tilt = self.tiltand_speed_estimator.Estimate(Q=self.q.copy(),
                                            Qd=self.qdot.copy(),
                                            base_id=1,
                                            contact_foot_id=contact_foot_id,
                                            ya=self.ag_imu.copy(),
                                            yg=self.w_imu.copy(), 
                                            dt=self.time_step)
        
        # filter results
        self.v_out[:] = self.speed_filter.RunFilter(self.v_tilt.copy(), self.a_imu.copy())
        # filter speed with data from imu
        if np.linalg.norm(self.ag_imu - self.a_imu)<9:
            self.logger.LogTheLog(f"anormal gravity input : {self.ag_imu - self.a_imu} on iter {self.iter}", "warn", to_print=self.talkative)

        return None
    

    
    
    def TiltfromG(self, g0) -> np.ndarray :
        """
        From estimated g in base frame, get the quaternion between world frame and base frame
        """
        g = g0[:]
        if abs(np.linalg.norm(g0) - 1) > 0.1:
            self.logger.LogTheLog(f"gravity computed on iter {self.iter} is anormal : {g0}", "warn",  to_print=self.talkative)
        g = g/np.linalg.norm(g)
        gworld = np.array([0, 0, -1])
        
        # compute the quaternion to pass from gworld to g0
        gg0 = utils.cross(g, gworld)
        q0 = np.array( [np.linalg.norm(g) * np.linalg.norm(gworld) + utils.scalar(g, gworld)] )

        q = np.concatenate((gg0, q0), axis=0)
        if np.linalg.norm(q) == 0 and self.talkative :
            self.logger.LogTheLog(f"Null norm quaternion computed for gworld -> grobot at iter {self.iter} with gworld {gworld} and g measured {g0}", "danger", to_print=self.talkative)
            return np.array([0, 0, 0, 1])    
        return q / np.linalg.norm(q)


    
    def CheckQuat(self, q, name="") -> np.ndarray:
        """
        Check if a quat is not of norm 1

        """
        if np.linalg.norm(q) < 1e-6:
            self.logger.LogTheLog(f"Norm of quaternion {name} is NULL : {q} on iter {self.iter}", "danger", to_print=self.talkative)
            return np.array([0, 0, 0, 1])
        if abs(np.linalg.norm(q)-1) > 1e-3:
            self.logger.LogTheLog(f"Norm of quaternion {name} is NOT ONE : {q} on iter {self.iter}", "danger", to_print=self.talkative)
            return q/np.linalg.norm(q)
        return q
        
    
    
    def Estimate(self, time_step=None):
        """ this is the main function"""
        if time_step is not None :
            self.time_step = time_step
        
       # update all variables with latest available measurements
        if self.device is not None :
            self.ReadSensor()
        # convert data in IMU frame to base frame (SEQUENTIAL ORDER MATTERS)
        self.theta_imu = self.ExternalDataCaster("attitude_imu", self.theta_imu)
        self.w_imu = self.ExternalDataCaster("angular_speed_imu", self.w_imu)
        self.a_imu = self.ExternalDataCaster("acceleration_imu", self.a_imu)
        self.v_imu = self.ExternalDataCaster("speed_imu", self.v_imu)
        # check normalization of quat
        utils.normalizeQ(self.q)
        
        # run contact estimator
        self.UpdateContactInformation()
        if self.iter < 3 :
            self.previous_left_contact = self.left_contact
            self.previous_right_contact = self.right_contact
            self.switch = False
        # estimate speed
        self.SpeedFusion(mitigate=[0., 0., 1.])
        
        # integrate speed to get position
        self.p_out += self.v_out[:]*self.time_step

        # derive data & runs filter

        #self.theta_kin = self.KinematicAttitude()
        self.theta_tilt = self.TiltfromG(self.g_tilt)
        self.theta_out = self.attitude_filter.RunFilterQuaternion(self.theta_tilt.copy(), self.w_imu.copy())
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
        self.p_out[2] = self.height_filter.RunFilter(self.p_out[2], self.v_out[2])

        # correcting x position using kin
        self.FootStepLen(self.left_contact, self.right_contact, Maxswitch_len=15)
        if self.time_running > self.t0pos_drift_comp :
            if self.endingswitch and self.step_duration > 70 :
                # switch just ended, position have been updated & step has dured long enough
                # compute increment to add over next step to compensate drift
                pos_drift = self.all_time_switch_deltas[0] - self.p_out[0]
                sat = 0.05
                if pos_drift < -sat:
                    pos_drift = -sat
                elif pos_drift > sat:
                    pos_drift = sat
                self.pos_drift_compensation = pos_drift/self.step_duration
            self.p_out[0] += self.pos_drift_compensation
        

        # update all logs & past variables
        if self.estimator_logging : self.UpdateLogMatrixes()
        # count iteration
        if self.iter % 100 == 0 :
            self.logger.LogTheLog(f" iter {self.iter} \t and time running {self.time_running} \t dt {self.time_step}, giving freq {1/self.time_step}", "subinfo", to_print=self.talkative)
        if self.iter==1 :
            self.logger.LogTheLog("executed Estimator for the first time", "subinfo", to_print=self.talkative)
        self.prev_w_imu = self.w_imu.copy()
        self.iter += 1
        self.time_running += self.time_step
        

        return None






