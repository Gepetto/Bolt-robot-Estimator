
import numpy as np
from scipy.spatial.transform import Rotation as R
import time as t
from decimal import Decimal

from bolt_estimator.estimator.Bolt_Estimator_0 import Estimator
from bolt_estimator.utils.Graphics import Graphics
from bolt_estimator.utils.Bolt_Utils import Log, utils

from bolt_estimator.utils.DeviceEmulator import DeviceEmulator
from bolt_estimator.utils.TrajectoryGenerator import TrajectoryGenerator
from bolt_estimator.estimator.Bolt_Filter_Complementary import ComplementaryFilter
from bolt_estimator.data.DataReader import DataReader



def TiltfromG(g0) -> np.ndarray:
    """
    From estimated g in base frame, get the euler angles between world frame and base frame
    """
    g = g0/np.linalg.norm(g0)
    gworld = np.array([0, 0, -1])

    v = np.cross(gworld, g)
    s = np.linalg.norm(v)
    c = utils.scalar(gworld, g0)
    RotM = np.eye(3) + utils.S(v) + utils.S(v**2)*(1/(1+c))

    euler = R.from_matrix(RotM).as_euler("zyx")

    return euler

def main():
    N  = 3000 - 1
    dt = 1e-3
    T  = N*dt
    kf = 3
    #ToPlot = "inputs, position, vitesse, accélération, attitude, omega, tau, q, qdot, contact, trust, forces, g "
    ToPlot = "position, forces contact"

    # create objects
    testlogger = Log("test", PrintOnFlight=True)
    grapher = Graphics(logger=testlogger)
    generator = TrajectoryGenerator(logger=testlogger)
    device = DeviceEmulator(TrajectoryGenerator=generator, logger=testlogger)
    reader = DataReader(logger=testlogger)

    # load traj generated by simulation 
    device.LoadSimulatedData(Reader=reader, kfile=kf)
    # init estimator
    logging = True
    estimator = Estimator(device=device,
                    ModelPath="",
                    UrdfPath="",
                    Talkative=False,
                    logger=testlogger,
                    AttitudeFilterType = "complementary",
                    parametersAF = [2],         # réglé
                    SpeedFilterType = "complementary",
                    parametersSF = [1.1],       # réglé
                    parametersPF = [0.15],      # réglé
                    parametersTI = [10, 60, 2], # réglé
                    TimeStep = dt,
                    IterNumber = N,
                    T0posDriftComp = 2.0,
                    EstimatorLogging=logging,
                    ContactLogging=logging,
                    TiltLogging=logging)
    estimator.SetIMUToBaseTransformation(XYZ=[0, 0, 0], EulerXYZ=[np.pi, 0, 0])
    #x = estimator.ExternalDataCaster("acceleration_imu", np.array([1, 2, 3]))


    
    # plot the pseudo-measured (ie noisy and drifting) generated trajectories
    Input_Acc = device.Acc
    Input_AccG = device.AccG
    Input_Speed = device.Speed
    Input_Omega = device.Omega
    Input_Theta = device.Theta
    Input_Q = device.Q
    Input_Qd = device.Qd
    Input_Tau = device.Tau

    # get some data from simulation to compare to
    True_LCF = reader.Get("lcf")
    True_RCF = reader.Get("rcf")

    True_v = reader.Get("v")
    True_a = reader.Get("a")[:, 1, :]
    True_g = device.AccG_true - device.Acc_true
    True_Q = reader.Get("q")
    True_Qd = reader.Get("qd")
    True_Tau = reader.Get("tau")
    True_w = reader.Get("omega")[:, 1, :]
    True_Theta = reader.Get("theta_euler")[:, 1, :]
    True_g_bis = np.zeros((N, 3))
    True_x = reader.Get("x")[:, 1, :]

    Computed_Theta = np.zeros((3, N))
    print(True_w.shape)


    
    # run the estimator as if
    # device will iterate over generated data each time estimator calls it
    start=0
    device.iter = start
    t0 = t.time()
    for j in range(start, N-1):
        # device.Read()
        # a = device.baseLinearAcceleration.copy()
        # ag = device.baseLinearAccelerationGravity.copy()
        # omega = device.baseAngularVelocity.copy()
        # q = device.q_mes.copy()
        # v = device.v_mes.copy()
        # tau = device.tau_mes.copy()
        # estimator.ReadExternalSensor(baseLinearAcceleration=a,
        #                                     baseLinearAccelerationGravity=ag,
        #                                     baseAngularVelocity=omega,  
        #                                     q_mes=q,
        #                                     v_mes=v, 
        #                                     tau_mes=tau)
        if  estimator.iter == 1 :
            v = True_v[start, 1, :]
            ag = device.AccG_true[start, :]
            ug = utils.normalize(True_g[start, :])
            udg = (True_g[start, :] -  True_g[start-1, :])/dt / np.linalg.norm(True_g[start, :])


            estimator.SetInitValues(BaseSpeed=v,
                                    BaseAccG=ag, 
                                    UnitGravity=ug, 
                                    UnitGravityDerivative=udg, 
                                    ContactFootID=10,
                                    Q = True_Q[start, :],
                                    Qd = True_Qd[start, :])
        estimator.Estimate()

    t1 = t.time()
    print(" ")
    print(f"Estimator : \n\t was executed {N-1 - start} times\n\t within {round(t1-t0, 3)}s\n\t @ f={Decimal(str((N-1-start)/(t1-t0))):.3E}")
    print(" ")
    
    # get logs from estimator
        # rotations
    theta_estimator = estimator.Get("theta_logs_euler")
    theta_imu = estimator.Get("theta_logs_imu")
        # tilt data
    g_tilt = estimator.Get("g_tilt_logs")
    v_tilt = estimator.Get("v_tilt_logs")
    theta_tilt = estimator.Get("theta_tilt_logs_euler")
    v_kin = estimator.Get("v_logs_kin")
        # out data
    v_out = estimator.Get("v_out_logs")
    p_out = estimator.Get("p_logs")
    theta_out = estimator.Get("theta_logs_euler")
    g_out = estimator.Get("g_out_logs")
    p_switch = estimator.Get("p_switch_logs")
        # quat
    quat_out = estimator.Get("theta_logs")
    quat_tilt = estimator.Get("theta_tilt_logs")

    g_rebuilt = np.zeros((3, N))
    for j in range(N-1):
        # rebuild gravity estimate (comprend rien)
        rotmat = R.from_euler("xyz", theta_out[:, j]).as_matrix()
        g_rebuilt[:, j] = rotmat @ np.array([0, 0, -9.81])
        g_rebuilt[2, j] = 5




    # get logs from contact estimator
        # contact forces
    LCF_1D, RCF_1D = estimator.ContactEstimator.Get("cf_1d")
    LCF_3D, RCF_3D = estimator.ContactEstimator.Get("cf_3d")
    LCF_T, RCF_T = estimator.ContactEstimator.Get("cf_torques")
        # trust etc
    Trust = estimator.ContactEstimator.Get("trust")
    Slip = estimator.ContactEstimator.Get("slip_prob")
    Contact = estimator.ContactEstimator.Get("contact_bool")
    ContactProb = estimator.ContactEstimator.Get("contact_prob")
    ContactProb_F = estimator.ContactEstimator.Get("contact_prob_force")
    ContactProb_T = estimator.ContactEstimator.Get("contact_prob_torque")


    # logs from tilt estimator
    omega_tilt = estimator.TiltandSpeedEstimator.x2_hat_dot_logs.copy()

    # plot input data
    if "q" in ToPlot :
        grapher.SetLegend(["Q in", "Q true "], 2)
        grapher.CompareNDdatas([Input_Q[:, 6:8].T, True_Q[:, 6:8].T], "theta", "Inputed and true Q, left leg", StyleAdapter=True)
    if "qdot" in ToPlot :
        grapher.SetLegend(["Qd in ", "Qd true"], 2)
        grapher.CompareNDdatas([Input_Qd[:, 5:7].T, True_Qd[:, 5:7].T], "omega", "Inputed and true Qd, left leg", StyleAdapter=True)
    if "tau" in ToPlot :
        grapher.SetLegend(["Tau in", "Tau true"], 3)
        grapher.CompareNDdatas([Input_Tau[:, :3].T, True_Tau[:, :3].T], "Nm", "Inputed and true Tau, left leg", StyleAdapter=True)
    if "input" in ToPlot :
        grapher.SetLegend(["a in", "a true"], 3)
        grapher.CompareNDdatas([Input_Acc.T, True_a.T], "m/s2", "Inputed acceleration", StyleAdapter=True)



        # plot rotation
        theta, omega = device.GetRotation()
        Inputs_rotations = [theta.T, omega.T]
        InOut_rotations = [theta.T[:2, :], theta_imu[:2, :], theta_estimator[:2, :]]
        
        grapher.SetLegend(["theta in", "theta imu","theta out"], 2)
        grapher.CompareNDdatas(InOut_rotations, "theta", "Noisy rot and out rot", StyleAdapter=True)
    
        grapher.SetLegend(["theta", "omega"], 3)
        grapher.CompareNDdatas(Inputs_rotations, "", "rotation and rotation speed as inputed", StyleAdapter=False, mitigate=[1])
    
        grapher.SetLegend(["g from tilt estimator", "theta"], 3)
        grapher.CompareNDdatas([-9.81*g_tilt, theta.T], "", "g and theta", StyleAdapter=False, mitigate=[1])


    
        # plot base acceleration with and without gravity
        ag = device.AccG_true
        a = device.Acc_true
        grapher.SetLegend(["ag", "a"], 3)
        grapher.CompareNDdatas([ag.T, a.T], "acceleration", "g", StyleAdapter=True, width=1.)
        
    
    # plot true base trajectory
    #reader.PlotBaseTrajectory()
    
    # plot true base torques
    #reader.PlotLeftFootCorrelation()
    
    # plot contact forces
    if "forces" in ToPlot :
        grapher.SetLegend(["Left force from 1D",], ndim=3)
        grapher.CompareNDdatas([LCF_1D], datatype='force', title='1D left contact force')
        
        grapher.SetLegend(["Left force from 3D",], ndim=3)
        grapher.CompareNDdatas([LCF_3D], datatype='force', title='3D left contact force')
        
        grapher.SetLegend(["Left force from torques",], ndim=3)
        grapher.CompareNDdatas([LCF_T], datatype='force', title='torque left contact force')
        
        grapher.SetLegend(["Right force from 1D",], ndim=3)
        grapher.CompareNDdatas([RCF_1D], datatype='force', title='1D right contact force')
        
        grapher.SetLegend(["Right force from 3D",], ndim=3)
        grapher.CompareNDdatas([RCF_3D], datatype='force', title='3D right contact force')
        
        grapher.SetLegend(["Right force from torques",], ndim=3)
        grapher.CompareNDdatas([RCF_T], datatype='force', title='torque right contact force')
    
        grapher.SetLegend(["Left force from 3D", "Left force from torques", "Right force from 3D", "Right force from torques",], ndim=1)
        grapher.CompareNDdatas([LCF_3D[2:, :], LCF_T[2:, :], RCF_3D[2:, :], RCF_T[2:, :]], datatype='force', title='comparing Z contact force')
        
        grapher.SetLegend(["Left Z force from 3D", "Left force from torques", "True left Force"], ndim=1)
        grapher.CompareNDdatas([LCF_3D[2:, :],  LCF_T[2:, :], [True_LCF[start:N, 2]]], datatype='force', title='comparing left Z contact force')
        
        grapher.SetLegend(["Right Z force from 3D", "Right force from torques", "True right Force"], ndim=1)
        grapher.CompareNDdatas([RCF_3D[2:, :],  RCF_T[2:, :], [True_RCF[start:N, 2]]], datatype='force', title='comparing right Z contact force')
    
    
    if "vitesse" in ToPlot :
        """
        grapher.SetLegend(["v from tilt estimator", "True v"], ndim=3)
        grapher.CompareNDdatas([v_tilt, True_v[start:N, 1, :].T], datatype='speed', title='Tilt estimator output', mitigate=[1])
        
        
        grapher.SetLegend(["v from tilt estimator", "Kin v", "True v"], ndim=3)
        grapher.CompareNDdatas([v_tilt, v_kin, True_v[start:N, 1, :].T], datatype='speed', title='speed estimation comparison', mitigate=[2])
        """
        
        grapher.SetLegend(["vx out", "True vx", "Error"], ndim=1)
        grapher.CompareNDdatas([v_out[:1],  True_v[:N, 1, :1].T, np.abs(True_v[:N, 1, :1].T-v_out[:1])], datatype='speed', title='X speed estimation comparison', mitigate=[1])
        grapher.SetLegend(["vy out", "True vy", "Error"], ndim=1)
        grapher.CompareNDdatas([v_out[1:2], True_v[:N, 1, 1:2].T, np.abs(True_v[:N, 1, 1:2].T-v_out[1:2])], datatype='speed', title='Y speed estimation comparison', mitigate=[1])
        
        
        grapher.SetLegend(["v out", "True v"], ndim=3)
        grapher.CompareNDdatas([v_out, True_v[start:N, 1, :].T], datatype='speed', title='Estimator output, speed', mitigate=[1])
        
        grapher.SetLegend(["v out", "v tilt", "True v"], ndim=3)
        grapher.CompareNDdatas([v_out, v_tilt, True_v[start:N, 1, :].T], datatype='speed', title='Etimator output, speed', mitigate=[2])
    if "position" in ToPlot :
        grapher.SetLegend(["base pos out", "True base pos"], ndim=3)
        grapher.CompareNDdatas([p_out,  True_x[start:N, :].T], datatype='position', title='Estimator output, position', mitigate=[1])

        grapher.SetLegend(["base pos switch", "True base pos"], ndim=3)
        grapher.CompareNDdatas([p_switch,  True_x[start:N, :].T], datatype='position', title='Estimator switch out, position', mitigate=[1])


    if "attitude" in ToPlot :
        grapher.SetLegend(["theta tilt", "theta_true"], ndim=3)
        grapher.CompareNDdatas([theta_tilt, True_Theta[start:N, :].T], datatype='orientation', title='Tilt estimator output, attitude', mitigate=[1])
        
        grapher.SetLegend(["theta tilt", "theta_out", "theta_true"], ndim=3)
        grapher.CompareNDdatas([theta_tilt, theta_out, True_Theta[start:N, :].T], datatype='orientation', title='Tilt estimator output, attitude', mitigate=[2])
        
        grapher.SetLegend(["quat tilt", "quat out"], ndim=4)
        grapher.CompareNDdatas([quat_tilt, quat_out], datatype='quaternion', title='Estimator quaternions', mitigate=[1])
        
        """
        grapher.SetLegend(["theta_true", "theta_computed"], ndim=3)
        grapher.CompareNDdatas([True_Theta[start:N, :].T, Computed_Theta], datatype='orientation', title='Computed output, attitude', mitigate=[1])
        """
        grapher.SetLegend(["theta out", "theta_true"], ndim=3)
        grapher.CompareNDdatas([theta_out, True_Theta[start:N, :].T,], datatype='orientation', title='Estimator output, attitude', mitigate=[1])
    
    if "g" in ToPlot :
        grapher.SetLegend(["g from tilt estimator", "True g",], ndim=3)
        grapher.CompareNDdatas([9.81*g_tilt, True_g[start:N, :].T], datatype='vertical', title='Tilt estimator output', mitigate=[1])
        
        grapher.SetLegend(["g out", "True g",], ndim=3)
        grapher.CompareNDdatas([9.81*g_out, True_g[start:N, :].T], datatype='vertical', title='Estimator output', mitigate=[1])
        """
        grapher.SetLegend(["g from tilt estimator", "Rebuilt g",], ndim=3)
        grapher.CompareNDdatas([-9.81*g_tilt, g_rebuilt], datatype='vertical', title='Tilt estimator output and rebuilt', mitigate=[1])
        """
    if "omega" in ToPlot :
        grapher.SetLegend(["omega tilt", "omega true"], ndim=3)
        grapher.CompareNDdatas([omega_tilt.T, True_w[start:N, :].T], datatype='rad per s', title='Tilt estimator output, angular speed', mitigate=[1])

    if "forces" in ToPlot :
        L3DNorm = np.linalg.norm(LCF_3D, axis=0)
        R3DNorm = np.linalg.norm(RCF_3D, axis=0)
    
        grapher.SetLegend(["Left force from 3D", "Left force from torques", "Right force from 3D", "Right force from torques",], ndim=1)
        grapher.CompareNDdatas([[L3DNorm], LCF_T[2:, :], [R3DNorm], RCF_T[2:, :]], datatype='force', title='comparing contact force norms')

        
    if "trust" in ToPlot :
        # plot trust, slips and contact
        grapher.SetLegend(["Slip","Trust"], ndim=2)
        grapher.CompareNDdatas([Slip, Trust], datatype='real value', title='probability of slip and trust value')
        
    if "contact" in ToPlot :
        grapher.SetLegend(["Contact boolean"], ndim=2)
        grapher.CompareNDdatas([[Contact[0, :]], [Contact[1, :]]], datatype='proba', title='contact', mitigate=[1])
        
        grapher.SetLegend(["Contact probability", "Contact probability Force", "Contact probability Torque", ], ndim=1)
        grapher.CompareNDdatas([ContactProb[:1, :], ContactProb_F[:1, :], ContactProb_T[:1, :]], datatype='proba', title='probability of left contact', mitigate=[1])
        
        
        grapher.SetLegend(["Contact probability force 3d", "Z Force"], ndim=1)
        grapher.CompareNDdatas([5*ContactProb_F[:1, :], LCF_3D[2:, :]], datatype='proba', title='probability of left contact')
    
    
    grapher.end()
    return None



















def main_fakedata():
    N = 5000
    T = 10

    # init updates
    testlogger = Log("test", PrintOnFlight=True)
    grapher = Graphics(logger=testlogger)
    generator = TrajectoryGenerator(logger=testlogger)
    device = DeviceEmulator(TrajectoryGenerator=generator, logger=testlogger)

    # generate traj
    device.GenerateTrajectory(N=N, NoiseLevelXY=10, NoiseLevelZ=10, Drift=30, NoiseLevelAttitude=10, T=T)
    # init estimator
    estimator = Estimator(device=device,
                    ModelPath="",
                    UrdfPath="",
                    Talkative=True,
                    logger=testlogger,
                    AttitudeFilterType = "complementary",
                    parametersAF = [2],
                    SpeedFilterType = "complementary",
                    parametersSF = [2],
                    TimeStep = T/N,
                    IterNumber = N)
    
    # plot the pseudo-measured (eg noisy and drifting) generated trajectories
    traj, speed, acc = device.GetTranslation()
    Inputs_translations = [traj.T, speed.T, acc.T]
    theta, omega, omegadot = device.GetRotation()
    Inputs_rotations = [theta.T, omega.T]

    grapher.SetLegend(["traj", "speed", "acc"], 3)
    grapher.CompareNDdatas(Inputs_translations, "", "Trajectory, speed and acceleration as inputed", StyleAdapter=False, width=1.)

    grapher.SetLegend(["theta", "omega"], 3)
    grapher.CompareNDdatas(Inputs_rotations, "", "rotation and rotation speed as inputed", StyleAdapter=False, width=1.)

    
    
    # run the estimator as if
    # device will iterate over generated data each time estimator calls it
    for j in range(N-1):
        estimator.Estimate()
    
    # get data as quat
    Qtheta_estimator = estimator.Get("theta_logs")
    Qtheta_imu = estimator.Get("theta_logs_imu")
    theta_estimator = np.zeros((3, N))
    theta_imu = np.zeros((3, N))
    for j in range(N-1):
        theta_estimator[:, j] = R.from_quat(Qtheta_estimator[:, j]).as_euler('xyz')
        theta_imu[:, j] = R.from_quat(Qtheta_imu[:, j]).as_euler('xyz')
    
    InOut_rotations = [theta.T[:2, :], theta_imu[:2, :], theta_estimator[:2, :]]
    grapher.SetLegend(["theta in", "theta imu","theta out"], 2)
    grapher.CompareNDdatas(InOut_rotations, "theta", "Noisy rot and out rot", StyleAdapter=True, width=1.)
    
    ag = device.AccG
    a = device.Acc
    grapher.SetLegend(["ag", "a"], 3)
    grapher.CompareNDdatas([ag.T, a.T], "acceleration", "g", StyleAdapter=True, width=1.)

    '''

    filter = ComplementaryFilter(ndim=3, parameters=[T/N, 2])
    theta_filter = np.zeros((3, N))

    # run filter over time, with noisy data as inputs
    for k in range(N):
        theta_filter[:, k] = filter.RunFilter(theta.T[:,k], omega.T[:,k])
    
    InOut_rotations = [theta.T[:2, :], theta_filter[:2, :]]
    grapher.SetLegend(["theta in","theta filter"], 2)
    grapher.CompareNDdatas(InOut_rotations, "theta", "theta", StyleAdapter=True, AutoLeg=False, width=1.)
    
    '''

    grapher.end()
    return None









if __name__ == "__main__":
    main()













