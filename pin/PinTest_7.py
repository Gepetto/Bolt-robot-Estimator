
import pinocchio as pin
import numpy as np
import sys
from os.path import dirname, join, abspath
import example_robot_data
from pinocchio.visualize import MeshcatVisualizer

import matplotlib.pyplot as plt
import time
from scipy.spatial.transform import Rotation as R



class boltomatic():
    def __init__(self):
        # Load the URDF model.
        self.bolt = example_robot_data.load("bolt")
        self.q = example_robot_data.readParamsFromSrdf(self.bolt.model, has_rotor_parameters=False, SRDF_PATH="/opt/openrobots/share/example-robot-data/robots/bolt_description/srdf/bolt.srdf", referencePose="standing")
        self.qd = pin.utils.zero(self.bolt.model.nv)
        self.qdd = np.zeros(self.bolt.model.nv)
        self.q = pin.neutral(self.bolt.model)
        self.tau = np.zeros(self.bolt.model.nv)
        
        
        self.RF_id = self.bolt.model.getFrameId("FR_FOOT")
        print("right foot frame id ", self.RF_id)
        self.LF_id = self.bolt.model.getFrameId("FL_FOOT")
        print("left foot frame id ", self.LF_id)
        self.C_id = [self.LF_id, self.RF_id]
        #self.C_id = [1]
        #for j in range(19):
        #    print(j)
        #    print('--  ', self.bolt.model.frames[j].parent)
        
        self.initView()
        self.initLog()
        
        self.bolt.model.gravity.linear[:] = 0.0
        self.t = 0.
        self.applyForce([np.zeros(3)])
        self.dt = 0.01
        self.qs = []
        
        #self.q[-6] = 1.
        
        

        
        
    def initView(self):
        self.viz = MeshcatVisualizer(self.bolt.model, self.bolt.collision_model, self.bolt.visual_model)
        # Start a new MeshCat server and client.
        self.viz.initViewer(open=True)    
        # Load the robot in the viewer.
        self.viz.loadViewerModel()
         
        # Display a robot configuration.
        self.viz.display(self.q)
        self.viz.displayVisuals(True)
         
        # Create a convex shape from solo main body
        mesh = self.bolt.visual_model.geometryObjects[0].geometry
        mesh.buildConvexRepresentation(True)
        convex = mesh.convex  
        self.viz.display()
    
    def initLog(self, n=500):
        self.log_q = np.zeros((self.bolt.model.nv+1, n))
        self.log_tau = np.zeros((n, self.bolt.model.nv, 3))
    

    def superupdateView(self, k, dt=0.1, Tdisp=50e-3):
        # display evry Tdisp
        if dt > Tdisp or not k % int(Tdisp/dt):
            self.viz.display(self.q)
            time.sleep(Tdisp)
    
    def updateView(self, k, dt=0.1, Tdisp=5):
        self.viz.display(self.q)
            
            
    def updateMove(self):
        pin.forwardKinematics(self.bolt.model, self.bolt.data, self.q)
        pin.updateFramePlacements(self.bolt.model, self.bolt.data)
        pin.centerOfMass(self.bolt.model, self.bolt.data, self.q)
        com = self.bolt.data.com[0]
    
    def Qcompare(self, newq_angular):
        print(self.lastq_angular - newq_angular)
        lastq_angular = newq_angular
        
        
    def updateLog(self, k):
        self.log_q[:, k] = self.q[:]
        #self.log_tau[k, :, :]  =self.tau[:]
        #print(self.q)
    
    def plotlog(self, fidlist):
        plt.figure(3, dpi=200)
        plt.grid()
        for fid in fidlist :
            plt.plot(self.log_q[fid, :])
        plt.show()
    def plottorque(self, jidlist):
        plt.figure(3, dpi=200)
        plt.grid()
        for jid in jidlist :
            plt.plot(self.log_tau[jid, :])
        plt.show()
    
    def kinMove(self, qf, n=10):
        qs = np.linspace( self.q, qf, n )
        self.initLog(n)
        v = 0
        preq = qs[0]
        for k in range(n):
            preq = self.q.copy()
            self.q = qs[k]
            self.updateMove()
            self.updateView(k)
            self.updateLog(k)
            
            #self.qs.append(self.q)
            prev = v
            v = (self.q - preq)/self.dt
            a = (v - prev)/self.dt
            
            pin.rnea(self.bolt.model, self.bolt.data, self.q, v, a, self.forces)
            self.tau_out = [self.bolt.data.f[j].angular for j in range(8)]
            print(self.tau_out)
        #self.plotlog([7,8,9,10])
    
    def torqueMove(self, Tf, dt, n=10000):
        Ts = np.linspace(self.tau, Tf, n)
        self.dt = dt
        self.initLog(n)
        for k in range(n):
            self.qdd = pin.aba(self.bolt.model, self.bolt.data, self.q, self.qd, self.tau, self.forces)
            
            # pour avoir l'accélération des frames
            pin.forwardKinematics(self.bolt.model, self.bolt.data, self.q, self.qd, np.zeros(self.bolt.model.nv))

            print("\n acc 1 et 3: \n")
            print(pin.getFrameAcceleration(self.bolt.model, self.bolt.data, 1, pin.ReferenceFrame.WORLD))
            print(pin.getFrameAcceleration(self.bolt.model, self.bolt.data, 3, pin.ReferenceFrame.LOCAL))
            self.qd += self.qdd*dt
            self.q = pin.integrate(self.bolt.model, self.q, self.qd*dt)
            
            
            self.updateMove()
            self.superupdateView(k, dt=dt)
            self.updateLog(k)
            self.t += dt
            self.tau = Ts[k]
            
            self.qs.append(self.q)
            
        self.plotlog([7,8,9,10])
    
    def forceMove(self, F, dt, n=10000):
        self.dt = dt
        self.initLog(n)
        self.applyForce(F)
        
        for k in range(n):
            preq = self.q.copy()
            self.qdd = pin.aba(self.bolt.model, self.bolt.data, self.q, self.qd, self.tau, self.forces)
            self.qd += self.qdd*dt
            #print(self.qd)
            self.q = pin.integrate(self.bolt.model, self.q, self.qd*dt)
            #print((self.q - preq)/dt)
            #print(self.qdd)
            
            self.updateMove()
            self.superupdateView(k, dt=dt)
            self.updateLog(k)
            self.t += dt
            self.qs.append(self.q)
            
            #print(self.bolt.model.joints[4])
            
        self.plotlog([7,8,9,10])
    
    def applyForce(self, ContactForces):
        # volé à victor
        ### Build force list for ABA
        forces = [ pin.Force.Zero() for _ in self.bolt.model.joints ]
        # I am supposing here that all contact frames are on separate joints. This is asserted below:
        #assert( len( set( [ cmodel.frames[idf].parentJoint for idf in contactIds ]) ) == len(contactIds) )
        
        for f,idf in zip(ContactForces,self.C_id):
            # Contact forces introduced in ABA as spatial forces at joint frame.
            forces[self.bolt.model.frames[idf].parent] = self.bolt.model.frames[idf].placement * pin.Force(f, 0.*f)
        self.forces = pin.StdVec_Force()
        for f in forces:
            self.forces.append(f)
    
    def video(self):
        with self.viz.create_video_ctx("../manualeap.mp4"):
            self.viz.play(self.qs, self.dt, callback=self.callback)
    
    def callback(self, i, *args):
        self.viz.drawFrameVelocities(self.RF_id)
    
    def jacob(self, fid):
        print('\n JACOBIAN \n')
        J = pin.computeFrameJacobian(self.bolt.model, self.bolt.data, self.q, fid)
        print(J)
        print(J.transpose())
    
    def PrintFrame(self,j):
        print(self.bolt.data.oMf[j].translation)

















bolt = example_robot_data.load("bolt")
q = example_robot_data.readParamsFromSrdf(bolt.model, has_rotor_parameters=False, SRDF_PATH="/opt/openrobots/share/example-robot-data/robots/bolt_description/srdf/bolt.srdf", referencePose="standing")
qd = pin.utils.zero(bolt.model.nv)
qdd = np.zeros(bolt.model.nv)
q = pin.neutral(bolt.model)
tau = np.zeros(bolt.model.nv)

x = bolt.model.getFrameId("FL_FOOOOT")
print(x)
print(bolt.model.nframes)

M1 = pin.SE3.Identity()
M2 = pin.SE3(R.from_euler("xyz", [0.5, 0.1, 0.1]).as_matrix(), np.array([0, 0, 33]))
print(M1)
print(M2)

v = np.array([1, 2, 3])
print(M2 * v)

XYZ = np.array([0, 0, 0])
Euler = np.array([0, 0, 0])
ROT = R.from_euler("xyz", Euler).as_matrix()
M3 = pin.SE3(ROT, XYZ)

print(M3)
print(pin.se3ToXYZQUAT(M3))

IMUFrame = pin.Frame("IMU", 1, 0, M3, pin.FrameType.OP_FRAME)
