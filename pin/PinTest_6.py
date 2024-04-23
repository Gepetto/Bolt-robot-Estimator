
import pinocchio as pin
import numpy as np
import sys
from os.path import dirname, join, abspath
import example_robot_data
from pinocchio.visualize import MeshcatVisualizer

import matplotlib.pyplot as plt
import time




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
            self.qd += self.qdd*dt
            self.q = pin.integrate(self.bolt.model, self.q, self.qd*dt)
            #print(self.qdd)
            
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
            print(self.qd)
            self.q = pin.integrate(self.bolt.model, self.q, self.qd*dt)
            print((self.q - preq)/dt)
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
        print(self.bolt.data.oMf[j].rotation)


    
        

 
def datagenerator(n, j=7, sp=-1):
    x = np.zeros(n)
    if sp==-1:
        x[j:] = np.random.random(n-j)*2 - 1
    else:
        x[sp] = np.random.random(1)[0]*2 - 1
    return x

tau = datagenerator(12, sp=7)/1000
qf = datagenerator(13, 8)
#tau = datagenerator(12, sp=11)/100

bolt = boltomatic()
#bolt.torqueMove(tau, dt=0.1, n=80)
#bolt.kinMove(qf)
#bolt.plottorque([2, 3, 4]) # left leg torques
bolt.forceMove([np.array([0., 0., 0.01]), np.array([0., 0., 0.]),], dt=0.01, n=100)
#bolt.jacob(10)
#bolt.video()


























# final desired position
qf = np.array(
    [0., 0., 0., # xyz pos
     0.0, 0.0, 0.0, # scale, orientation
     0.0, 
     0.0, 0.0, -0.99, # articulations gauche
     -0.0, 0.0, -0.99] # articulations droite
    )

