
import pinocchio as pin
import numpy as np
import sys
from os.path import dirname, join, abspath
import example_robot_data
from pinocchio.visualize import MeshcatVisualizer

 
# Load the URDF model.

bolt = example_robot_data.load("bolt")
q1 = example_robot_data.readParamsFromSrdf(bolt.model, has_rotor_parameters=False, SRDF_PATH="/opt/openrobots/share/example-robot-data/robots/bolt_description/srdf/bolt.srdf", referencePose="standing")
v1 = pin.utils.zero(bolt.model.nv)



q1 = pin.neutral(bolt.model)
viz1 = MeshcatVisualizer(bolt.model, bolt.collision_model, bolt.visual_model)



# Start a new MeshCat server and client.
viz1.initViewer(open=True)

# Load the robot in the viewer.
viz1.loadViewerModel()
 
# Display a robot configuration.
q1 = pin.neutral(bolt.model)
viz1.display(q1)
viz1.displayVisuals(True)
 
# Create a convex shape from solo main body
mesh = bolt.visual_model.geometryObjects[0].geometry
mesh.buildConvexRepresentation(True)
convex = mesh.convex


viz1.display()

 

# final desired position
qf = np.array(
    [0., 0., 0., # xyz pos
     0.0, 0.0, 0.0, # scale, orientation
     0.0, 
     0.0, 0.0, -0.99, # articulations gauche
     -0.0, 0.0, -0.99] # articulations droite
)

# random speeds
v1 = np.random.randn(bolt.model.nv) * 2
v1 = np.zeros(bolt.model.nv)
data = pin.Data(bolt.model)
pin.forwardKinematics(bolt.model, bolt.data, q1, v1)
RF_id = bolt.model.getFrameId("FR_FOOT")
print("right foot frame id ", RF_id)
LF_id = bolt.model.getFrameId("FL_FOOT")
print("left foot frame id ", LF_id)


 
bolt.model.gravity.linear[:] = 0.5
dt = 0.1
 

def sim_loop(nsteps):
    tau1 = np.zeros(bolt.model.nv)
    qs = [q1]
    vs = [v1]

    for i in range(nsteps):
        q = qs[i]
        v = vs[i]
        a1 = pin.aba(bolt.model, data, q1, v1, tau1)
        vnext = v + dt * a1
        qnext = pin.integrate(bolt.model, q1, dt * vnext)
        
        qs.append(qnext)
        vs.append(vnext)
        
        

    return qs, vs
 





def supersim(nsteps):
    tau1 = np.zeros(bolt.model.nv)
    vs = [v1]
    qs = np.linspace( q1, qf, nsteps )
    RFAttitude = []
    LFAttitude = []

    for i in range(nsteps):
        q = qs[i]
        pin.framesForwardKinematics(bolt.model, data,q)
        
        pin.updateFramePlacements(bolt.model, data)
        
        RFAttitude.append( data.oMf[RF_id].translation )
        LFAttitude.append( data.oMf[LF_id].translation )
    return qs, RFAttitude, LFAttitude


def ultrasim(nsteps):
    tau1 = np.zeros(bolt.model.nv)
    vs = [v1]
    qs = np.linspace( q1, qf, nsteps )
    RFAttitude = []
    LFAttitude = []
    for i in range(nsteps):
        q = qs[i]
        pin.forwardKinematics(bolt.model, bolt.data, q)
        pin.updateFramePlacements(bolt.model, bolt.data)
        pin.centerOfMass(bolt.model, bolt.data, q)
        com = bolt.data.com[0]
        
        RFAttitude.append( bolt.data.oMf[RF_id].translation.copy() )
        LFAttitude.append( bolt.data.oMf[LF_id].translation.copy() )
        
        viz1.display(q)
    
        #print(bolt.data.oMf[RF_id].translation)
    return qs, RFAttitude, LFAttitude
        
        


qs, RfootAttitude, LfootAttitude = ultrasim(50)



