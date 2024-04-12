
import pinocchio as pin
import numpy as np
import sys
from os.path import dirname, join, abspath
 
from pinocchio.visualize import MeshcatVisualizer
 
# Load the URDF model.
pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "models")
 
model_path = join(pinocchio_model_dir, "example-robot-data/robots")
mesh_dir = pinocchio_model_dir
urdf_filename = "bolt.urdf"
urdf_model_path = join(
    join(model_path, "bolt_description/robots"), urdf_filename)
 
model, collision_model, visual_model = pin.buildModelsFromUrdf(
    urdf_model_path, mesh_dir, pin.JointModelFreeFlyer()
)

viz1 = MeshcatVisualizer(model, collision_model, visual_model)



# Start a new MeshCat server and client.
viz1.initViewer(open=True)

# Load the robot in the viewer.
viz1.loadViewerModel()
 
# Display a robot configuration.
q1 = pin.neutral(model)
viz1.display(q1)
viz1.displayVisuals(True)
 
# Create a convex shape from solo main body
mesh = visual_model.geometryObjects[0].geometry
mesh.buildConvexRepresentation(True)
convex = mesh.convex

 

# final desired position
qf = np.array(
    [0., 0., 0., # xyz pos
     0.0, 0.0, 0.0, # scale, orientation
     0.0, 
     0.0, 0.0, -0.99, # articulations gauche
     -0.0, 0.0, -0.99] # articulations droite
)

# random speeds
v1 = np.random.randn(model.nv) * 2
v1 = np.zeros(model.nv)
data1 = viz1.data
pin.forwardKinematics(model, data1, q1, v1)
RF_id = model.getFrameId("FR_FOOT")
print("right foot frame id ", RF_id)
LF_id = model.getFrameId("FL_FOOT")
print("left foot frame id ", LF_id)

viz1.display()
viz1.drawFrameVelocities(frame_id=RF_id)
 
model.gravity.linear[:] = 0.5
dt = 0.1
 

def sim_loop(nsteps):
    tau1 = np.zeros(model.nv)
    qs = [q1]
    vs = [v1]

    for i in range(nsteps):
        q = qs[i]
        v = vs[i]
        a1 = pin.aba(model, data1, q1, v1, tau1)
        vnext = v + dt * a1
        qnext = pin.integrate(model, q1, dt * vnext)
        
        qs.append(qnext)
        vs.append(vnext)
        
        viz1.display(qnext)
        viz1.drawFrameVelocities(frame_id=RF_id)
    return qs, vs
 





def supersim(nsteps):
    tau1 = np.zeros(model.nv)
    vs = [v1]
    qs = np.linspace( q1, qf, nsteps )
    RFAttitude = []
    LFAttitude = []

    for i in range(nsteps):
        q = qs[i]
        
        viz1.display(q)
        viz1.drawFrameVelocities(frame_id=RF_id)
        
        pin.forwardKinematics(model, data1, q)
        pin.updateFramePlacement(model, data1, RF_id)
        
        RFAttitude.append( data1.oMf[RF_id].translation )
        LFAttitude.append( data1.oMf[LF_id].translation )
    return qs, RFAttitude, LFAttitude





qs, RfootAttitude, LfootAttitude = supersim(250)



def my_callback(i, *args):
    viz1.drawFrameVelocities(RF_id)
    viz1.drawFrameVelocities(LF_id)
 
 
with viz1.create_video_ctx("../manualeap.mp4"):
    viz1.play(qs, dt, callback=my_callback)

