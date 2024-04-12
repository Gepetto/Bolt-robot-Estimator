
# This examples shows how to load and move a robot in meshcat.
# Note: this feature requires Meshcat to be installed, this can be done using
# pip install --user meshcat
 
import pinocchio as pin
import numpy as np
import sys
from os.path import dirname, join, abspath
 
from pinocchio.visualize import MeshcatVisualizer
 
# Load the URDF model.
# Conversion with str seems to be necessary when executing this file with ipython
pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "models")
 
model_path = join(pinocchio_model_dir, "example-robot-data/robots")
mesh_dir = pinocchio_model_dir
# urdf_filename = "talos_reduced.urdf"
urdf_filename = "bolt.urdf"
urdf_model_path = join(
    join(model_path, "bolt_description/robots"), urdf_filename)
 
model, collision_model, visual_model = pin.buildModelsFromUrdf(
    urdf_model_path, mesh_dir, pin.JointModelFreeFlyer()
)

viz1 = MeshcatVisualizer(model, collision_model, visual_model)



# Start a new MeshCat server and client.
# Note: the server can also be started separately using the "meshcat-server" command in a terminal:
# this enables the server to remain active after the current script ends.
#
# Option open=True pens the visualizer.
# Note: the visualizer can also be opened seperately by visiting the provided URL.
try:
    viz1.initViewer(open=True)
except ImportError as err:
    print(
        "Erroooooor while initializing the viewer. It seems you should install Python meshcat"
    )
    print(err)
    sys.exit(0)
 
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

 
# Display another robot.
viz2 = MeshcatVisualizer(model, collision_model, visual_model)
viz2.initViewer(viz1.viewer)
viz2.loadViewerModel(rootNodeName="pinocchio2")
q2 = q1.copy()
q2[1] = 1.0
#viz2.display(q2)
 
# standing config
q2 = np.array(
    [0.2, 0.4, 0.6, # xyz pos
     0.0, 0.0, 0.0, # scale, orientation
     0.0, 
     0.05, 0.8, -1.2, # articulations gauche
     -0.05, 0.8, -1.2] # articulations droite
)
viz2.display(q2)

qf = np.array(
    [0., 0., 0., # xyz pos
     0.0, 0.0, 0.0, # scale, orientation
     0.0, 
     0.05, 0.5, -0.8, # articulations gauche
     -0.05, 0.5, -0.8] # articulations droite
)

# random speeds
v1 = np.random.randn(model.nv) * 2
v1 = np.zeros(model.nv)
data1 = viz1.data
pin.forwardKinematics(model, data1, q1, v1)
frame_id = model.getFrameId("FR_FOOT")
print("right foot frame id ", frame_id)


viz1.display()
viz1.drawFrameVelocities(frame_id=frame_id)
 
model.gravity.linear[:] = 0.5
dt = 1.0
 

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
        viz1.drawFrameVelocities(frame_id=frame_id)
    return qs, vs
 


def supersim(nsteps):
    tau1 = np.zeros(model.nv)
    vs = [v1]
    qs = np.linspace( q1, qf, nsteps )

    for i in range(nsteps):
        q = qs[i]
        
        viz1.display(q)
        viz1.drawFrameVelocities(frame_id=frame_id)
    return qs
 

qs = supersim(3)
fid2 = model.getFrameId("FL_FOOT")
print("left foot frame id ", fid2)


def my_callback(i, *args):
    viz1.drawFrameVelocities(frame_id)
    viz1.drawFrameVelocities(fid2)
 
 
with viz1.create_video_ctx("../manualeap.mp4"):
    viz1.play(qs, dt, callback=my_callback)

