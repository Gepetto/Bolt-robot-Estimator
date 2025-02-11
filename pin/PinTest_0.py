import pinocchio as pin
from sys import argv
from os.path import dirname, join, abspath
 
# This path refers to Pinocchio source code but you can define your own directory here.
pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "models")

# You should change here to set up your own URDF file or just pass it as an argument of this example.
# urdf_filename = pinocchio_model_dir + '/example-robot-data/robots/ur_description/urdf/ur5_robot.urdf' if len(argv)<2 else argv[1]
urdf_filename =  "/home/nalbrecht/Bolt-Estimator/Bolt-robot---Estimator/pin/bolt_description/robots/bolt.urdf" 
 
# Load the urdf model
model    = pin.buildModelFromUrdf(urdf_filename)
print('model name: ' + model.name)
 
# Create data required by the algorithms
data     = model.createData()
 
# Sample a random configuration
q        = pin.randomConfiguration(model)
print('q: %s' % q.T)
 
# Perform the forward kinematics over the kinematic tree
pin.forwardKinematics(model,data,q)
 
# Print out the placement of each joint of the kinematic tree
for name, oMi in zip(model.names, data.oMi):
    print(("{:<24} : {: .2f} {: .2f} {: .2f}"
          .format( name, *oMi.translation.T.flat )))

'''
model, collision_model, visual_model = pin.buildModelsFromUrdf(
    urdf_model_path, mesh_dir, pin.JointModelFreeFlyer()
)

'''

