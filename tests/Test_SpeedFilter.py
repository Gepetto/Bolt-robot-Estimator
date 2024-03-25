import sys
sys.path.append('/home/nalbrecht/Bolt-Estimator/Bolt-robot---Estimator/src/python')
from Bolt_Utils import utils
from Bolt_Utils import Log
from TrajectoryGenerator import TrajectoryGenerator, Graphics, Metal



def main():
    testlogger = Log("test", PrintOnFlight=True)
    grapher = Graphics(logger=testlogger)

    generator = TrajectoryGenerator(logger=testlogger)
    generator.Generate("sinus", NoiseLevel=15)
    TrueTraj, TrueSpeed, TrueAcc = generator.GetTrueTraj()
    NoisyTraj, NoisySpeed, NoisyAcc = generator.GetNoisyTraj()


    dataset = [NoisyTraj, TrueTraj]
    grapher.CompareNDdatas(dataset, "position", "essai", StyleAdapter=False)




main()