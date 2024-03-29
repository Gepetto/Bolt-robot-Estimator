import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline


"""
Increase the number of points in Constant's curves

N is the final number of points

"""


def main(N=300):

    # get data and extract data
    pcom = np.load("/home/nalbrecht/Bolt-Estimator/Bolt-robot---Estimator/tests/com_pos.npy")
    vcom = np.load("/home/nalbrecht/Bolt-Estimator/Bolt-robot---Estimator/tests/vcom_pos.npy")
    trajX = pcom[0, :, 0]
    trajY = pcom[0, :, 1]
    trajZ = pcom[0, :, 2]
    AdaptedTraj = np.array([trajX, trajY, trajZ])
    speedX = vcom[0, :, 0]
    speedY = vcom[0, :, 1]
    speedZ = vcom[0, :, 2]
    AdaptedSpeed = np.array([speedX, speedY, speedZ])


    # preparing
    x = np.linspace(0, 100, 100)
    xplus = np.linspace(0, 100, N)
    y = [traj for traj in AdaptedTraj]
    z = [s for s in AdaptedSpeed]

    # increase data size with scipy
    trajXplus = CubicSpline(x, y[0])(xplus)
    trajYplus = CubicSpline(x, y[1])(xplus)
    trajZplus = CubicSpline(x, y[2])(xplus)

    speedXplus = CubicSpline(x, z[0])(xplus)
    speedYplus = CubicSpline(x, z[1])(xplus)
    speedZplus = CubicSpline(x, z[2])(xplus)

    # plot a before / after data exemple
    plt.clf()
    plt.plot(xplus, trajXplus, '-')
    plt.grid()
    plt.plot(x, trajX)
    #plt.show()


    # save data to file to same shape it was before
    pcom = np.zeros((1, N, 3))
    pcom[0, :, 0] = trajXplus
    pcom[0, :, 1] = trajYplus
    pcom[0, :, 2] = trajZplus

    vcom = np.zeros((1, N, 3))
    vcom[0, :, 0] = speedXplus
    vcom[0, :, 1] = speedYplus
    vcom[0, :, 2] = speedZplus

    # saving
    np.save("/home/nalbrecht/Bolt-Estimator/Bolt-robot---Estimator/tests/com_pos_superplus.npy", pcom)
    np.save("/home/nalbrecht/Bolt-Estimator/Bolt-robot---Estimator/tests/vcom_pos_superplus.npy", vcom)



main()











