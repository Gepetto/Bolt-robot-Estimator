import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline


"""
Increase the number of points in Constant's curves

N is the final number of points

"""

def main():
    #improve_0(N=500)
    improve(N=500, filename="/home/nalbrecht/Bolt-Estimator/Bolt-robot---Estimator/data/X_array_4.npy",
                     outname="/home/nalbrecht/Bolt-Estimator/Bolt-robot---Estimator/data/improved/X_array_4.npy")
    

def improve_0(N=300):

    # get data and extract data
    pcom = np.load("/home/nalbrecht/Bolt-Estimator/Bolt-robot---Estimator/tests/com_pos.npy")
    vcom = np.load("/home/nalbrecht/Bolt-Estimator/Bolt-robot---Estimator/tests/vcom_pos.npy")
    print(pcom.shape)
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
    plt.plot(xplus, trajXplus)
    plt.grid()
    plt.plot(x, trajX, '--')
    plt.show()


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

def improve(N=1000, filename = "", outname="", talk=True):
    data = np.load(filename)
    if talk : print(data.ndim)
    if data.ndim == 3:
        improve_3(N, filename, outname, talk)
    elif data.ndim == 1:
        improve_1(N, filename, outname, talk)
    elif data.ndim == 2:
        improve_2(N, filename, outname, talk)
    elif data.ndim == 4:
        improve_4(N, filename, outname, talk)


def improve_3(N=1000, filename = "", outname="", talk=True):
    """Load data, improve its resolution et save it """

    # get data and extract data
    # data is n * ndata * ndim=3
    data = np.load(filename)
    Ninit, ndata, ndim = data.shape
  
    if talk : print(f"Loaded data with\n\t\t Ninit={Ninit}, \n\t\t ndata = {ndata}\n\t\t ndim={ndim}")
    
    # preparing
    x = np.linspace(0, Ninit, Ninit)
    xplus = np.linspace(0, Ninit, N)
    dataplus = np.zeros((N, ndata, ndim))
    for k in range(ndata):
        y = data[:, k, :]
        #print(y)
        # improving size of y
        improved_data = CubicSpline(x, y)(xplus)
        dataplus[:, k, :] = improved_data

    # plot a before / after data exemple
    # plt.clf()
    # plt.grid()
    # plt.plot(x, data[:, 2, :], label="original data")
    # plt.plot(xplus, dataplus[:, 2, :], '--', label="improved data")
    # plt.show()

    # saving
    if outname=="":
        outname = filename[:-4] + "_improved.npy"
    np.save(outname, dataplus)
    if talk : print(f"-> improved data saved with size {N} to " + outname)




def improve_2(N=1000, filename = "", outname="", talk=True):
    """Load data, improve its resolution et save it """

    # get data and extract data
    # data is n * ndata * ndim=2
    data = np.load(filename)
    Ninit, ndim = data.shape
    ndata = 1
  
    if talk : print(f"Loaded data with\n\t\t Ninit={Ninit}, \n\t\t ndata = {ndata}\n\t\t ndim={ndim}")
    
    # preparing
    x = np.linspace(0, Ninit, Ninit)
    xplus = np.linspace(0, Ninit, N)
    dataplus = np.zeros((N, ndim))
    y = data
    #print(y)
    # improving size of y
    improved_data = CubicSpline(x, y)(xplus)
    dataplus = improved_data

    # plot a before / after data exemple
    # plt.clf()
    # plt.grid()
    # plt.plot(x, data[:, 2], label="original data")
    # plt.plot(xplus, dataplus[:, 2], '--', label="improved data")
    #plt.show()

    # saving
    if outname=="":
        outname = filename[:-4] + "_improved.npy"
    np.save(outname, dataplus)
    if talk : print(f"-> improved data saved with size {N} to " + outname)


def improve_1(N=1000, filename = "", outname="", talk=True):
    """Load data, improve its resolution et save it """

    # get data and extract data
    # data is n * ndata * ndim=1
    data = np.load(filename)
    #print(data.ndim)
    Ninit, = data.shape
    ndata = 1
    ndim = 1
  
    if talk : print(f"Loaded data with\n\t\t Ninit={Ninit}, \n\t\t ndata = {ndata}\n\t\t ndim={ndim}")
    
    # preparing
    x = np.linspace(0, Ninit, Ninit)
    xplus = np.linspace(0, Ninit, N)
    dataplus = np.zeros(N)
    y = data
    #print(y)
    # improving size of y
    improved_data = CubicSpline(x, y)(xplus)
    dataplus[:] = improved_data[:]

    # plot a before / after data exemple
    # plt.clf()
    # plt.grid()
    # plt.plot(x, data, label="original data")
    # plt.plot(xplus, dataplus, '--', label="improved data")
    # plt.show()

    # saving
    if outname=="":
        outname = filename[:-4] + "_improved.npy"
    np.save(outname, dataplus)
    if talk : print(f"-> improved data saved with size {N} to " + outname)


def improve_4(N=1000, filename = "", outname="", talk=True):
    """Load data, improve its resolution et save it """

    # get data and extract data
    # data is n * ndata * ndim=3
    data = np.load(filename)
    Ninit, ndata, ndim1, ndim2 = data.shape
  
    if talk : print(f"Loaded data with\n\t\t Ninit={Ninit}, \n\t\t ndata = {ndata}\n\t\t ndim={ndim1} by {ndim2}")
    
    # preparing
    x = np.linspace(0, Ninit, Ninit)
    xplus = np.linspace(0, Ninit, N)
    dataplus = np.zeros((N, ndata, ndim1, ndim2))
    for k in range(ndata):
        y = data[:, k, :, :]
        #print(y)
        # improving size of y
        improved_data = CubicSpline(x, y)(xplus)
        dataplus[:, k, :, :] = improved_data

    # plot a before / after data exemple
    # plt.clf()
    # plt.grid()
    # plt.plot(x, data[:, 2, :, 1], label="original data")
    # plt.plot(xplus, dataplus[:, 2, :, 1], '--', label="improved data")
    # plt.show()

    # saving
    if outname=="":
        outname = filename[:-4] + "_improved.npy"
    np.save(outname, dataplus)
    if talk : print(f"-> improved data saved with size {N} to " + outname)




if __name__ =='main':
    main()











