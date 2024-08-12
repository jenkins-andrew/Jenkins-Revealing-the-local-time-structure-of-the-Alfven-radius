from mpl_toolkits import mplot3d

from scipy.interpolate import griddata
import creating2DProfiles
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import os
import matplotlib.ticker as tick

speciesList = {'e-': [1, 2451, -6.27, 4.21],
               'o+': [0.24, 592, -7.36, 0.368],
               'o++': [0.03, 76.3, -6.73, 0.086],
               's+': [0.07, 163, -6.81, 0.169],
               's++': [0.22, 538, -6.74, 0.598],
               's+++': [0.004, 90.7, -6.21, 0.165],
               'h+': [0.02, 50.6, -5.31, 0.212],
               'na+': [0.04, 97.2, -6.75, 0.106],
               'hoto+': [0.06, 134, -4.63, 1.057]}
ME = 0.00054858
speciesMass = {'e-': 0.00054858,
               'o++': 15.999 - (ME * 2),
               's+': 32.065 - ME,
               's++': 32.065 - (ME * 2),
               's+++': 32.065 - (ME * 3),
               'h+': 1.00784 - ME,
               'na+': 22.989769 - ME,
               'hoto+': 15.999 - (ME * 2),
               'o+': 15.999 - ME
               }


def sph_cart(r, theta, phi):
    """
    From Chris' code
    :param r:
    :param theta:
    :param phi:
    :return:
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def combineTraces(path, skipNumber=1):
    loaded = np.load(path, allow_pickle=True)
    output = []
    for i in range(0, len(loaded), skipNumber):
        output.extend(loaded[i])
    np.savetxt('temp.txt', np.c_[output], delimiter='\t')


def orbitalTraces(path):
    file = open('orbitalAngle.txt', 'w+')
    lineNumber, tNorth, ftNorth, tSouth, ftSouth, fD = np.loadtxt(path, delimiter='\t', unpack=True)
    angleconversion = 2*np.pi / 9.9250
    mask = (lineNumber < 60)
    print(lineNumber)
    tNorth = angleconversion * (tNorth[mask] + tSouth[mask])/3600
    fig, ax = plt.subplots()
    step = 9.9250*3600 * 0.01/2
    for i in range(len(lineNumber[mask])):
        x, y = [], []
        file.write(str(lineNumber[i])+'\t')
        for j in np.arange(0, tNorth[i], 0.01*np.pi):
            lineNumber[i] += step * creating2DProfiles.radialVelocityFunc(lineNumber[i], speciesList, speciesMass)/71492e3
            x.append(lineNumber[i] * np.cos(j))
            y.append(lineNumber[i] * np.sin(j))
        plt.plot(x, y, color='yellow', linewidth=4)
        file.write(str(lineNumber[i])+'\t'+str(tNorth[i]*180/np.pi)+'\n')
    # plt.tick_params(right=True, which='both', labelsize=18)
    # plt.xlabel(r'x (R$_J$)', size=18)
    # plt.ylabel(r'y (R$_J$)', size=18)
    plt.xlim(-60, 60)
    plt.ylim(-60, 60)
    Jupiter = plt.Circle((0, 0), radius=1, color='k')
    outerCircle = plt.Circle((0, 0), radius=60, color='k', fill=False)
    ax.add_artist(Jupiter)
    ax.add_artist(outerCircle)
    fig.patch.set_visible(False)
    ax.axis('off')
    plt.tight_layout()
    # plt.legend(fontsize=18)
    plt.show()
    # plt.savefig('Figures/orbitalTrace.png', transparent=True)

def vPhiCalc(r):
    vPhi = -0.1461*r**2 + 16.59*r - 64.99
    # vPhi = r*9.286 + 23.57
    return vPhi*1e3/71492e3

def betterOrbitalTrace(path):
    file = open('orbitalAngle.txt', 'w+')
    lineNumber, tNorth, ftNorth, tSouth, ftSouth, fD = np.loadtxt(path, delimiter='\t', unpack=True)
    mask = (lineNumber < 60)
    totalTravelTime = (tNorth[mask] + tSouth[mask])
    fig, ax = plt.subplots()
    step = 1
    for i in range(len(lineNumber[mask])):
        x, y = [], []
        file.write(str(round(lineNumber[i], 2))+'\t')
        angle = 0
        length = 0
        newRadialDistance = lineNumber[i]
        for j in np.arange(0, totalTravelTime[i], step):
            angle += step * vPhiCalc(newRadialDistance) *np.pi/180
            newRadialDistance += step * creating2DProfiles.radialVelocityFunc(newRadialDistance, speciesList, speciesMass)/71492e3
            x.append(newRadialDistance * np.cos(angle))
            y.append(newRadialDistance * np.sin(angle))
            if len(x) > 1:
                length += np.sqrt((x[-1]-x[-2])**2+(y[-1]-y[-2])**2)
        plt.plot(x, y, color='yellow', linewidth=4)
        file.write(str(round(totalTravelTime[i], 2))+'\t'+str(round(newRadialDistance-lineNumber[i], 2))+'\t'+str(round(angle*180/np.pi, 2))+'\t'+str(round(length, 2))+'\n')
    # plt.tick_params(right=True, which='both', labelsize=18)
    # plt.xlabel(r'x (R$_J$)', size=18)
    # plt.ylabel(r'y (R$_J$)', size=18)
    plt.xlim(-60, 60)
    plt.ylim(-60, 60)
    Jupiter = plt.Circle((0, 0), radius=1, color='k')
    outerCircle = plt.Circle((0, 0), radius=60, color='k', fill=False)
    ax.add_artist(Jupiter)
    ax.add_artist(outerCircle)
    fig.patch.set_visible(False)
    ax.axis('off')
    plt.tight_layout()
    # plt.legend(fontsize=18)
    # plt.show()
    plt.savefig('Figures/latestOrbitalTrace.png', transparent=True)

def angleTravelledThrough(path):
    lineNumber, tNorth, ftNorth, tSouth, ftSouth, fD = np.loadtxt(path, delimiter='\t', unpack=True)
    totalTravelTime = (tNorth + tSouth)
    step = 1
    angle = 0
    for j in np.arange(0, totalTravelTime, step):
        angle += step * vPhiCalc(lineNumber)
    return angle

def plotMultiplePhis(directory):

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    files = [f for f in os.listdir(directory)]
    for i in range(len(files)):
        path = directory + files[i]
        combineTraces(path)
        x, y, z, B = np.loadtxt('temp.txt', delimiter='\t', unpack=True)
        plt.plot(x, y, z, '-k')
    plt.show()


def plotOnePhiSet(path):
    combineTraces(path)
    x, y, z, B, rho, alfvenVelocity, radialVelocity = np.loadtxt('temp.txt', delimiter='\t', unpack=True)
    #xc, yc, zc, Bc = np.loadtxt('newoutput/radius15.00to15.00phi5.10CurrentOn=False.txt', delimiter='\t', unpack=True)

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    plt.plot(x, y, z, '-k')
    # plt.plot(xc, yc, zc, '--')
    # img = ax.scatter(x, y, z, c=B, cmap=plt.cm.get_cmap('gist_rainbow'))
    # clb = fig.colorbar(img)
    # clb.ax.set_title(r'B (nT)', fontsize=18)

    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    xs = np.cos(u)*np.sin(v)
    ys = np.sin(u)*np.sin(v)
    zs = np.cos(v)
    ax.plot_surface(xs, ys, zs, color="r")

    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0

    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_xlabel('x RJ', size=18)
    ax.set_ylabel('y RJ', size=18)
    ax.set_zlabel('z RJ', size=18)

    plt.xticks(size=10)
    plt.yticks(size=10)
    ax.zaxis.set_tick_params(labelsize=10)

    plt.show()


def plotCorotation(path):
    combineTraces(path)
    x, y, z, B, rho, alfvenVelocity, radialVelocity = np.loadtxt('temp.txt', delimiter='\t', unpack=True)
    maxR = 30
    minR = 6
    step = 0.25
    radius = np.sqrt(x ** 2 + y ** 2)
    xtest = np.arange(minR, np.amax(radius) + step, step)
    ztest = np.arange(np.amin(z), np.amax(z) + step, step)
    xtest, ztest = np.meshgrid(xtest, ztest)

    corotationMask = (radialVelocity < alfvenVelocity)
    corotationBackwardsMask = (radialVelocity > alfvenVelocity)

    alfvenPointCheck = []
    for i in range(len(alfvenVelocity)):
        if alfvenVelocity[i] > radialVelocity[i]:
            alfvenPointCheck.append(0)
        else:
            alfvenPointCheck.append(1)

    # Masking a circle of radius minR R_J
    # mask = (xtest < minR) | (np.sqrt(xtest ** 2 + ztest ** 2) > maxR)
    mask = (rho > 10 ** (-24)) & (rho < 10 ** (-16))

    # Making the 3D grid for the magnetic field
    BGrid = griddata((radius, z), np.log10(B), (xtest, ztest))
    outputradius, outputdensity = [], []

    NGrid = griddata((radius[corotationMask], z[corotationMask]), np.log10(rho[corotationMask]), (xtest, ztest))
    notNGrid = griddata((radius[corotationBackwardsMask], z[corotationBackwardsMask]), np.log10(rho[corotationBackwardsMask]), (xtest, ztest))
    # NGrid[mask] = np.nan

    AlfvenGrid = griddata((radius, z), np.log10(alfvenVelocity), (xtest, ztest))
    # AlfvenGrid[mask] = np.nan
    #
    RadialGrid = griddata((radius, z), np.log10(radialVelocity), (xtest, ztest))
    # RadialGrid[mask] = np.nan

    file = open('equatorialValues.txt', 'w+')
    equator = int((0 - np.amin(z)) / step)
    for rtest in np.arange(6, 80):
        arrayNumber = int((rtest - minR) / step)
        file.write(str(rtest)+'\t'+str(BGrid[equator, arrayNumber])+'\n')

    # AlfvenPointGrid = griddata((radius, z), alfvenPointCheck, (xtest, ztest))
    # AlfvenPointGrid[mask] = np.nan
    combineTraces(path, 3)
    x2, y2, z2, B2, rho2, alfvenVelocity2, radialVelocity2 = np.loadtxt('temp.txt', delimiter='\t', unpack=True)
    radius2 = np.sqrt(x2 ** 2 + y2 ** 2)
    #

    # plt.figure()
    # plt.plot(r, alfvenVelocity/1000, 'k', label='Alfven')
    # plt.plot(r, radialVelocity/1000, 'r', Label='Radial')
    # plt.yscale('log')
    # plt.ylim(1)
    # plt.ylabel('Velocity (km/s)', fontsize=18)
    # plt.xlabel('RJ', fontsize=18)
    # plt.legend(fontsize=18)
    # plt.xticks(size=18)
    # plt.yticks(size=18)
    # plt.tight_layout()
    #
    # plt.figure()
    # plt.rcParams['xtick.labelsize'] = 18
    # plt.rcParams['ytick.labelsize'] = 18
    # heatmap = plt.contourf(xtest, ztest, BGrid, cmap=plt.cm.get_cmap('gist_rainbow'), levels=30, alpha=0.4)
    # clb = plt.colorbar(heatmap)
    # clb.ax.set_title('B$_n$ $\log$(nT)', fontsize=18)
    # plt.plot(radius2, z2, '--k')
    # plt.xlabel('r $(R_J)$', fontsize=18)
    # plt.ylabel('z $(R_J)$', fontsize=18)
    # plt.xticks(size=18)
    # plt.yticks(size=18)
    # plt.xlim(minR)
    # plt.tight_layout()
    # #
    # plt.figure()
    # heatmap = plt.contourf(xtest, ztest, NGrid, cmap=plt.cm.get_cmap('gist_rainbow'), alpha=0.4)
    # lines = plt.contour(xtest, ztest, NGrid, 5, colors='k')
    # plt.clabel(lines, fontsize=18, inline=1, colors='k')
    # clb = plt.colorbar(heatmap)
    # clb.ax.set_title(r'$\rho$ kgm$^{-3}$', fontsize=18)
    # plt.rcParams['xtick.labelsize'] = 18
    # plt.rcParams['ytick.labelsize'] = 18
    # plt.xlabel('x $(R_J)$', fontsize=18)
    # plt.ylabel('z $(R_J)$', fontsize=18)
    # plt.xlim(minR)
    # plt.xticks(size=18)
    # plt.yticks(size=18)
    # plt.tight_layout()

    fig, ax = plt.subplots()
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    heatmap = plt.contourf(xtest, ztest, NGrid, cmap=plt.cm.get_cmap('gist_rainbow'), levels=20,
                           alpha=0.4)
    plt.contourf(xtest, ztest, notNGrid, colors='white', alpha=0.8)
    plt.plot(radius2, z2, '--k')
    #lines = plt.contour(xtest, ztest, NGrid, 5, colors='k')
    #plt.clabel(lines, inline=1, colors='k')
    clb = plt.colorbar(heatmap)
    clb.ax.set_title(r'$\log$(kgm$^{-3}$)', fontsize=18)
    plt.xlabel('Radius $(R_J)$', fontsize=18)
    plt.ylabel('z $(R_J)$', fontsize=18)
    plt.xticks(size=18)
    plt.yticks(size=18)
    ax.tick_params(axis='both', which='major', size=6)
    ax.tick_params(axis='both', which='minor', size=4)
    ax.xaxis.set_major_locator(tick.MultipleLocator(10))
    ax.xaxis.set_minor_locator(tick.MultipleLocator(5))
    ax.yaxis.set_major_locator(tick.MultipleLocator(10))
    ax.yaxis.set_minor_locator(tick.MultipleLocator(5))
    ax.tick_params(right=True, which='both', labelsize=18)
    plt.xlim(minR)
    #plt.ylim(np.amin(z[corotationMask]), np.amax(z[corotationMask]))
    plt.tight_layout()

    plt.figure()
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    plt.subplots_adjust(wspace=0.5, hspace=0.5, right=1, left=0.15)

    #
    ax = plt.subplot(121)
    heatmap = plt.contourf(xtest, ztest, AlfvenGrid, cmap=plt.cm.get_cmap('gist_rainbow'), levels=15, alpha=0.4)
    #lines = plt.contour(xtest, ztest, AlfvenGrid, 1, colors='k')
    clb = plt.colorbar(heatmap)
    plt.plot(radius2, z2, '--k', alpha=0.6)
    clb.ax.set_title(r'$\log$(ms$^{-1}$)', fontsize=18)
    plt.title('Alfvén Velocity', fontsize=18, wrap=True)
    plt.xlabel('Radius $(R_J)$', fontsize=18)
    plt.ylabel('z $(R_J)$', fontsize=18)
    plt.xticks(size=18)
    plt.yticks(size=18)
    ax.tick_params(axis='both', which='major', size=6)
    ax.tick_params(axis='both', which='minor', size=4)
    ax.xaxis.set_major_locator(tick.MultipleLocator(10))
    ax.xaxis.set_minor_locator(tick.MultipleLocator(5))
    ax.yaxis.set_major_locator(tick.MultipleLocator(10))
    ax.yaxis.set_minor_locator(tick.MultipleLocator(5))
    plt.xlim(minR)

    ax = plt.subplot(122)
    heatmap = plt.contourf(xtest, ztest, RadialGrid, cmap=plt.cm.get_cmap('gist_rainbow'), levels=15, alpha=0.4)
    #lines = plt.contour(xtest, ztest, RadialGrid, 5, colors='k')
    #plt.clabel(lines, inline=1, colors='k')
    clb = plt.colorbar(heatmap)
    plt.plot(radius2, z2, '--k', alpha=0.6)
    clb.ax.set_title(r'$\log$(ms$^{-1}$)', fontsize=18)
    plt.title('Radial Velocity', fontsize=18, wrap=True)
    plt.xlabel('Radius $(R_J)$', fontsize=18)
    plt.ylabel('z $(R_J)$', fontsize=18)
    plt.xticks(size=18)
    plt.yticks(size=18)
    ax.tick_params(axis='both', which='major', size=6)
    ax.tick_params(axis='both', which='minor', size=4)
    ax.xaxis.set_major_locator(tick.MultipleLocator(10))
    ax.xaxis.set_minor_locator(tick.MultipleLocator(5))
    ax.yaxis.set_major_locator(tick.MultipleLocator(10))
    ax.yaxis.set_minor_locator(tick.MultipleLocator(5))
    plt.xlim(minR)
    plt.tight_layout()

    # plt.figure()
    # heatmap = plt.contourf(xtest, ztest, AlfvenPointGrid)
    # plt.title('Alfven Radius', fontsize=18, wrap=True)
    # plt.xlabel('x $(R_J)$', fontsize=18)
    # plt.ylabel('z $(R_J)$', fontsize=18)
    # plt.xticks(size=18)
    # plt.yticks(size=18)
    # plt.xlim(minR)

    #
    # plt.figure()
    # cmap = colors.ListedColormap(['#196F3D', '#1A5276'])
    # boundaries = [0, 1]
    # norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    # heatmap = plt.contourf(xtest, ztest, AlfvenPointGrid, cmap=cmap)
    # lines = plt.contour(xtest, ztest, AlfvenPointGrid, 1, colors='k')
    # plt.xlabel('x $(R_J)$', fontsize=18)
    # plt.ylabel('y $(R_J)$', fontsize=18)
    # plt.xticks(size=18)
    # plt.yticks(size=18)
    # plt.xlim(minR)
    # plt.tight_layout()

    # ax = plt.subplot(224)
    # alfvenmask = (alfvenVelocity > 0.95*radialVelocity) & (alfvenVelocity < 1.05*radialVelocity)
    # calculatedRadius = np.sqrt(x ** 2 + y ** 2)
    # phiwrong = np.arctan2(x, z)
    # phi = np.mod(phiwrong, 2*np.pi) * 180 / np.pi
    # # fit = np.poly1d(np.polyfit(phi[alfvenmask], calculatedRadius[alfvenmask], 3))
    # plt.scatter(phi[alfvenmask], calculatedRadius[alfvenmask], s=0.1, color='k')
    # # fitrange = np.arange(0, 360, 1)
    # # plt.plot(fit(fitrange))
    # plt.title('Alfven Radius', fontsize=18, wrap=True)
    # plt.xlabel('Angle (Degrees)', fontsize=18)
    # plt.ylabel('Radius (R$_J)$', fontsize=18)
    # plt.xticks(size=18)
    # plt.yticks(size=18)
    # plt.xlim(minR)
    plt.show()

#plotOnePhiSet('newoutput/radius15.00to15.00phi5.10CurrentOn=True.txt')