import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.ticker import (AutoMinorLocator)
import matplotlib.ticker as tick
from matplotlib import ticker, cm
from matplotlib import colors


def sph_cart(r, phi):
    """
    From Chris' code
    :param r: in R_J
    :param theta: in radians
    :param phi: in radians
    :return:
    """
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y


# Loading the data
x, y, b, p, alfvenVelocity, corotation, corotationcheck = np.loadtxt('alfvenCheck.txt', delimiter='\t', unpack=True)
# r, scaleHeight = np.loadtxt('scaleheighttest.txt', delimiter='\t', unpack=True)
#
# radius, alfven, radial = np.loadtxt('alfvenradial.txt', delimiter='\t', unpack=True)

# Creating grid
maxR = 100
minR = 6
step = 0.2
xtest = np.arange(-maxR, maxR+step, step)
ytest = xtest
xtest, ytest = np.meshgrid(xtest, ytest)

# Masking a circle of radius 20 R_J
mask = (np.sqrt(xtest ** 2 + ytest ** 2) < minR) | (np.sqrt(xtest ** 2 + ytest ** 2) > maxR)

# Making the 3D grid for the magnetic field
BGrid = griddata((x, y), np.log10(b), (xtest, ytest))
BGrid[mask] = np.nan

# Making the 3D grid for the plasma density
PGrid = griddata((x, y), np.log10(p), (xtest, ytest))
PGrid[mask] = np.nan

# Making the 3D grid for the Alfven Velocity
AVGrid = griddata((x, y), alfvenVelocity/1000, (xtest, ytest))
AVGrid[mask] = np.nan

# Making the 3D grid for the Corotation Velocity
VGrid = griddata((x, y), corotation/1000, (xtest, ytest))
VGrid[mask] = np.nan

# Making the 3D grid for the Alfven radius
CheckGrid = griddata((x, y), corotationcheck, (xtest, ytest))
CheckGrid[mask] = np.nan

#
# r = [point for point in range(6, 100)]
# r = np.array(r)
# x1, y1 = sph_cart(r, 0.5*np.pi)

# Plotting
fig, ax = plt.subplots()
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
heatmap = plt.contourf(xtest, ytest, BGrid, cmap=plt.cm.get_cmap('gist_rainbow'), levels=40, alpha=0.4)
# plt.plot(x1, y1)
# lines = plt.contour(xtest, ytest, BGrid, 5, colors='k')
# plt.clabel(lines, fontsize=18, inline=1, colors='k')
clb = plt.colorbar(heatmap)
clb.ax.set_title('B$_n$ (nT)', fontsize=18)
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.xlabel('x $(R_J)$', fontsize=18)
plt.ylabel('y $(R_J)$', fontsize=18)
#plt.axis(xlim=(np.amin(xtest), np.amax(xtest)), ylim=(np.amin(ytest), np.amax(ytest)))
plt.xticks(np.arange(np.amin(xtest), np.amax(xtest)+1, 20), size=18)
ax.tick_params(axis='both', which='major', size=6)
ax.tick_params(axis='both', which='minor', size=4)
ax.xaxis.set_major_locator(tick.MultipleLocator(25))
ax.xaxis.set_minor_locator(tick.MultipleLocator(5))
ax.yaxis.set_major_locator(tick.MultipleLocator(25))
ax.yaxis.set_minor_locator(tick.MultipleLocator(5))
ax.tick_params(right=True, which='both', labelsize=18)
# ax.xaxis.set_minor_locator(AutoMinorLocator())
# plt.grid(which='minor', axis='both', visible=True)
plt.yticks(size=18)
plt.tight_layout()
plt.text(10, -30, r'$\leftarrow$ To the Sun', size=18)

# plt.figure()
# heatmap = plt.contourf(xtest, ytest, PGrid, cmap=plt.cm.get_cmap('gist_rainbow'), alpha=0.4)
# lines = plt.contour(xtest, ytest, PGrid, 5, colors='k')
# plt.clabel(lines, fontsize=18, inline=1, colors='k')
# clb = plt.colorbar(heatmap)
# clb.ax.set_title(r'(cm$^{-3}$)', fontsize=18)
# plt.title('Plasma density at Jupiter', fontsize=18, wrap=True)
# plt.rcParams['xtick.labelsize'] = 18
# plt.rcParams['ytick.labelsize'] = 18
# plt.xlabel('x $(R_J)$', fontsize=18)
# plt.ylabel('y $(R_J)$', fontsize=18)
# plt.xticks(size=18)
# plt.yticks(size=18)
# plt.tight_layout()

# plt.figure()
# cmap = colors.ListedColormap(['red', '#ffffff'])
# boundaries = [0, 1]
# norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
# CheckGrid = griddata((x, y), corotationcheck, (xtest, ytest), method='linear')
# CheckGrid[mask] = np.nan
# #heatmap = plt.contourf(xtest, ytest, CheckGrid, cmap=cmap)
# #heatmap = plt.contourf(xtest, ytest, CheckGrid, cmap=plt.cm.get_cmap('gist_rainbow'), alpha=0.4)
# lines = plt.contour(xtest, ytest, CheckGrid, 1, colors='k')
# plt.xlabel('x $(R_J)$', fontsize=18)
# plt.ylabel('y $(R_J)$', fontsize=18)
# plt.xticks(size=18)
# plt.yticks(size=18)
# plt.text(10, -20, r'$\rightarrow$ To the Sun')
# plt.tight_layout()

# plt.figure()
# alfvenmask = (alfvenVelocity > 0.95*corotation) & (alfvenVelocity < 1.05*corotation)
# calculatedRadius = np.sqrt(x ** 2 + y ** 2)
# phiwrong = np.arctan2(x, y)
# phi = np.mod(phiwrong, 2*np.pi) * 180 / np.pi
# fit = np.poly1d(np.polyfit(phi[alfvenmask], calculatedRadius[alfvenmask], 3))
# plt.scatter(phi[alfvenmask], calculatedRadius[alfvenmask], s=0.1, color='k')
# fitrange = np.arange(0, 360, 1)
# plt.plot(fit(fitrange))
# plt.xlabel('Angle (Degrees)', fontsize=18)
# plt.ylabel('Radius (R$_J)$', fontsize=18)
# plt.xticks(size=18)
# plt.yticks(size=18)
# plt.tight_layout()

plt.figure()
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.tight_layout()
ax1 = plt.subplot(221)
heatmap = plt.contourf(xtest, ytest, AVGrid, cmap=plt.cm.get_cmap('gist_rainbow'), alpha=0.4)
lines = plt.contour(xtest, ytest, CheckGrid, 1, colors='k', linewidths=2)
Jupiter = plt.Circle((0, 0), radius=1, color='k')
ax1.add_artist(Jupiter)
clb = plt.colorbar(heatmap)
clb.ax.set_title(r'(kms$^{-1}$)', fontsize=18)
plt.title('Alfvén V', fontsize=18, wrap=True)
plt.xlabel('x $(R_J)$', fontsize=18)
plt.ylabel('y $(R_J)$', fontsize=18)
plt.xticks(size=18)
plt.yticks(size=18)
ax1.tick_params(axis='both', which='major', size=6)
ax1.tick_params(axis='both', which='minor', size=4)
ax1.xaxis.set_major_locator(tick.MultipleLocator(50))
ax1.xaxis.set_minor_locator(tick.MultipleLocator(10))
ax1.yaxis.set_major_locator(tick.MultipleLocator(50))
ax1.yaxis.set_minor_locator(tick.MultipleLocator(10))

ax = plt.subplot(222)
heatmap = plt.contourf(xtest, ytest, VGrid, cmap=plt.cm.get_cmap('gist_rainbow'), alpha=0.4)
# lines = plt.contour(xtest, ytest, VGrid, 5, colors='k')
# plt.clabel(lines, inline=1, colors='k')
Jupiter = plt.Circle((0, 0), radius=1, color='k')
ax.add_artist(Jupiter)
clb = plt.colorbar(heatmap)
clb.ax.set_title(r'(kms$^{-1}$)', fontsize=18)
plt.title('Radial V', fontsize=18, wrap=True)
plt.xlabel('x $(R_J)$', fontsize=18)
plt.ylabel('y $(R_J)$', fontsize=18)
plt.xticks(size=18)
plt.yticks(size=18)
ax.tick_params(axis='both', which='major', size=6)
ax.tick_params(axis='both', which='minor', size=4)
ax.xaxis.set_major_locator(tick.MultipleLocator(50))
ax.xaxis.set_minor_locator(tick.MultipleLocator(10))
ax.yaxis.set_major_locator(tick.MultipleLocator(50))
ax.yaxis.set_minor_locator(tick.MultipleLocator(10))

ax = plt.subplot(223)
lines = plt.contour(xtest, ytest, CheckGrid, 1, colors='k', linewidths=2)
Jupiter = plt.Circle((0, 0), radius=1, color='k')
ax.add_artist(Jupiter)
plt.title('Alfvén Radius', fontsize=18, wrap=True)
plt.xlabel('x $(R_J)$', fontsize=18)
plt.ylabel('y $(R_J)$', fontsize=18)
plt.xticks(size=18)
plt.yticks(size=18)
ax.tick_params(axis='both', which='major', size=6)
ax.tick_params(axis='both', which='minor', size=4)
ax.xaxis.set_major_locator(tick.MultipleLocator(50))
ax.xaxis.set_minor_locator(tick.MultipleLocator(10))
ax.yaxis.set_major_locator(tick.MultipleLocator(50))
ax.yaxis.set_minor_locator(tick.MultipleLocator(10))

ax = plt.subplot(224)
alfvenmask = (alfvenVelocity > 0.95*corotation) & (alfvenVelocity < 1.05*corotation)
calculatedRadius = np.sqrt(x ** 2 + y ** 2)
phiwrong = np.arctan2(x, y)
phi = np.mod(phiwrong, 2*np.pi) * 180 / np.pi
# fit = np.poly1d(np.polyfit(phi[alfvenmask], calculatedRadius[alfvenmask], 3))
plt.scatter(phi[alfvenmask], calculatedRadius[alfvenmask], s=0.5, color='k')
# fitrange = np.arange(0, 360, 1)
# plt.plot(fit(fitrange))
y = [r for r in range(28, 60)]
dawn = [90] * len(y)
noon = [180] * len(y)
dusk = [270] * len(y)
plt.plot(dawn, y, linestyle='-', color='grey', alpha=0.4)
plt.plot(noon, y, linestyle='-', color='grey', alpha=0.4)
plt.plot(dusk, y, linestyle='-', color='grey', alpha=0.4)
plt.title('Alfvén Radius', fontsize=18, wrap=True)
plt.xlabel('Angle (Degrees)', fontsize=18)
plt.ylabel('Radius (R$_J)$', fontsize=18)
ax.tick_params(axis='both', which='major', size=6)
ax.tick_params(axis='both', which='minor', size=4)
ax.xaxis.set_major_locator(tick.MultipleLocator(90))
ax.xaxis.set_minor_locator(tick.MultipleLocator(30))
# ax.yaxis.set_major_locator(tick.MultipleLocator(30))
ax.yaxis.set_minor_locator(tick.MultipleLocator(5))
plt.xticks(size=18)
plt.yticks(size=18)
plt.xlim(0, 360)
plt.ylim(49.8, 50.7)
# plt.savefig('tester.png', transparent=True)
# plt.figure()
# plt.plot(radius, alfven/1000, 'k', label='Alfven')
# plt.plot(radius, radial/1000, 'r', Label='Radial')
# plt.yscale('log')
# plt.ylim(1)
# plt.ylabel('Velocity (km/s)', fontsize=18)
# plt.xlabel('RJ', fontsize=18)
# plt.legend(fontsize=18)
# plt.xticks(size=18)
# plt.yticks(size=18)
# plt.tight_layout()

# plt.figure()
# heatmap = plt.contourf(xtest, ytest, AVGrid, cmap=plt.cm.get_cmap('gist_rainbow'), alpha=0.4)
# lines = plt.contour(xtest, ytest, AVGrid, 5, colors='k')
# plt.clabel(lines, inline=1, colors='k')
# clb = plt.colorbar(heatmap)
# clb.ax.set_title(r'V$_A$ (ms$^{-1}$)', fontsize=18)
# plt.title('Alfven Velocity at Jupiter', fontsize=18, wrap=True)
# plt.rcParams['xtick.labelsize'] = 18
# plt.rcParams['ytick.labelsize'] = 18
# plt.xlabel('x $(R_J)$', fontsize=18)
# plt.ylabel('y $(R_J)$', fontsize=18)
# #plt.axis(xlim=(np.amin(xtest), np.amax(xtest)), ylim=(np.amin(ytest), np.amax(ytest)))
# plt.xticks(size=18)
# plt.yticks(size=18)
# plt.text(10, -70, r'$\rightarrow$ To the Sun')
# plt.tight_layout()

# plt.figure()
# plt.plot(r, scaleHeight)
# plt.axis(xmax=100, ymax=scaleHeight[100]+0.2)
# plt.xlabel('Radius (R$_J)$', fontsize=18)
# plt.ylabel('Scale Height (R$_J)$', fontsize=18)
# plt.xticks(size=18)
# plt.yticks(size=18)
# plt.tight_layout()

# plt.figure()
# heatmap = plt.contourf(xtest, ytest, AVGrid, cmap=plt.cm.get_cmap('gist_rainbow'), alpha=0.4)
# plt.contour(xtest, ytest, CheckGrid, 1, colors='k')
# clb = plt.colorbar(heatmap)
# clb.ax.set_title(r'(kms$^{-1}$)', fontsize=18)
# plt.xlabel('x $(R_J)$', fontsize=18)
# plt.ylabel('y $(R_J)$', fontsize=18)
# plt.xticks(size=18)
# plt.yticks(size=18)
# plt.tight_layout()

plt.show()