import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib import ticker, cm
from matplotlib import colors

# Loading the data
x, y, b, p, alfvenVelocity, corotation, corotationcheck = np.loadtxt('alfvenCheck.txt', delimiter='\t', unpack=True)
# r, scaleHeight = np.loadtxt('scaleheighttest.txt', delimiter='\t', unpack=True)
#
# radius, alfven, radial = np.loadtxt('alfvenradial.txt', delimiter='\t', unpack=True)

# Creating grid
maxR = 100
minR = 6
xtest = np.arange(-maxR, maxR+1, 0.5)
ytest = xtest
xtest, ytest = np.meshgrid(xtest, ytest)

# Masking a circle of radius 20 R_J
mask = (np.sqrt(xtest ** 2 + ytest ** 2) < minR) | (np.sqrt(xtest ** 2 + ytest ** 2) > maxR)

# Making the 3D grid for the magnetic field
BGrid = griddata((x, y), b, (xtest, ytest))
BGrid[mask] = np.nan

# Making the 3D grid for the plasma density
PGrid = griddata((x, y), p, (xtest, ytest))
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

# Plotting
# plt.figure()
# heatmap = plt.contourf(xtest, ytest, BGrid, cmap=plt.cm.get_cmap('gist_rainbow'), alpha=0.4)
# lines = plt.contour(xtest, ytest, BGrid, 5, colors='k')
# plt.clabel(lines, fontsize=18, inline=1, colors='k')
# clb = plt.colorbar(heatmap)
# clb.ax.set_title('B$_n$ (nT)', fontsize=18)
# plt.rcParams['xtick.labelsize'] = 18
# plt.rcParams['ytick.labelsize'] = 18
# plt.xlabel('x $(R_J)$', fontsize=18)
# plt.ylabel('y $(R_J)$', fontsize=18)
# #plt.axis(xlim=(np.amin(xtest), np.amax(xtest)), ylim=(np.amin(ytest), np.amax(ytest)))
# plt.xticks(size=18)
# plt.yticks(size=18)
# plt.tight_layout()
# plt.text(10, -70, r'$\rightarrow$ To the Sun')

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
# plt.text(10, -70, r'$\rightarrow$ To the Sun')
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
ax = plt.subplot(221)
heatmap = plt.contourf(xtest, ytest, AVGrid, cmap=plt.cm.get_cmap('gist_rainbow'), alpha=0.4)
lines = plt.contour(xtest, ytest, CheckGrid, 1, colors='k')
Jupiter = plt.Circle((0, 0), radius=1, color='k')
ax.add_artist(Jupiter)
clb = plt.colorbar(heatmap)
clb.ax.set_title(r'(kms$^{-1}$)', fontsize=18)
plt.title('Alfven V', fontsize=18, wrap=True)
plt.xlabel('x $(R_J)$', fontsize=18)
plt.ylabel('y $(R_J)$', fontsize=18)
plt.xticks(size=18)
plt.yticks(size=18)

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

ax = plt.subplot(223)
lines = plt.contour(xtest, ytest, CheckGrid, 1, colors='k')
Jupiter = plt.Circle((0, 0), radius=1, color='k')
ax.add_artist(Jupiter)
plt.title('Alfven Radius', fontsize=18, wrap=True)
plt.xlabel('x $(R_J)$', fontsize=18)
plt.ylabel('y $(R_J)$', fontsize=18)
plt.xticks(size=18)
plt.yticks(size=18)

ax = plt.subplot(224)
alfvenmask = (alfvenVelocity > 0.99*corotation) & (alfvenVelocity < 1.01*corotation)
calculatedRadius = np.sqrt(x ** 2 + y ** 2)
phiwrong = np.arctan2(x, y)
phi = np.mod(phiwrong, 2*np.pi) * 180 / np.pi
# fit = np.poly1d(np.polyfit(phi[alfvenmask], calculatedRadius[alfvenmask], 3))
plt.scatter(phi[alfvenmask], calculatedRadius[alfvenmask], s=0.1, color='k')
# fitrange = np.arange(0, 360, 1)
# plt.plot(fit(fitrange))
plt.title('Alfven Radius', fontsize=18, wrap=True)
plt.xlabel('Angle (Degrees)', fontsize=18)
plt.ylabel('Radius (R$_J)$', fontsize=18)
plt.xticks(size=18)
plt.yticks(size=18)

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