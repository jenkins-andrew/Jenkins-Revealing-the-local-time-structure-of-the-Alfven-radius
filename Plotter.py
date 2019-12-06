from scipy.interpolate import griddata
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib import ticker, cm

x, r, z, B, rho, alfvenVelocity, radialVelocity, alfvenPointCheck = np.loadtxt('temporaryFile.txt', delimiter='\t', unpack=True)
maxR = 30
minR = 6
xtest = np.arange(-maxR, maxR+1, 0.5)
ztest = np.arange(-12, 12+1, 0.5)
xtest, ztest = np.meshgrid(xtest, ztest)

# Masking a circle of radius minR R_J
#mask = (xtest < minR) | (np.sqrt(xtest ** 2 + ztest ** 2) > maxR)
mask = (rho > 10**(-24)) & (rho < 10**(-16))

# Making the 3D grid for the magnetic field
# BGrid = griddata((x, z), B, (xtest, ztest))
# BGrid[mask] = np.nan

NGrid = griddata((x[mask], z[mask]), rho[mask], (xtest, ztest))
#NGrid[mask] = np.nan

# AlfvenGrid = griddata((x, z), alfvenVelocity/1000, (xtest, ztest))
# AlfvenGrid[mask] = np.nan
#
# RadialGrid = griddata((x, z), radialVelocity/1000, (xtest, ztest))
# RadialGrid[mask] = np.nan
#
# AlfvenPointGrid = griddata((x, z), alfvenPointCheck, (xtest, ztest))
# AlfvenPointGrid[mask] = np.nan


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
# heatmap = plt.contourf(xtest, ztest, BGrid, cmap=plt.cm.get_cmap('gist_rainbow'), alpha=0.4)
# lines = plt.contour(xtest, ztest, BGrid, 5, colors='k')
# plt.clabel(lines, fontsize=18, inline=1, colors='k')
# clb = plt.colorbar(heatmap)
# clb.ax.set_title('B$_n$ (nT)', fontsize=18)
# plt.rcParams['xtick.labelsize'] = 18
# plt.rcParams['ytick.labelsize'] = 18
# plt.xlabel('x $(R_J)$', fontsize=18)
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

plt.figure()
heatmap = plt.contourf(xtest, ztest, NGrid, cmap=plt.cm.get_cmap('gist_rainbow'), locator=ticker.LogLocator(), alpha=0.4)
lines = plt.contour(xtest, ztest, NGrid, 5, colors='k')
plt.clabel(lines, inline=1, colors='k')
clb = plt.colorbar(heatmap)
#clb.ax.set_title(r'(kgm$^{-1}$)', fontsize=18)
plt.xlabel('x $(R_J)$', fontsize=18)
plt.ylabel('z $(R_J)$', fontsize=18)
plt.xticks(size=18)
plt.yticks(size=18)
plt.xlim(minR)
plt.tight_layout()

# plt.figure()
# plt.rcParams['xtick.labelsize'] = 18
# plt.rcParams['ytick.labelsize'] = 18
# plt.subplots_adjust(wspace=0.5, hspace=0.5)
# plt.tight_layout()
#
# ax = plt.subplot(221)
# heatmap = plt.contourf(xtest, ztest, AlfvenGrid, cmap=plt.cm.get_cmap('gist_rainbow'), alpha=0.4)
# lines = plt.contour(xtest, ztest, AlfvenGrid, 1, colors='k')
# clb = plt.colorbar(heatmap)
# clb.ax.set_title(r'(kms$^{-1}$)', fontsize=18)
# plt.title('Alfven V', fontsize=18, wrap=True)
# plt.xlabel('x $(R_J)$', fontsize=18)
# plt.ylabel('z $(R_J)$', fontsize=18)
# plt.xticks(size=18)
# plt.yticks(size=18)
# plt.xlim(minR)
#
# ax = plt.subplot(222)
# heatmap = plt.contourf(xtest, ztest, RadialGrid, cmap=plt.cm.get_cmap('gist_rainbow'), alpha=0.4)
# lines = plt.contour(xtest, ztest, RadialGrid, 5, colors='k')
# plt.clabel(lines, inline=1, colors='k')
# clb = plt.colorbar(heatmap)
# clb.ax.set_title(r'(kms$^{-1}$)', fontsize=18)
# plt.title('Radial V', fontsize=18, wrap=True)
# plt.xlabel('x $(R_J)$', fontsize=18)
# plt.ylabel('z $(R_J)$', fontsize=18)
# plt.xticks(size=18)
# plt.yticks(size=18)
# plt.xlim(minR)
#
# ax = plt.subplot(212)
# lines = plt.contour(xtest, ztest, AlfvenPointGrid, 1)
# plt.title('Alfven Radius', fontsize=18, wrap=True)
# plt.xlabel('x $(R_J)$', fontsize=18)
# plt.ylabel('z $(R_J)$', fontsize=18)
# plt.xticks(size=18)
# plt.yticks(size=18)
# plt.xlim(minR)
#
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
