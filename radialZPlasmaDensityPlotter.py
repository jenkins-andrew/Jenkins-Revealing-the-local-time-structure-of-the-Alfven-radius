import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from scipy.interpolate import griddata


r, z, plasmaDensity, radialVelocityAtZ, alfvenVelocityAtZ = np.loadtxt('zPlasmaDensity.txt', delimiter='\t', unpack=True)
xB, yB, zB, BB = np.loadtxt('radius6.00to100.00phi0.00CurrentOn=False.txt', delimiter='\t', unpack=True)


maxR = 100
xtest = np.arange(6, maxR+1, 0.5)
ytest = np.arange(np.amin(z), np.amax(z), 0.1)
xtest, ytest = np.meshgrid(xtest, ytest)

# theta = -10*np.pi/180
# xB = xB*np.cos(theta) - zB*np.sin(theta)
# zB = xB*np.sin(theta) + zB*np.cos(theta)

mask = (plasmaDensity > 10**(-1))
maskedFieldLines = (np.abs(xB) > 6)
maskedRadialVelocity = (radialVelocityAtZ < 10**(8))

corotationMask = (radialVelocityAtZ < alfvenVelocityAtZ)

DensityGrid = griddata((r[corotationMask], z[corotationMask]), plasmaDensity[corotationMask], (xtest, ytest))
FieldGrid = griddata((xB, zB), BB, (xtest, ytest))
radialGrid = griddata((r[maskedRadialVelocity], z[maskedRadialVelocity]), radialVelocityAtZ[maskedRadialVelocity], (xtest, ytest))
alfvenGrid = griddata((r, z), alfvenVelocityAtZ, (xtest, ytest))



plt.figure()
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
heatmap = plt.contourf(xtest, ytest, DensityGrid, cmap=plt.cm.get_cmap('gist_rainbow'), locator=ticker.LogLocator(), alpha=0.4)
plt.plot(xB, zB, '--k')
# lines = plt.contour(xtest, ytest, FieldGrid, 10, colors='k')
# plt.clabel(lines, fontsize=18, inline=1, colors='k')
clb = plt.colorbar(heatmap)
clb.ax.set_title(r'(cm$^{-3}$)', fontsize=18)
plt.xlabel('R $(R_J)$', fontsize=18)
plt.ylabel('Z $(R_J)$', fontsize=18)
plt.xticks(size=18)
plt.yticks(size=18)
plt.ylim(np.amin(zB), np.amax(zB))

plt.tight_layout()

plt.figure()
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.tight_layout()

ax = plt.subplot(211)
heatmap = plt.contourf(xtest, ytest, alfvenGrid, cmap=plt.cm.get_cmap('gist_rainbow'), locator=ticker.LogLocator(), alpha=0.4)
lines = plt.contour(xtest, ytest, alfvenGrid, 10, colors='k')
plt.clabel(lines, fontsize=18, inline=1, colors='k')
clb = plt.colorbar(heatmap)
clb.ax.set_title(r'(ms$^{-1}$)', fontsize=18)
plt.title('Alfven Velocity', fontsize=18)
plt.xlabel('R $(R_J)$', fontsize=18)
plt.ylabel('Z $(R_J)$', fontsize=18)
plt.xticks(size=18)
plt.yticks(size=18)
#plt.ylim(-4, 4)

ax = plt.subplot(212)
heatmap = plt.contourf(xtest, ytest, radialGrid, cmap=plt.cm.get_cmap('gist_rainbow'), locator=ticker.LogLocator(), alpha=0.4)
lines = plt.contour(xtest, ytest, radialGrid, 10, colors='k')
plt.clabel(lines, fontsize=18, inline=1, colors='k')
clb = plt.colorbar(heatmap)
clb.ax.set_title(r'(ms$^{-1}$)', fontsize=18)
plt.title('Radial Velocity', fontsize=18)
plt.xlabel('R $(R_J)$', fontsize=18)
plt.ylabel('Z $(R_J)$', fontsize=18)
plt.xticks(size=18)
plt.yticks(size=18)
#plt.ylim(-4, 4)

plt.show()