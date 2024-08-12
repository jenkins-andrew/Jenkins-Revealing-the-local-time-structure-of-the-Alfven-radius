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
xAlfven, yAlfven = np.loadtxt('alfvenRadiusPoints.txt', delimiter='\t', unpack=True)
# r, scaleHeight = np.loadtxt('scaleheighttest.txt', delimiter='\t', unpack=True)
#
# radius, alfven, radial = np.loadtxt('alfvenradial.txt', delimiter='\t', unpack=True)

maskedAlfven = np.sqrt(xAlfven ** 2 + yAlfven ** 2) < 60


# Creating grid
maxR = 60
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


plt.figure()
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.subplots_adjust(wspace=0.5, hspace=0.5)
# plt.tight_layout()
ax1 = plt.subplot(221)
heatmap = plt.contourf(-xtest, -ytest, AVGrid, cmap=plt.cm.get_cmap('gist_rainbow'), levels=100, alpha=0.4, antialiased=True)
# lines = plt.contour(-xtest, -ytest, CheckGrid, 1, colors='k', linewidths=2)
plt.plot(-xAlfven[maskedAlfven], -yAlfven[maskedAlfven], 'k', linewidth=3, solid_capstyle='round')
Jupiter = plt.Circle((0, 0), radius=1, color='k')
ax1.add_artist(Jupiter)
clb = plt.colorbar(heatmap, ticks=range(0, 201, 25), drawedges=False)
clb.ax.set_title(r'(kms$^{-1}$)', fontsize=18)
plt.title('Alfvén Vel.', fontsize=18, wrap=True)
plt.xlabel('x $(R_J)$', fontsize=18)
plt.ylabel('y $(R_J)$', fontsize=18)
plt.xticks(size=18)
plt.yticks(size=18)
plt.text(-100, 70, 'a)', size=18, weight='bold')
ax1.tick_params(axis='both', which='major', size=6, direction='in', top='true', right='true')
ax1.tick_params(axis='both', which='minor', size=4, direction='in', top='true', right='true')
ax1.xaxis.set_major_locator(tick.MultipleLocator(30))
ax1.xaxis.set_minor_locator(tick.MultipleLocator(10))
ax1.yaxis.set_major_locator(tick.MultipleLocator(30))
ax1.yaxis.set_minor_locator(tick.MultipleLocator(10))
ax1.axis('equal')

ax = plt.subplot(222)
heatmap = plt.contourf(xtest, ytest, VGrid, cmap=plt.cm.get_cmap('gist_rainbow'), levels=100, alpha=0.4, antialiased=True)
# lines = plt.contour(xtest, ytest, VGrid, 5, colors='k')
# plt.clabel(lines, inline=1, colors='k')
Jupiter = plt.Circle((0, 0), radius=1, color='k')
ax.add_artist(Jupiter)
clb = plt.colorbar(heatmap, ticks=range(0, 91, 10), drawedges=False)
clb.ax.set_title(r'(kms$^{-1}$)', fontsize=18)
plt.title('Radial Vel.', fontsize=18, wrap=True)
plt.xlabel('x $(R_J)$', fontsize=18)
plt.ylabel('y $(R_J)$', fontsize=18)
plt.xticks(size=18)
plt.yticks(size=18)
plt.text(-100, 70, 'b)', size=18, weight='bold')
ax.tick_params(axis='both', which='major', size=6, direction='in', top='true', right='true')
ax.tick_params(axis='both', which='minor', size=4, direction='in', top='true', right='true')
ax.xaxis.set_major_locator(tick.MultipleLocator(30))
ax.xaxis.set_minor_locator(tick.MultipleLocator(10))
ax.yaxis.set_major_locator(tick.MultipleLocator(30))
ax.yaxis.set_minor_locator(tick.MultipleLocator(10))
ax.axis('equal')

ax = plt.subplot(223)
# lines = plt.contour(-xtest, -ytest, CheckGrid, 1, colors='k', linewidths=2)
Jupiter = plt.Circle((0, 0), radius=1, color='k')
plt.plot(-xAlfven[maskedAlfven], -yAlfven[maskedAlfven], 'k', linewidth=3, solid_capstyle='round')
ax.add_artist(Jupiter)
plt.title('Alfvén Radius', fontsize=18, wrap=True)
plt.xlabel('x $(R_J)$', fontsize=18)
plt.ylabel('y $(R_J)$', fontsize=18)
plt.xticks(size=18)
plt.yticks(size=18)
ax.tick_params(axis='both', which='major', size=6, direction='in', top='true', right='true')
ax.tick_params(axis='both', which='minor', size=4, direction='in', top='true', right='true')
ax.axis('equal')
cb = plt.colorbar()
cb.remove()
ax = plt.gca()
ax.set(ylim=(-60, 60))
plt.text(-100, 70, 'c)', size=18, weight='bold')
ax.xaxis.set_major_locator(tick.MultipleLocator(30))
ax.xaxis.set_minor_locator(tick.MultipleLocator(10))
ax.yaxis.set_major_locator(tick.MultipleLocator(30))
ax.yaxis.set_minor_locator(tick.MultipleLocator(10))
plt.text(-18, 13, 'Inside', size=16)
plt.text(-40, -40, 'Outside', size=16)

ax = plt.subplot(224)
calculatedRadius = np.sqrt(xAlfven ** 2 + yAlfven ** 2)
phiwrong = np.arctan2(yAlfven, xAlfven)
LTHours = phiwrong * 180 / np.pi
LTHours = LTHours * (24/360)
for i in range(len(LTHours)):
    if LTHours[i] <0:
        LTHours[i]=LTHours[i]+24
temp = zip(LTHours, calculatedRadius)
sortlist = sorted(temp)
LTHours, calculatedRadius = zip(*sortlist)
plt.plot(LTHours, calculatedRadius, 'k', linewidth=3, solid_capstyle='round')
y = [r for r in range(28, 70)]
dawn = [6] * len(y)
noon = [12] * len(y)
dusk = [18] * len(y)
plt.plot(dawn, y, linestyle='-', color='grey', alpha=0.4)
plt.plot(noon, y, linestyle='-', color='grey', alpha=0.4)
plt.plot(dusk, y, linestyle='-', color='grey', alpha=0.4)
plt.title('Alfvén Radius', fontsize=18, wrap=True)
plt.xlabel('Local Time (Hours)', fontsize=18)
plt.ylabel('Radius (R$_J)$', fontsize=18)
plt.text(-6, 63, 'd)', size=18, weight='bold')
ax.tick_params(axis='both', which='major', size=6, direction='in', top='true', right='true')
ax.tick_params(axis='both', which='minor', size=4, direction='in', top='true', right='true')
ax.xaxis.set_major_locator(tick.MultipleLocator(6))
ax.xaxis.set_minor_locator(tick.MultipleLocator(2))
ax.yaxis.set_major_locator(tick.MultipleLocator(10))
ax.yaxis.set_minor_locator(tick.MultipleLocator(5))
plt.xticks(size=18)
plt.yticks(size=18)
plt.xlim(0, 24)
plt.ylim(28, 60)
# plt.savefig('4panelAlfven.png', transparent=True)

# Alfven Radius transparent plot
# fig, ax = plt.subplots()
# Jupiter = plt.Circle((0, 0), radius=1, color='k')
# plt.plot(-xAlfven[maskedAlfven], -yAlfven[maskedAlfven], color='yellow', linewidth=10, solid_capstyle='round')
# outerCircle = plt.Circle((0, 0), radius=60, color='k', fill=False)
# ax.add_artist(Jupiter)
# ax.add_artist(outerCircle)
# fig.patch.set_visible(False)
# ax.axis('off')
# plt.xlim(-60, 60)
# plt.ylim(-60, 60)
# plt.tight_layout()
# plt.savefig('tester.png', transparent=True)

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

#
plt.show()