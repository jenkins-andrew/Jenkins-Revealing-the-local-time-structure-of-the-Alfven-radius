import numpy as np
from Magnetic_field_models import field_models
import matplotlib.pyplot as plt

radius, bcxarray, bcyarray, bczarray, bxarray, byarray, bzarray, xcombined, ycombined, zcombined, bcalculated, barray = np.loadtxt('magfieldcompoents.txt', unpack=True, delimiter='\t')

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
ax1.tick_params(right=True, which='both', labelsize=18)
ax2.tick_params(right=True, which='both', labelsize=18)
ax3.tick_params(right=True, which='both', labelsize=18)
ax1.plot(radius, bxarray)
ax1.set_ylabel('x (nT)', size=18)
ax2.plot(radius, bcxarray)
ax2.set_ylabel('cX (nT)', size=18)
ax3.plot(radius, xcombined)
ax3.set_ylabel('Combined (nT)', size=18)
ax3.set_xlabel('Radius RJ', size=18)
plt.tight_layout()


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
ax1.tick_params(right=True, which='both', labelsize=18)
ax2.tick_params(right=True, which='both', labelsize=18)
ax3.tick_params(right=True, which='both', labelsize=18)
ax1.plot(radius, byarray)
ax1.set_ylabel('y (nT)', size=18)
ax2.plot(radius, bcyarray)
ax2.set_ylabel('cY (nT)', size=18)
ax3.plot(radius, ycombined)
ax3.set_ylabel('Combined (nT)', size=18)
ax3.set_xlabel('Radius RJ', size=18)
plt.tight_layout()


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
ax1.tick_params(right=True, which='both', labelsize=18)
ax2.tick_params(right=True, which='both', labelsize=18)
ax3.tick_params(right=True, which='both', labelsize=18)
ax1.plot(radius, bzarray)
ax1.set_ylabel('z (nT)', size=18)
ax2.plot(radius, bczarray)
ax2.set_ylabel('cZ (nT)', size=18)
ax3.plot(radius, zcombined)
ax3.set_ylabel('Combined (nT)', size=18)
ax3.set_xlabel('Radius RJ', size=18)
plt.tight_layout()


plt.figure()
plt.tick_params(right=True, which='both', labelsize=18)
plt.plot(radius, np.abs(bcxarray), label='cX')
plt.plot(radius, np.abs(bcyarray), label='cY')
plt.plot(radius, np.abs(bczarray), label='cZ')
plt.xlabel('Radius RJ', size=18)
plt.ylabel('Magnitude (nT)', size=18)
plt.legend(fontsize=18)
plt.tight_layout()


plt.figure()
plt.tick_params(right=True, which='both', labelsize=18)
plt.plot(radius, barray, label='B')
plt.plot(radius, bcalculated, label='B calculated')
plt.xlabel('Radius RJ', size=18)
plt.ylabel('Magnitude (nT)', size=18)
plt.legend(fontsize=18)
plt.yscale('Log')
plt.tight_layout()
plt.show()