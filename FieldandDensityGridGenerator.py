from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.interpolate import griddata

x, y, z, B, N = [], [], [], [], []

for field_trace_path in glob.glob('output/postFieldLine/*.txt'):
    print(field_trace_path)
    x0, y0, z0, B0, N0 = np.loadtxt(field_trace_path, delimiter='\t', unpack=True)
    x.extend(x0)
    y.extend(y0)
    z.extend(z0)
    B.extend(B0)
    N.extend(N0)

np.savetxt('temporaryFile.txt', np.c_[x, y, z, B, N], delimiter='\t')
x, y, z, B, N = np.loadtxt('temporaryFile.txt', delimiter='\t', unpack=True)

maxR = 30
minR = 6
xtest = np.arange(-maxR, maxR+1, 0.5)
ztest = xtest
xtest, ztest = np.meshgrid(xtest, ztest)

# Masking a circle of radius minR R_J
mask = (xtest < minR) | (np.sqrt(xtest ** 2 + ztest ** 2) > maxR)

# Making the 3D grid for the magnetic field
BGrid = griddata((x, z), B, (xtest, ztest))
BGrid[mask] = np.nan

NGrid = griddata((x, z), N*1e19, (xtest, ztest))
NGrid[mask] = np.nan


plt.figure()
heatmap = plt.contourf(xtest, ztest, BGrid, cmap=plt.cm.get_cmap('gist_rainbow'), alpha=0.4)
lines = plt.contour(xtest, ztest, BGrid, 5, colors='k')
plt.clabel(lines, fontsize=18, inline=1, colors='k')
clb = plt.colorbar(heatmap)
clb.ax.set_title('B$_n$ (nT)', fontsize=18)
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.xlabel('x $(R_J)$', fontsize=18)
plt.ylabel('z $(R_J)$', fontsize=18)
plt.xticks(size=18)
plt.yticks(size=18)
plt.xlim(minR)
plt.tight_layout()

plt.figure()
heatmap = plt.contourf(xtest, ztest, NGrid, cmap=plt.cm.get_cmap('gist_rainbow'), alpha=0.4)
lines = plt.contour(xtest, ztest, NGrid, 5, colors='k')
plt.clabel(lines, fontsize=18, inline=1, colors='k')
clb = plt.colorbar(heatmap)
clb.ax.set_title(r'$\rho$ 1e-19 kgm$^{-3}$', fontsize=18)
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.xlabel('x $(R_J)$', fontsize=18)
plt.ylabel('z $(R_J)$', fontsize=18)
plt.xlim(minR)
plt.xticks(size=18)
plt.yticks(size=18)
plt.tight_layout()

# x, y, z, B, N = np.loadtxt('output/postFieldLine/radius6theta0.txt', delimiter='\t', unpack=True)
# img = ax.scatter(x, y, z, c=B, cmap=plt.cm.get_cmap('gist_rainbow'))



# max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
#
# mid_x = (x.max()+x.min()) * 0.5
# mid_y = (y.max()+y.min()) * 0.5
# mid_z = (z.max()+z.min()) * 0.5
# ax.set_xlim(mid_x - max_range, mid_x + max_range)
# ax.set_ylim(mid_y - max_range, mid_y + max_range)
# ax.set_zlim(mid_z - max_range, mid_z + max_range)
plt.show()