from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm


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


x, y, z, B = np.loadtxt('plotmagfieldlines.txt', delimiter='\t', unpack=True)

fig = plt.figure()
ax = plt.axes(projection="3d")
plt.plot(x, y, z, '-k')

# img = ax.scatter(x, y, z, c=B, cmap=plt.cm.get_cmap('gist_rainbow'))
# clb = fig.colorbar(img)
# clb.ax.set_title(r'B (nT)', fontsize=18)

u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
xs = np.cos(u)*np.sin(v)
ys = np.sin(u)*np.sin(v)
zs = np.cos(v)
ax.plot_surface(xs, ys, zs, color="r")

minimumAxis = min([min(x), min(y), min(z), np.amin(xs), np.amin(ys), np.amin(zs)])
maximumAxis = max([max(x), max(y), max(z)])

maxz = max(z)

max = 0.5*maximumAxis

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

