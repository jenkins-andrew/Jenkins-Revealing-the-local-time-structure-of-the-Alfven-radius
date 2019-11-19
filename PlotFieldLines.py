from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt


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


r, theta, phi, Br, Bt, Bp = np.loadtxt('magfieldvalues.txt', delimiter='\t', unpack=True)

maxR = 10
minR = 1
# xtest = np.arange(-maxR, maxR+1, 0.5)
# ytest = xtest
# ztest = xtest
# xtest, ytest, ztest = np.meshgrid(xtest, ytest, ztest)
x, y, z = sph_cart(r, theta, phi)

mask = (np.sqrt(x**2 + y**2 + z**2) < 2)

B = np.sqrt(Br**2 + Bt**2 + Bp**2)

x, y, z, B = np.loadtxt('test.txt', delimiter='\t', unpack=True)

fig = plt.figure()
ax = plt.axes(projection="3d")
plt.plot(x, y, z, '-k')

u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
xs = np.cos(u)*np.sin(v)
ys = np.sin(u)*np.sin(v)
zs = np.cos(v)
ax.plot_surface(xs, ys, zs, color="k")

minimumAxis = min([min(x), min(y), min(z), np.amin(xs), np.amin(ys), np.amin(zs)])
maximumAxis = max([max(x), max(y), max(z)])

max = 0.5*maximumAxis

ax.set_xlim3d(minimumAxis, maximumAxis)
ax.set_ylim3d(-max, max)
ax.set_zlim3d(minimumAxis, maximumAxis)
ax.set_xlabel('x RJ')
ax.set_ylabel('y RJ')
ax.set_zlabel('z RJ')

# img = ax.scatter(x[mask], y[mask], z[mask], c=B[mask], cmap=plt.cm.get_cmap('gist_rainbow'))
# clb = fig.colorbar(img)
# clb.ax.set_title(r'B (nT)', fontsize=18)
# ax.set_xlabel('x RJ')
# ax.set_ylabel('y RJ')
# ax.set_zlabel('z RJ')
plt.show()

