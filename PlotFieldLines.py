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

plt.plot(x, y, z, '+-k')
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

