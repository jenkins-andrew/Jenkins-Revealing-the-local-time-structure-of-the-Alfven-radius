import numpy as np
from Magnetic_field_models import field_models
import statistics as stats


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


def cart_sph(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arctan2(np.sqrt(x ** 2 + y ** 2), z)
    phi = np.arctan2(y, x)
    return r, theta, phi


def find_nearest(array, value):
    """
    From unutbu on stack overflow
    :param array:
    :param value:
    :return:
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def unitVector(x0, x1, x2):
    vector = [x0, x1, x2]
    v_hat = vector / np.sqrt((np.square(vector)).sum())
    return v_hat[0], v_hat[1], v_hat[2]


def magntiudeVector(x0, x1, x2):
    vector = [x0, x1, x2]
    return np.sqrt((np.square(vector)).sum())


xInRJ, yInRJ, zInRJ, Bmag = [], [], [], []
printTester = 0

fieldGenerator = field_models()

for phi0 in np.arange(0, 2*np.pi, 0.25*np.pi):
    for r0 in np.arange(2, 30):
        theta = 0.5*np.pi
        r = r0
        phi = phi0
        step = 1000
        x, y, z = sph_cart(r, theta, phi)
        tempBmag = []
        print('Radius = %5.2f and Phi = %5.2f started' % (r, phi*180/np.pi))
        while r >= 1:
            Br, Bt, Bp = fieldGenerator.Internal_Field(r, theta, phi, 'simple')
            tempBmag.append(magntiudeVector(Br, Bt, Bp))
            if printTester % 10 == 0:
                xInRJ.append(x)
                yInRJ.append(y)
                zInRJ.append(z)
                Bmag.append(magntiudeVector(Br, Bt, Bp))
            xMove, yMove, zMove = unitVector(Br, Bt, Bp)
            step = magntiudeVector(Br, Bt, Bp)/stats.median(tempBmag) * 100
            r += -xMove / step
            theta += -yMove / step
            phi += -zMove / step
            x, y, z = sph_cart(r, theta, phi)
            printTester += 1


# theta = 0.5*np.pi
# r = 10
# phi = np.pi
# step = 1000
# x, y, z = sph_cart(r, theta, phi)
# print('theta= %5.2f and Phi = %5.2f' %(theta*180/np.pi, phi))
# while r > 1:
#     Br, Bt, Bp = fieldGenerator.Internal_Field(r, theta, phi, 'VIP4')
#     if printTester % 10 == 0:
#         xInRJ.append(x)
#         yInRJ.append(y)
#         zInRJ.append(z)
#         Bmag.append(magntiudeVector(Br, Bt, Bp))
#         print(r)
#     xMove, yMove, zMove = unitVector(Br, Bt, Bp)
#     step = magntiudeVector(Br, Bt, Bp)/stats.median(Bmag) * 100
#     r += -xMove / step
#     theta += -yMove / step
#     phi += -zMove / step
#     x, y, z = sph_cart(r, theta, phi)
#     printTester += 1

np.savetxt('plotmagfieldlines.txt', np.c_[xInRJ, yInRJ, zInRJ, Bmag], delimiter='\t', header='x\ty\tz\tB')

xInRJ = np.array(xInRJ)
yInRJ = np.array(yInRJ)
zInRJ = np.array(zInRJ)
radius = np.sqrt(xInRJ**2 + yInRJ**2 + zInRJ**2)
maxRadialDistance = max(radius)
print(maxRadialDistance)