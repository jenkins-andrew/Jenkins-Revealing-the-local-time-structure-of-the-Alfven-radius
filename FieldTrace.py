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

def produceTraceArrays():

    return

xInRJ, yInRJ, zInRJ, Bmag = [], [], [], []
printTester = 0

fieldGenerator = field_models()
signArray = [-1, 1]

for phi0 in np.arange(np.pi, np.pi+0.001, 0.25*np.pi):
    for r0 in np.arange(6, 30, 2):
        for sign in signArray:
            theta = 0.5*np.pi
            r = r0
            phi = phi0
            step = 1000
            tempxInRJ, tempyInRJ, tempzInRJ = [], [], []
            x, y, z = sph_cart(r, theta, phi)
            print('Radius = %5.2f and Phi = %5.2f started' % (r, phi*180/np.pi))
            while r >= 1:
                x, y, z = sph_cart(r, theta, phi)
                Br, Bt, Bp = fieldGenerator.Internal_Field(r, theta, phi, 'simple')
                if printTester % 1 == 0:
                    tempxInRJ.append(x)
                    tempyInRJ.append(y)
                    tempzInRJ.append(z)
                    Bmag.append(magntiudeVector(Br, Bt, Bp))
                xMove, yMove, zMove = unitVector(Br, Bt, Bp)
                step = np.log10(magntiudeVector(Br, Bt, Bp)) * 10
                r += sign*xMove / step
                theta += sign*yMove / step
                phi += sign*zMove / step
                printTester += 1
            tempxInRJ = tempxInRJ[::sign]
            tempyInRJ = tempyInRJ[::sign]
            tempzInRJ = tempzInRJ[::sign]
            xInRJ.extend(tempxInRJ)
            yInRJ.extend(tempyInRJ)
            zInRJ.extend(tempzInRJ)

# theta = 0.5*np.pi
# r = 30
# phi = np.pi
# x, y, z = sph_cart(r, theta, phi)
# print('theta= %5.2f and Phi = %5.2f' %(theta*180/np.pi, phi))
# while r > 1:
#     Br, Bt, Bp = fieldGenerator.Internal_Field(r, theta, phi, 'VIP4')
#     if printTester % 1 == 0:
#         xInRJ.append(x)
#         yInRJ.append(y)
#         zInRJ.append(z)
#         Bmag.append(magntiudeVector(Br, Bt, Bp))
#         print(r)
#     xMove, yMove, zMove = unitVector(Br, Bt, Bp)
#     step = np.log10(magntiudeVector(Br, Bt, Bp)) * 10
#     r += -xMove / step
#     theta += -yMove / step
#     phi += -zMove / step
#     x, y, z = sph_cart(r, theta, phi)
#     printTester += 1

np.savetxt('plotmagfieldlines.txt', np.c_[xInRJ, yInRJ, zInRJ, Bmag], delimiter='\t', header='x\ty\tz\tB')
