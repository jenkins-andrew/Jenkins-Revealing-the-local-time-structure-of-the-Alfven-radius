import numpy as np
from Magnetic_field_models import field_models


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
    """
    From Chris' code
    :param x:
    :param y:
    :param z:
    :return:
    """
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arctan2(np.sqrt(x ** 2 + y ** 2), z)
    phi = np.arctan2(y, x)
    return r, theta, phi


def unitVector(x0, x1, x2):
    """

    :param x0:
    :param x1:
    :param x2:
    :return:
    """
    vector = [x0, x1, x2]
    v_hat = vector / np.sqrt((np.square(vector)).sum())
    return v_hat[0], v_hat[1], v_hat[2]


def magnitudeVector(x0, x1, x2):
    """

    :param x0:
    :param x1:
    :param x2:
    :return:
    """
    vector = [x0, x1, x2]
    return np.sqrt((np.square(vector)).sum())


def produceTraceArrays(modelType='VIP4'):
    """

    :param modelType:
    :return:
    """
    printTester = 0
    fieldGenerator = field_models()
    signArray = [-1, 1]

    for phi0 in np.arange(0, 0 + 0.001, 0.25 * np.pi):
        for r0 in np.arange(50, 100, 2):
            xInRJ, yInRJ, zInRJ, Bmag = [], [], [], []
            for sign in signArray:
                theta = 0.5 * np.pi
                r = r0
                phi = phi0
                tempxInRJ, tempyInRJ, tempzInRJ, tempBmag = [], [], [], []
                x, y, z = sph_cart(r, theta, phi)
                print('Radius = %5.2f and Phi = %1.2f started going %1.0f' % (r, phi * 180 / np.pi, sign))
                while r >= 1:
                    x, y, z = sph_cart(r, theta, phi)
                    Br, Bt, Bp = fieldGenerator.Internal_Field(r, theta, phi, modelType)
                    if printTester % 1 == 0:
                        tempxInRJ.append(x)
                        tempyInRJ.append(y)
                        tempzInRJ.append(z)
                        tempBmag.append(magnitudeVector(Br, Bt, Bp))
                    xMove, yMove, zMove = unitVector(Br, Bt, Bp)
                    step = np.log10(magnitudeVector(Br, Bt, Bp)) * 10
                    r += sign * xMove / step
                    theta += sign * yMove / step
                    phi += sign * zMove / step
                    printTester += 1
                tempxInRJ = tempxInRJ[::sign]
                tempyInRJ = tempyInRJ[::sign]
                tempzInRJ = tempzInRJ[::sign]
                tempBmag = tempBmag[::sign]
                xInRJ.extend(tempxInRJ)
                yInRJ.extend(tempyInRJ)
                zInRJ.extend(tempzInRJ)
                Bmag.extend(tempBmag)
            np.savetxt('output/radius%0.0fphi%0.0f.txt' % (r0, phi0), np.c_[xInRJ, yInRJ, zInRJ, Bmag], delimiter=',')
    return


produceTraceArrays()

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


