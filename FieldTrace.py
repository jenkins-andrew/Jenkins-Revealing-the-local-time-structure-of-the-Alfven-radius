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


# r, theta, phi, Br, Btheta, Bphi = np.loadtxt('test.txt', delimiter='\t', unpack=True)

xInRJ, yInRJ, zInRJ = [], [], []
Bmag = []

fieldGenerator = field_models()

for phi0 in np.arange(0, np.pi):
    for theta0 in np.arange(0, np.pi, 0.1):
        theta = theta0
        r = 1.0
        phi = phi0
        x, y, z = sph_cart(r, theta, phi)
        print(r'$\theta$' '= %5.2f and Phi = %5.2f' %(theta, phi))
        while r >= 1:
            xInRJ.append(x)
            yInRJ.append(y)
            zInRJ.append(z)
            Br, Bt, Bp = fieldGenerator.Internal_Field(r, theta, phi, 'simple')
            Bmag.append(magntiudeVector(Br, Bt, Bp))
            xMove, yMove, zMove = unitVector(Br, Bt, Bp)
            r += xMove / 1000
            theta += yMove / 1000
            phi += zMove / 1000
            x, y, z = sph_cart(r, theta, phi)




np.savetxt('test.txt', np.c_[xInRJ, yInRJ, zInRJ,
                             Bmag], delimiter='\t', header='x\ty\tz\tB')

