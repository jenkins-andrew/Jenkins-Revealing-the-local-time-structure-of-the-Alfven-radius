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
    v_hat = vector / np.sqrt((vector**2).sum())
    return v_hat[0], v_hat[1], v_hat[2]


# r, theta, phi, Br, Btheta, Bphi = np.loadtxt('test.txt', delimiter='\t', unpack=True)

x, y, z = [], [], []
rInRJ = []
thetaInRadians = []
phiInRadians = []
Bradius, Btheta, Bphi = [], [], []

fieldGenerator = field_models()

Br0, Bt0, Bp0 = fieldGenerator.Internal_Field(10, 0.5*np.pi, 0, 'JRM09')#





