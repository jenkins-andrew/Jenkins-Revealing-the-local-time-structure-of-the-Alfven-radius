import numpy as np


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


