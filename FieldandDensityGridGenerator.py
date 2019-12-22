from mpl_toolkits import mplot3d
import numpy as np
import glob


speciesList = {'e-': [1, 2451, -6.27, 4.21],
               'o+': [0.24, 592, -7.36, 0.368],
               'o++': [0.03, 76.3, -6.73, 0.086],
               's+': [0.07, 163, -6.81, 0.169],
               's++': [0.22, 538, -6.74, 0.598],
               's+++': [0.004, 90.7, -6.21, 0.165],
               'h+': [0.02, 50.6, -5.31, 0.212],
               'na+': [0.04, 97.2, -6.75, 0.106],
               'hoto+': [0.06, 134, -4.63, 1.057]}
ME = 0.00054858
speciesMass = {'e-': 0.00054858,
               'o++': 15.999 - (ME * 2),
               's+': 32.065 - ME,
               's++': 32.065 - (ME * 2),
               's+++': 32.065 - (ME * 3),
               'h+': 1.00784 - ME,
               'na+': 22.989769 - ME,
               'hoto+': 15.999 - (ME * 2),
               'o+': 15.999 - ME
               }


def cart_sph(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arctan2(np.sqrt(x ** 2 + y ** 2), z)
    phi = np.arctan2(y, x)
    return r, theta, phi


def alfvenVelocityFuncForArray(magneticFieldArray, totalMassDensityArray):

    Va = magneticFieldArray * 1e-9 / np.sqrt(1.25663706212e-6 * totalMassDensityArray)

    return Va


def radialScaleHeight(r):
    """
    Finds the scale height at a radius
    :param r: Radius in R_J
    :return: Scale height in R_J
    """
    h = -0.116 + 2.14*np.log10(r/6) - 2.05*np.log10(r/6)**2 + 0.491*np.log10(r/6)**3 + 0.126*np.log10(r/6)**4
    H = 10 ** h
    return H


def radialVelocityFuncForArray(r, totalMassDensityArray):

    vr = 1000/(2 * totalMassDensityArray * radialScaleHeight(r) * np.pi * r * 71492e3 ** 2)
    return vr


def generateAlfvenAndRadial(path):
    # for field_trace_path in glob.glob('output*.txt'):
    #     alfvenPointCheck = []
    x, y, z, B, rho = np.loadtxt(path, delimiter='\t', unpack=True)

    r, theta, phi = cart_sph(x, y, z)
    alfvenVelocity = alfvenVelocityFuncForArray(B, rho)
    radialVelocity = radialVelocityFuncForArray(r, rho)
    # for i in range(len(alfvenVelocity)):
    #     if alfvenVelocity[i] > radialVelocity[i]:
    #         alfvenPointCheck.append(0)
    #     else:
    #         alfvenPointCheck.append(1)

    np.savetxt(path, np.c_[x, y, z, B, rho, alfvenVelocity, radialVelocity], delimiter='\t')

