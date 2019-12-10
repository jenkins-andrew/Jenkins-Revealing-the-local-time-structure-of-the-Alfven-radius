from mpl_toolkits import mplot3d
import numpy as np
import glob

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


for field_trace_path in glob.glob('output*.txt'):
    alfvenPointCheck = []
    x, y, z, B, rho = np.loadtxt(field_trace_path, delimiter='\t', unpack=True)

    r = np.sqrt(x**2 + y**2 + z**2)
    alfvenVelocity = alfvenVelocityFuncForArray(B, rho)
    radialVelocity = radialVelocityFuncForArray(r, rho)

    # for i in range(len(alfvenVelocity)):
    #     if alfvenVelocity[i] > radialVelocity[i]:
    #         alfvenPointCheck.append(0)
    #     else:
    #         alfvenPointCheck.append(1)

    np.savetxt('temporaryFile.txt', np.c_[x, y, z, B, rho, alfvenVelocity, radialVelocity], delimiter='\t')

