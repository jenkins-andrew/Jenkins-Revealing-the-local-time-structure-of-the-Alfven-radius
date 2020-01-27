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


def equatorialPlasmaNumberDensity(r, speciesValues=None):
    """
    Calculates the plasma density at the equator using the method from Bagenal 2011
    :param r: The radius in R_J
    :param speciesValues:
    :return: The plasma number density at the equator in cm^-3
    """
    b2011 = []
    n = []
    try:
        percentage, a, b, c = speciesValues
        for i in range(len(r)):
            b2011.append(1987 * (r[i] / 6) ** (-8.2) + 14 * (r[i] / 6) ** (-3.2) + 0.05 * (r[i] / 6) ** (-0.65))
            if r[i] <= 15.2:
                n.append(a * (r[i] / 6) ** b)
            else:
                n.append(c * b2011[i])
    except:
        print("Exception found")
        n = 1987 * (r / 6) ** (-8.2) + 14 * (r / 6) ** (-3.2) + 0.05 * (r / 6) ** (-0.65)
    n = np.array(n)
    return n


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


def totalMassDensity(r, species, massAmuArray):
    """
    Total mass density of the plasma in kg/m^3
    :param r: radius in R_J
    :param species: List of species with parameters needed for equatorialTotalPlasmaNumberDensity()
    :param massAmuArray: List of species with masses in amu
    :return: Mass Density in kg/m^3
    """
    M = np.zeros(len(r))
    for i in massAmuArray:
        mass = massAmuArray[i]
        try:
            n = equatorialPlasmaNumberDensity(r, species[i])
        except:
            n = np.zeros(len(r))
            print('Species do not match')
        M += n*1e6 * mass*1.67e-27

    return M


def massDensityAtZFromEquator(r, z, species, massArray):
    """
    Mass density at height z from the equator
    :param z: in RJ
    :param r: in RJ
    :param species: List of species with parameters needed for equatorialTotalPlasmaNumberDensity()
    :param massArray: List of species with masses in amu
    :return: mass in kg/m^3
    """
    equatorialDensity = totalMassDensity(r, species, massArray)

    mZ = equatorialDensity * np.exp(-1 * (z / radialScaleHeight(r)) ** 2)

    for i in range(len(mZ)):
        if mZ[i] < 1.67e-23:
            mZ[i] = 1.67e-23
    return mZ


def radialVelocityFuncForArray(r, species, massArray):

    vr = 1000/(2 * totalMassDensity(r, species, massArray) * radialScaleHeight(r) * np.pi * r * 71492e3 ** 2)
    return vr


def generateAlfvenAndRadial(path):
    # for field_trace_path in glob.glob('output*.txt'):
    #     alfvenPointCheck = []

    loaded = np.load(path, allow_pickle=True)
    output = []
    for i in range(len(loaded)):
        np.savetxt('temp.txt', np.c_[loaded[i]], delimiter='\t')

        x, y, z, B = np.loadtxt('temp.txt', delimiter='\t', unpack=True)

        r, theta, phi = cart_sph(x, y, z)

        equatorialdistance = np.sqrt(x**2+y**2)

        rho = massDensityAtZFromEquator(equatorialdistance, z, speciesList, speciesMass)

        alfvenVelocity = alfvenVelocityFuncForArray(B, rho)
        radialVelocity = radialVelocityFuncForArray(r, speciesList, speciesMass)

        output.append(np.c_[x, y, z, B, rho, alfvenVelocity, radialVelocity])

    np.save(path, output)


def generateAlfvenAndRadialFromDefusive(path):
    # for field_trace_path in glob.glob('output*.txt'):
    #     alfvenPointCheck = []
    x, y, z, B, rho = np.loadtxt(path, delimiter='\t', unpack=True)

    r, theta, phi = cart_sph(x, y, z)

    alfvenVelocity = alfvenVelocityFuncForArray(B, rho)
    radialVelocity = radialVelocityFuncForArray(r, speciesList, speciesMass)
    # for i in range(len(alfvenVelocity)):
    #     if alfvenVelocity[i] > radialVelocity[i]:
    #         alfvenPointCheck.append(0)
    #     else:
    #         alfvenPointCheck.append(1)

    np.savetxt(path, np.c_[x, y, z, B, rho, alfvenVelocity, radialVelocity], delimiter='\t')


def generateAlfvenTravelTimes(path):
    loaded = np.load(path, allow_pickle=True)
    lineNumber = []
    travelTime = []
    start = float(path[16:20])
    end = float(path[22:27])
    phi = float(path[30:34])
    fieldLineStep = int((end - start) / len(loaded) + 1)

    for i in range(len(loaded)):
        np.savetxt('temp.txt', np.c_[loaded[i]], delimiter='\t')

        lineNumber.append(i*fieldLineStep + start)

        x, y, z, B, rho, alfven, radial = np.loadtxt('temp.txt', delimiter='\t', unpack=True)
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        time = 0
        for j in range(len(r)-1):
            deltaR = r[j+1] - r[j]
            time += (1/alfven[j+1]) * deltaR
        travelTime.append(time)

    np.savetxt('alfvenTravelTimesfor%0.2fto%0.2fatPhi%0.2f.txt' % (start, end, phi), np.c_[lineNumber, travelTime],
               delimiter='\t')
