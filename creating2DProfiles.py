import numpy as np
from Magnetic_field_models import field_models


def equatorialMagneticField(r, phi):
    """
    Finds the equatorial magnetic field strength using Vogt et. al 2011 method
    :param r: The radius in R_J
    :param phi: The angle in radians, 0 at the Sun, anti-clockwise
    :return: The equatorial magnetic field in nT
    """
    B = 1.030e6 * r ** (-3.756 - 0.12 * np.cos(phi - 3.562)) + \
        (3.797 - 4.612 * np.cos(phi - 0.825) + 0.606 * np.cos(2 * (phi - 0.473)) +
         0.847 * np.cos(3 * (phi - 0.913))) * np.exp((-1 * r) / 150)
    return B


def equatorialPlasmaNumberDensity(r, speciesValues=None):
    """
    Calculates the plasma density at the equator using the method from Bagenal 2011
    :param r: The radius in R_J
    :param speciesValues:
    :return: The plasma number density at the equator in cm^-3
    """
    b2011 = 1987 * (r / 6) ** (-8.2) + 14 * (r / 6) ** (-3.2) + 0.05 * (r / 6) ** (-0.65)

    try:
        percentage, a, b, c = speciesValues
        if r <= 15.2:
            n = a * (r / 6) ** b
        else:
            n = (c * b2011)
    except:
        print("Species do not match at number density step")
        n = b2011
    return n


def equatorialTotalPlasmaNumberDensity(r, species):
    """
    Calculates the plasma density at the equator using the method from Bagenal 2011
    :param r: The radius in R_J
    :param species:
    :return: The plasma number density at the equator in cm^-3
    """
    b2011 = 1987 * (r / 6) ** (-8.2) + 14 * (r / 6) ** (-3.2) + 0.05 * (r / 6) ** (-0.65)
    n = []
    for i in species:
        percentage, a, b, c = species[i]
        if r <= 15.2:
            n.append(a * (r / 6) ** b)
        else:
            n.append(c * b2011)

    totalN = np.sum(n)

    return totalN


def averageAmu(r, species, massAmuArray):
    """
    The reduced mass for the plasma
    :param r: in RJ
    :param species: List of species with parameters needed for equatorialTotalPlasmaNumberDensity()
    :param massAmuArray: List of species with masses in amu
    :return: Reduced mass in amu
    """
    sumofmasses = 0
    N = []
    for i in massAmuArray:
        mass = massAmuArray[i]
        try:
            n = equatorialPlasmaNumberDensity(r, species[i])
            N.append(n)
        except:
            n = 0
            print('Species do not match')
        sumofmasses += n * mass

    amu = sumofmasses/sum(N)
    return amu


def totalMassDensity(r, species, massAmuArray):
    """
    Total mass density of the plasma in kg/m^3
    :param r: radius in R_J
    :param species: List of species with parameters needed for equatorialTotalPlasmaNumberDensity()
    :param massAmuArray: List of species with masses in amu
    :return: Mass Density in kg/m^3
    """
    M = 0
    for i in massAmuArray:
        mass = massAmuArray[i]
        try:
            n = equatorialPlasmaNumberDensity(r, species[i])
        except:
            n = 0
            print('Species do not match')
        M += n*1e6 * mass*1.67e-27

    return M


def alfvenVelocityAtRPhi(r, phi, species, massArray):
    """
    Alfven velocity at a given r and phi
    :param r: radius in RJ
    :param phi: in radians from sun, going anticlockwise
    :param species: List of species with parameters needed for equatorialTotalPlasmaNumberDensity()
    :param massArray: List of species with masses in amu
    :return: Alfven velocity in m/s
    """

    Va = equatorialMagneticField(r, phi) * 1e-9 / np.sqrt(1.25663706212e-6 * totalMassDensity(r, species, massArray))

    return Va


def alfvenVelocityAtRThetaPhi(fieldModel, r, theta, phi, species, massArray, model='VIP4'):
    """
    Alfven velocity at a given r, theta and phi
    :param fieldModel: an object to make the field model work
    :param r: radius in RJ
    :param theta: the angle from the pole in radians
    :param phi: in radians from sun, going anticlockwise
    :param species: List of species with parameters needed for equatorialTotalPlasmaNumberDensity()
    :param massArray: List of species with masses in amu
    :param model: The type of model being used to find the magnetic field at r, theta and phi
    :return: Alfven velocity in m/s
    """

    Va = averageMagFieldModel(fieldModel, r, theta, phi, model) * 1e-9 / np.sqrt(1.25663706212e-6 * totalMassDensity(
        r, species, massArray))

    return Va


def corotationVelocityFunc(x, y):
    """
    Calculate the corotational velocity at x and y
    :param x: In R_J
    :param y: In R_J
    :return: The corotation velocity assuming 10 hr rotation period in m/s
    """
    v = (2*np.pi/(3600*9.9250)) * np.sqrt(x**2 + y**2) * 71492e3
    return v


def radialScaleHeight(r):
    """
    Finds the scale height at a radius
    :param r: Radius in R_J
    :return: Scale height in R_J
    """
    h = -0.116 + 2.14*np.log10(r/6) - 2.05*np.log10(r/6)**2 + 0.491*np.log10(r/6)**3 + 0.126*np.log10(r/6)**4
    H = 10 ** h
    return H


def densityAtZFromEquator(z, r, species):
    """
    Plasma density at height z from the equator
    :param z: in RJ
    :param r: in RJ
    :param species: List of species with parameters needed for equatorialTotalPlasmaNumberDensity()
    :return: plasma density in cm^-3
    """

    nZ = equatorialTotalPlasmaNumberDensity(r, species) * np.exp(-1 * (z / radialScaleHeight(r)) ** 2)
    return nZ


def massDensityAtZFromEquator(r, z, species, massArray):
    """
    Mass density at height z from the equator
    :param z: in RJ
    :param r: in RJ
    :param species: List of species with parameters needed for equatorialTotalPlasmaNumberDensity()
    :param massArray: List of species with masses in amu
    :return: mass in kg/m^3
    """

    mZ = totalMassDensity(r, species, massArray) * np.exp(-1 * (z / radialScaleHeight(r)) ** 2)
    return mZ


def radialVelocityFunc(r, species, massArray):
    """
    The radial velocity at distance r
    :param r: radius in RJ
    :param species: List of species with parameters needed for equatorialTotalPlasmaNumberDensity()
    :param massArray: List of species with masses in amu
    :return: radial velocity in m/s
    """
    vr = 1000/(2 * totalMassDensity(r, species, massArray) *
              radialScaleHeight(r) * np.pi * r * 71492e3 ** 2)
    return vr


def radialVelocityFuncAtZ(r, z, species, massArray):
    """
    The radial velocity at height z
    :param r: radius in RJ
    :param species: List of species with parameters needed for equatorialTotalPlasmaNumberDensity()
    :param massArray: List of species with masses in amu
    :return: radial velocity in m/s
    """
    vr = 500/(2 * massDensityAtZFromEquator(r, z, species, massArray) *
              radialScaleHeight(r) * np.pi * r * 71492e3 ** 2)
    return vr


def magnitudeVector(x0, x1, x2):
    """
    Returns the magnitude of a vector with 3 components
    :param x0:
    :param x1:
    :param x2:
    :return: The magnitude of the 3 components
    """
    vector = [x0, x1, x2]
    return np.sqrt((np.square(vector)).sum())


def averageMagFieldModel(fieldObject, r, theta, phi, model='VIP4'):
    """
    Finds the magnitude of the magnetic field at a position dictated by r, theta and phi
    :param fieldObject: an object to make the field model work
    :param r: radius in RJ
    :param theta: the angle from the pole in radians
    :param phi: in radians from sun, going anticlockwise
    :param model: the type of model for the magnetic field
    :return: the magnitude of the magnetic field in nT
    """
    br, bt, bp, bx, by, bz = fieldObject.Internal_Field(r, theta, phi, False, model)
    b = magnitudeVector(br, bt, bp)
    return b


# Create a series of arrays to hold values
xInRJ = []
yInRJ = []
zInRJ = []
equatorialMagField = []
numberDensity = []
# radius = []
alfvenPointCheck = []
plasmaZDensity = []
radiusForZDensity = []
radialVelocity = []
# radialVelocityAtPi = []
radialVelocityAtZ = []
alfvenVelocity = []
# alfvenVelocityATPi = []
alfvenVelocityAtZ = []

# No longer have to be in the same order
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

fieldGenerator = field_models()
# Calculate radius, scale height, x, y, equatorial magnetic field, Alfven and radial velocity
# and number density by iterating over radius and angle
for r in np.arange(6, 100, 0.5):
    # radius.append(r) # No longer needed
    # scaleHeight.append(radialScaleHeight(r))
    # radialVelocityAtPi.append(radialVelocityFunc(r, speciesList, speciesMass))
    # alfvenVelocityATPi.append(alfvenVelocityAtRPhi(r, 0, speciesMass, speciesMass))
    for phi in np.arange(0, 2 * np.pi + 0.01, 0.1):
        xInRJ.append(r * np.cos(phi))
        yInRJ.append(r * np.sin(phi))
        equatorialMagField.append(equatorialMagneticField(r, phi))
        numberDensity.append(equatorialTotalPlasmaNumberDensity(r, speciesList))
        radialVelocity.append(radialVelocityFunc(r, speciesList, speciesMass))
        alfvenVelocity.append(alfvenVelocityAtRThetaPhi(fieldGenerator, r, 0.5*np.pi, phi, speciesList, speciesMass))


# Check if Alfven velocity is greater than radial, if so set a binary choice to 0
# will be used to create a plot later
for i in range(len(alfvenVelocity)):
    if alfvenVelocity[i] > radialVelocity[i]:
        alfvenPointCheck.append(0)
    else:
        alfvenPointCheck.append(1)

# for r in np.arange(6, 100, 0.5):
#     for z in np.arange(-12, 12, 0.1):
#         theta = np.arctan2(z, r)
#         radiusForZDensity.append(r)
#         zInRJ.append(z)
#         plasmaZDensity.append(densityAtZFromEquator(z, r, speciesList))
#         radialVelocityAtZ.append(radialVelocityFuncAtZ(r, z, speciesList, speciesMass))
#         alfvenVelocityAtZ.append(alfvenVelocityAtRThetaPhi(fieldGenerator, r, theta, 120, speciesMass, speciesMass))


# Save outputs
np.savetxt('alfvenCheck.txt', np.c_[xInRJ, yInRJ, equatorialMagField, numberDensity, alfvenVelocity, radialVelocity,
                                    alfvenPointCheck], delimiter='\t', header='x\ty\tb\tp\tAlfvenV\tRadialV\tCheck')
# np.savetxt('scaleheighttest.txt', np.c_[radius, scaleHeight], delimiter='\t', header='r\tscaleHeight')
# No longer needed

# np.savetxt('zPlasmaDensity.txt', np.c_[radiusForZDensity, zInRJ, plasmaZDensity, radialVelocityAtZ, alfvenVelocityAtZ],
#            delimiter='\t', header='r\tz\tplasmaZDensity\tRadialV\tAflvenV')

# np.savetxt('alfvenradial.txt', np.c_[radius, alfvenVelocityATPi, radialVelocityAtPi], delimiter='\t', header='r\tscaleHeight')
