import numpy as np
from Magnetic_field_models import field_models
import matplotlib.pyplot as plt


def radialScaleHeight(r):
    """
    Finds the scale height at a radius
    :param r: Radius in R_J
    :return: Scale height in R_J
    """
    h = -0.116 + 2.14*np.log10(r/6) - 2.05*np.log10(r/6)**2 + 0.491*np.log10(r/6)**3 + 0.126*np.log10(r/6)**4
    H = 10 ** h
    return H


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
        n = b2011
    return n


def frankPatersonMassProfile(r):
    """
    Total mass density of the plasma in kg/m^3
    :param r: radius in R_J
    :param species: List of species with parameters needed for equatorialTotalPlasmaNumberDensity()
    :param massAmuArray: List of species with masses in amu
    :return: Mass Density in kg/m^3
    """
    n = 3.2e8*r**(-6.9) + 9.9*r**(-1.28)

    M = n*1e6 * 16*1.67e-27
    return M


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
            n = np.zeros(len(r))
            print('Species do not match')
        M += n*1e6 * mass*1.67e-27

    return M


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

    Va = averageMagFieldModel(fieldModel, r, theta, phi, model) * 1e-9 / np.sqrt(1.25663706212e-6 *
                                                                                 totalMassDensity(r, species, massArray))

    return Va


def radialVelocityFrankPaterson(r, species, massArray, mdot):
    """
    The radial velocity at distance r
    :param r: radius in RJ
    :return: radial velocity in m/s
    """
    vr = mdot/(2 * totalMassDensity(r, species, massArray) * radialScaleHeight(r) * np.pi * r * 71492e3 ** 2)
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


def averageMagFieldModel(fieldObject, r, theta, phi, model='VIP4', currentOn=True):
    """
    Finds the magnitude of the magnetic field at a position dictated by r, theta and phi
    :param fieldObject: an object to make the field model work
    :param r: radius in RJ
    :param theta: the angle from the pole in radians
    :param phi: in radians from sun, going anticlockwise
    :param model: the type of model for the magnetic field
    :return: the magnitude of the magnetic field in nT
    """
    br, bt, bp, bx, by, bz = fieldObject.Internal_Field(r, theta, phi, currentOn, model)
    b = magnitudeVector(br, bt, bp)
    return b


# Create a series of arrays to hold values

radius = []
magneticFieldDipole = []
magneticFieldNotDipole = []
radialVelocity500 = []
alfvenVelocityATPi = []
radialVelocity1300 = []
radialVelocity280 = []

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
    radius.append(r)  # No longer needed
    radialVelocity500.append(radialVelocityFrankPaterson(r, speciesMass, speciesMass, 500)/1000)
    radialVelocity280.append(radialVelocityFrankPaterson(r, speciesMass, speciesMass, 280)/1000)
    radialVelocity1300.append(radialVelocityFrankPaterson(r, speciesMass, speciesMass, 1300)/1000)
    alfvenVelocityATPi.append(alfvenVelocityAtRThetaPhi(fieldGenerator, r, 0.5*np.pi, 120, speciesMass, speciesMass, 'VIP4')/1000)
    magneticFieldDipole.append(averageMagFieldModel(fieldGenerator, r, 0.5*np.pi, 120, 'simple', False))
    magneticFieldNotDipole.append(averageMagFieldModel(fieldGenerator, r, 0.5 * np.pi, 120, 'VIP4'))

plt.figure()

plt.plot(radius, radialVelocity500, linestyle='-', label='Radial Velocity Mdot = 500 kgm/s')
plt.plot(radius, radialVelocity280, linestyle='-.', label='Radial Velocity Mdot = 280 kgm/s')
plt.plot(radius, radialVelocity1300, linestyle='--', label='Radial Velocity Mdot = 1300 kgm/s')
plt.plot(radius, alfvenVelocityATPi, linestyle=':', label='Alfven Velocity')
plt.legend(fontsize=18)
plt.xticks(size=18)
plt.yticks(size=18)
plt.xlabel('Radius $(R_J)$', fontsize=18)
plt.ylabel('Velocity (km/s)', fontsize=18)
plt.yscale('log')
plt.ylim(1e0, 1e3)
plt.tight_layout()

plt.figure()
plt.plot(radius, magneticFieldDipole, label='Dipole')
plt.plot(radius, magneticFieldNotDipole, label='VIP4')
plt.legend(fontsize=18)
plt.xticks(size=18)
plt.yticks(size=18)
plt.xlabel('Radius $(R_J)$', fontsize=18)
plt.ylabel('Magnitude (nT)', fontsize=18)
plt.yscale('log')
plt.tight_layout()
plt.show()

