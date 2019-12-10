import numpy as np
from Magnetic_field_models import field_models


def magnitudeVector(x0, x1, x2):
    """
    Get the magnitude of a vector with 3 components
    :param x0:
    :param x1:
    :param x2:
    :return: a float for the magnitude of the vector
    """
    vector = [x0, x1, x2]
    return np.sqrt((np.square(vector)).sum())


def unitVector(x0, x1, x2):
    """
    Get the unit vector for a given vector of 3 components
    :param x0:
    :param x1:
    :param x2:
    :return: unit vector in 3 components
    """
    vector = [x0, x1, x2]
    v_hat = vector / np.sqrt((np.square(vector)).sum())
    return v_hat[0], v_hat[1], v_hat[2]


def cart_sph(x, y, z):
    """
    From Chris' code
    :param x: in R_J
    :param y: in R_J
    :param z: in R_J
    :return: r, theta, phi in R_J and radians
    """
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arctan2(np.sqrt(x ** 2 + y ** 2), z)
    phi = np.arctan2(y, x)
    return r, theta, phi


def sph_cart(r, theta, phi):
    """
    From Chris' code
    :param r: in R_J
    :param theta: in radians
    :param phi: in radians
    :return:
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


class FieldTrace(object):

    @staticmethod
    def produceTraceArrays(currentOn=False, modelType='VIP4'):
        """
        To do the field trace using a model over a range of phi and radii. Each phi and radius field trace is saved as a
        separate text file with the radius and phi value saved in the name. Made as a function such that this could be
        used as a class in the future.
        :param modelType: Default VIP4
        :param currentOn: Default False
        """
        printTester = 0
        fieldGenerator = field_models()
        signArray = [-1, 1]  # To swap the direction of travel along the field line as well as fix array ordering
        rmin = 6
        rmax = 50
        pmin = 0
        pmax = 0
        xInRJ, yInRJ, zInRJ, Bmag = [], [], [], []
        for phi0 in np.arange(pmin, pmax + 0.001, 0.25 * np.pi):
            for r0 in np.arange(rmin, rmax + 0.001, 2):
                # Start a new field line trace
                for sign in signArray:
                    # Start a new direction along the field line
                    theta = 0.5 * np.pi
                    r = r0
                    phi = phi0
                    tempxInRJ, tempyInRJ, tempzInRJ, tempBmag = [], [], [], []  # So I can have two sets of arrays to
                    # combine later
                    x, y, z = sph_cart(r, theta, phi)
                    print('Radius = %5.2f and Phi = %1.2f started going %1.0f' % (r, phi * 180 / np.pi, sign))
                    while r >= 1:
                        Br, Bt, Bp, Bx, By, Bz = fieldGenerator.Internal_Field(r, theta, phi, currentOn, modelType)
                        if printTester % 1 == 0:
                            tempxInRJ.append(x)
                            tempyInRJ.append(y)
                            tempzInRJ.append(z)
                            tempBmag.append(magnitudeVector(Br, Bt, Bp))
                        # print(magnitudeVector(Br, Bt, Bp))
                        # print(magnitudeVector(Bx, By, Bz))
                        xMove, yMove, zMove = unitVector(Bx, By, Bz)
                        step = np.log10(magnitudeVector(Bx, By, Bz)) * 10
                        x += sign * xMove / step
                        y += sign * yMove / step
                        z += sign * zMove / step
                        r, theta, phi = cart_sph(x, y, z)
                        printTester += 1
                    # Flipping the arrays if they need to be before putting them together and saving the final trace array
                    tempxInRJ = tempxInRJ[::sign]
                    tempyInRJ = tempyInRJ[::sign]
                    tempzInRJ = tempzInRJ[::sign]
                    tempBmag = tempBmag[::sign]
                    xInRJ.extend(tempxInRJ)
                    yInRJ.extend(tempyInRJ)
                    zInRJ.extend(tempzInRJ)
                    Bmag.extend(tempBmag)
            np.savetxt('newoutput/radius%0.0fto%0.0fphi%0.0fCurrentOn=%s.txt' % (rmin, rmax, phi0, currentOn),
                       np.c_[xInRJ, yInRJ, zInRJ, Bmag], delimiter='\t')
        pass
