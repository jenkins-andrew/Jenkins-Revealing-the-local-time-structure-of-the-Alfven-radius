#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 09:51:55 2019

@author: lorch

Diffusive Equilibrium Solver.

We solve the diffusive equilibrium equation from Dougherty+ [2017] for all 
species given in their work.

This currently works under the assumption of a spin aligned dipole and an 
isotropic plasma.
"""

import numpy as np
import scipy.constants as con
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.gridspec import GridSpec
import glob


# =============================================================================
# FUNCTIONS USED IN SOLVER
# =============================================================================

# We split the equation by Dougherty+ into two parts, an absolute part
# and a correction term, which accounts for the ambipolar potential.

def abs_part(r0, r1, T_par, n0, m):
    r0 *= 71492000  # convert from Rj -> m
    r1 *= 71492000
    Omega = ((2 * np.pi) / (9 * 3600 + 56 * 60))
    ni = n0 * np.e ** ((0.5 * m * Omega ** 2) * ((r1 ** 2 - r0 ** 2) / T_par))
    return ni, n0


def correction(Z, T_par, pot0, pot1):
    cor = np.e ** (-(((Z * con.e) / T_par) * (pot1 - pot0)))
    return cor


# =============================================================================
# number density and temperature at equator models (BagDel 2011 / Dougherty+ 2016)
# =============================================================================

def n0_2017(r, species=None):
    '''
    DESCRIPTION: Improved function for plasma number density, dependent on
    species.

    INPUT:
        r - radius from planet in Rj
        species - species of choice, (list given if species = None)
    OUTPUT:
        n - equatorial number density for specific species in cm^-3

    '''

    # Determines if any species input has been given, if not prints a list of
    # possible choices
    if species == None:
        print('Please select a particle species: \n \
                  e- \n \
                  O+ \n \
                  O++ \n \
                  S+ \n \
                  S++ \n \
                  S+++ \n \
                  H+ \n \
                  Na+ \n \
                  hot O+')
        return None

    species = species.lower()

    # stores values of model parameters
    Spec_params = {'e-': [1, 2451, -6.27, 4.21],
                   'o+': [0.24, 592, -7.36, 0.368],
                   'o++': [0.03, 76.3, -6.73, 0.086],
                   's+': [0.07, 163, -6.81, 0.169],
                   's++': [0.22, 538, -6.74, 0.598],
                   's+++': [0.004, 90.7, -6.21, 0.165],
                   'h+': [0.02, 50.6, -5.31, 0.212],
                   'na+': [0.04, 97.2, -6.75, 0.106],
                   'hot o+': [0.06, 134, -4.63, 1.057]}

    # retrieves the values and tests for a valid species choice
    try:
        ab, a, b, c = Spec_params[species]
    except:
        print('Invlaid species choice')

    # if r values are given in the form of a meshgrid, we can shuffle it about
    # to fit our function. If not, we leave it the way nature intended
    s = None
    try:
        t = r.shape  # stores the shape temporarily
        r = np.concatenate(r)  # turns meshgrid into 1D array
        s = t
        n = np.zeros(len(r))  # creates exact preallocated array
    except:
        r = r
        n = np.zeros(len(r))

    # determine regions where we apply various equations
    eq1 = np.where(r <= 15.2)[0]
    eq2 = np.where(r > 15.2)[0]
    # BagDel origional function for number density
    bd2011 = 1987 * (r[eq2] / 6) ** (-8.2) + 14 * (r[eq2] / 6) ** -3.2 + 0.05 * (r[eq2] / 6) ** -0.65

    # apply the equations and assign values to our pre-allocated array
    if len(eq1 > 0):
        n[eq1] = a * (r[eq1] / 6) ** b
    if len(eq2 > 0):
        n[eq2] = c * bd2011

    # If we concatonated the meshgrid, here we reshape it back to its input form
    if s != None:
        n = np.reshape(n, s)

    return n


# =============================================================================

def T0_2017(r, species=None):
    '''
        DESCRIPTION: Improved function for Equatorial temperatures over radial
        distance, quasi-dependent on species.
        
        INPUT: 
            r - radius from planet in Rj
            species - species of choice, (list given if species = None)
        OUTPUT:
            T - equatorial temperature, given in K
    '''

    # Determines if any species input has been given, if not prints a list of
    # possible choices
    if species == None:
        print('Please select a particle species: \n \
              ions \n \
              protons \n \
              electrons \n \
              hot \n \
              hot electrons \n')
        return None

    species = species.lower()

    # stores values of model parameters
    Spec_params = {'ions': [79.3, 0.714],
                   'protons': [94.1, 0.14],
                   'electrons': [4.6, 3.4],
                   'hot': [362, 0.91],
                   'hot electrons': [35, 4.2]}

    # retrieves the values and tests for a valid species choice
    try:
        a, b = Spec_params[species]
    except:
        print('Invlaid species choice')
        return None

    # if r values are given in the form of a meshgrid, we can shuffle it about
    # to fit our function. If not, we leave it the way nature intended
    s = None
    try:
        t = r.shape  # stores the shape temporarily
        r = np.concatenate(r)  # turns meshgrid into 1D array
        s = t
        T = np.zeros(len(r))  # creates exact preallocated array
    except:
        r = r
        T = np.zeros(len(r))

    T = a * (r / 6) ** b

    # If we concatonated the meshgrid, here we reshape it back to its input form
    if s != None:
        T = np.reshape(T, s)

    T = T * (1.6e-19) / (1.38e-23)
    return T


def TotalNumberDensity(arrayIons, arrayElectrons):
    """
    Andrew Jenkins
    :param arrayIons:
    :param arrayElectrons:
    :return:
    """
    ntotal = np.zeros(len(arrayIons))
    for i in range(len(arrayIons)):
        for j in range(len(arrayIons[0])):
            ntotal[i] += arrayIons[i][j] + arrayElectrons[i][0]
    return ntotal


def totalMassDensity(numberDensityIons, numberDensityElectrons, massArrayIons, massArrayElectrons):
    """
    Andrew Jenkins
    :param numberDensityIons:
    :param numberDensityElectrons:
    :param massArrayIons:
    :param massArrayElectrons:
    :return:
    """
    M = np.zeros(len(numberDensityIons))
    print('This is what I want: '+str(len(numberDensityIons)))
    for i in range(len(numberDensityIons)):
        for j in range(len(numberDensityIons[0])):
            M[i] = numberDensityIons[i][j]*massArrayIons[j] + numberDensityElectrons[i][0]*massArrayElectrons
    M = np.append(M, M[-1])
    return M


# =============================================================================

# labels to use in plotting:
colours = ['red', 'pink', 'maroon', 'blue', 'skyblue', 'violet', 'cyan', 'green']
labels = ['S$^{+}$', 'S$^{++}$', 'S$^{+++}$', 'O$^{+}$', 'O$^{++}$', 'Na$^{+}$', 'Hot O$^{+}$', 'H$^{+}$']


# copy a B field trace path here.

for field_trace_path in glob.glob('output/*.csv'):

    x, y, z, B = np.loadtxt(field_trace_path, delimiter=',', unpack=True)
    rho = np.sqrt(x ** 2 + y ** 2)
    print(field_trace_path)

    # start by getting x, y and z distance along field line from magnetosphere
    xfl = x - x[0]
    yfl = y - y[0]
    zfl = z - z[0]

    # distance along field line from magnetosphere
    s = np.sqrt(xfl ** 2 + yfl ** 2 + zfl ** 2)

    # just a test step while constructing function
    s2 = s[2]
    s1 = s[1]
    s0 = s[0]

    # L = np.array([15])
    # r = np.arange(15,0,-0.1)
    # S = np.arange(0,15,0.1)
    L = np.array([int(np.amax(x))])
    r = np.array(rho)
    S = np.array(s)

    # Now down to the good stuff...
    # We set up a while loop to step through each of the increments along the
    # field line
    # Our initial starting parameters

    pot0 = 0
    ni, n0 = [], []

    # fixed temperatures and mass to be used throughout calculation
    # Temperature of each species:
    TS1 = T0_2017(L, 'ions')  # S+
    TS2 = T0_2017(L, 'ions')  # S++
    TS3 = T0_2017(L, 'ions')  # S+++
    TO1 = T0_2017(L, 'ions')  # O+
    TO2 = T0_2017(L, 'ions')  # O++
    TNa1 = T0_2017(L, 'ions')  # Na+
    THO1 = T0_2017(L, 'hot')  # hot O+
    TH1 = T0_2017(L, 'protons')  # H+
    TE = T0_2017(L, 'electrons')  # e-

    T_ions = np.array([TS1, TS2, TS3, TO1, TO2, TNa1, THO1, TH1])
    T_ions = (T_ions * con.k)  # convert to J
    TE = (TE * con.k)  # convert to J

    # species mass in amu, accounting for loss of electron:
    u = con.u  # atomic mass unit
    ME = 0.00054858  # electron mass in amu
    MS1 = (32.065 - ME)  # S+
    MS2 = 32.065 - (ME * 2)  # S++
    MS3 = 32.065 - (ME * 3)  # S+++
    MO1 = 15.999 - ME  # O+
    MO2 = 15.999 - (ME * 2)  # O++
    MNa1 = 22.989769 - ME  # Na+
    MHO1 = 15.999 - (ME * 2)  # hot O+
    MH1 = 1.00784 - (ME)  # H+

    m_ions = np.array([MS1, MS2, MS3, MO1, MO2, MNa1, MHO1, MH1])
    m_ions = m_ions * u  # convert to kg
    ME = ME * u  # convert to kg

    # Calculate the initial number density a the equator
    NE = n0_2017(L, 'e-')  # e-
    NS1 = n0_2017(L, 's+')  # S+
    NS2 = n0_2017(L, 's++')  # S++
    NS3 = n0_2017(L, 's+++')  # S+++
    NO1 = n0_2017(L, 'O+')  # O+
    NO2 = n0_2017(L, 'O++')  # O++
    NNa1 = n0_2017(L, 'Na+')  # Na+
    NHO1 = n0_2017(L, 'hot O+')  # hot O+
    NH1 = n0_2017(L, 'H+')  # H+

    n_ions = np.array([NS1, NS2, NS3, NO1, NO2, NNa1, NHO1, NH1])
    n_ions = n_ions * 1e6  # convert to m^-3
    NE = NE * 1e6  # convert to m^-3

    # The charge number of each ion
    Z_no = [1, 2, 3, 1, 2, 1, 1, 1]

    for i in range(len(r) - 1):
        nit, n0t = [], []
        for j in range(len(Z_no)):
            ni_temp, n0_temp = abs_part(r[i], r[i + 1], T_ions[j], n_ions[j], m_ions[j])
            nit = np.append(nit, ni_temp)
            n0t = np.append(n0t, n0_temp)
        n_ions = nit
        ni.append(nit)
        n0.append(n0t)
    n0 = np.array(n0)
    ni = np.array(ni)

    ne, ne0 = [], []
    for i in range(len(r) - 1):
        ne_temp, ne0_temp = abs_part(r[i], r[i + 1], TE, NE, ME)
        ne.append(ne_temp)
        ne0.append(ne0_temp)
        NE = ne_temp
    # =============================================================================
    #       PLOTTING NUMBER DENSITIES
    #
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    #
    # for i, j in enumerate(ni.T):
    #     ax.plot(S[0:-1], j, c=colours[i], label=labels[i])
    #
    # ni = ni * Z_no
    # ax.plot(S[0:-1], sum(ni.T), c='k', label=r'$\Sigma N_i Z_i$')
    #
    # # ax.plot(S[0:-1], ne, c='k', label = 'n$_e$', ls = '--')
    # ax.set_xlabel('S [R$_J$]')
    # ax.set_ylabel('n [m$^{-3}$]')
    # ax.set_xlim(0, 14.7)
    # ax.grid(which='both', color=[0.9, 0.9, 0.9])
    # ax.yaxis.set_minor_locator(MultipleLocator(250000))
    # ax.xaxis.set_minor_locator(MultipleLocator(1))
    # ax.legend()

    # Now we need to determine an appropriate value for the ambipolar potential
    # We create two functions, f and g, for each step. We look for the intersect of
    # these points.

    pot0 = 0
    count = 0
    base = 0
    cover = 500
    inc = 0.0001

    ni_s = []
    ne_s = []
    ZN_s = []
    pot_s = []
    cor_s = []
    e_cor = []
    while count < len(S) - 1:
        # print(count)

        pot1 = np.arange(base - cover, base + cover, 0.1)
        cor_term = []
        for i in range(len(Z_no)):
            cor_term.append(correction(Z_no[i], T_ions[i], pot0, pot1))

        e_cor_term = correction(-1, TE, pot0, pot1)

        cor_term = np.array(cor_term)
        n_corrected = []
        for i in range(len(cor_term)):
            n_corrected.append(ni[count][i] * cor_term[i])
        e_corrected = (ne[count] * e_cor_term)

        ZN = []
        for i in range(len(n_corrected)):
            ZN.append(n_corrected[i] * Z_no[i])

        f = e_corrected
        g = sum(ZN)
        diff = f - g

        # test to see if there is a crossing point:
        signs = np.sign(diff)
        # if the first and last signs are the same, then there is no crossing point
        # keep cycling through ambipolar potentials until you reach a crossing point
        while signs[0] == signs[-1]:
            if diff[0] > diff[-1]:
                base += 200
            else:
                base -= 200

            pot1 = np.arange(base - cover, base + cover, 0.1)
            cor_term = []
            for i in range(len(Z_no)):
                cor_term.append(correction(Z_no[i], T_ions[i], pot0, pot1))

            e_cor_term = correction(-1, TE, pot0, pot1)

            cor_term = np.array(cor_term)
            n_corrected = []
            for i in range(len(cor_term)):
                n_corrected.append(ni[count][i] * cor_term[i])
            e_corrected = (ne[count] * e_cor_term)

            ZN = []
            for i in range(len(n_corrected)):
                ZN.append(n_corrected[i] * Z_no[i])

            f = e_corrected
            g = sum(ZN)
            diff = f - g

            signs = np.sign(diff)

        # determine the crossing point
        cross = np.where(abs(diff) == min(abs(diff)))[0]
        #    cross = np.where(signs[0] != signs)[0][0]

        # calculate the electron and ion density for that potential, replace old variables
        ne_s.append(e_corrected[cross])
        ni_s.append([n_corrected[i][cross] for i in range(len(n_corrected))])
        ZN_s.append([ZN[i][cross] for i in range(len(n_corrected))])
        cor_s.append([cor_term[i][cross] for i in range(len(cor_term))])

        pot_s.append(pot1[cross])
        count += 1

    # ============================================================================
    ##       PLOT POTENTIAL VS N FOR EACH STEP UP FIELD LINE
    ##       I used this to make a gif... its not really needed
    #
    # gs = GridSpec(6, 12)
    # fig = plt.figure()
    # fig.set_size_inches(13, 10)
    # ax = fig.add_subplot(gs[:, 0:7])
    # for ii in range(len(n_corrected)):
    #     ax.plot(pot1, n_corrected[ii], color=colours[ii], label=labels[ii])
    #     ax.axhline(ni[count][ii], c=colours[ii], ls='--', alpha=0.5)
    #     ax.set_yscale('log')
    #
    # ax.set_title(str(round((count + 1) * 0.075, 5)) + ' [R$_J$]')
    # ax.plot(pot1, e_corrected, c='k', label='e$^-$')
    # ax.plot(pot1, sum(ZN), c='k', ls='--', label=r'$\Sigma N_i Z_i$')
    # ax.axhline(ne[count], c='grey')
    # ax.grid(which='both')
    # ax.axvline(pot1[cross], c='grey', ls=':')
    # ax.axhline(sum(ni[count]), c='grey', ls='--')
    # ax.set_ylabel('n [m$^{-3}$]')
    # ax.set_xlabel(r'$\Phi$ [V]')
    # ax.set_ylim(10e-2, 10e11)
    # ax.legend()
    #
    # ax1 = fig.add_subplot(gs[0:3, 8:13])
    # ax1.set_xlabel('X [R$_J$]')
    # ax1.set_ylabel('Z [R$_J$]')
    # Jupiter = Circle((0, 0), 1, color='k')
    # ax1.plot(-B['x'], B['z'], c='grey')
    #
    # interest = np.array([count, count + 1])
    # ax1.plot(-B['x'][interest], B['z'][interest], c='r', lw=3)
    # ax1.add_artist(Jupiter)
    # ax1.set_ylim(0, 10)
    #
    # ax2 = fig.add_subplot(gs[3:6, 8:13])
    # ax2.plot(S[0:-1], pot_s1, c='k')
    # ax2.axvline(S[count], c='grey', ls=':')
    # ax2.set_xlabel('S [R$_J$]')
    # ax2.set_ylabel(r'$\Phi $ [V]')
    # ax2.set_xlim(-0.05, 14.8)
    #
    # if count <= 9:
    #     label = '00' + str(count)
    # elif count <= 99:
    #     label = '0' + str(count)
    # else:
    #     label = str(count)
    # fig.savefig(label + '.png')
    # plt.close()
    #
    # # ============================================================================
    #
    # # '''
    # cor_s = np.array(cor_s)[:, :, 0]
    # ni_s = np.array(ni_s)[:, :, 0]
    # ZN_s = np.array(ZN_s)[:, :, 0]
    # ni = np.array(ni)
    # for i in range(len(Z_no)):
    #     ax.plot(S[0:-1], ni.T[i] * cor_s.T[i], color=colours[i], label=labels[i], ls='--')
    # ax.legend()
    # ax.xaxis.set_minor_locator(MultipleLocator(1))
    # ax.set_xlabel('S [R$_J$]')
    # ax.set_ylabel('n [m$^{-3}$]')
    # ax.set_xlim(0, 14.7)
    # ax.set_ylim(0, 2.5e6)
    # ax.set_yticks([0, 0.5e6, 1e6, 1.5e6, 2e6, 2.5e6])
    # ax.set_yticklabels(['0', '0.5x10$^{6}$', '1.0x10$^{6}$', '1.5x10$^{6}$', '2.0x10$^{6}$', '2.5x10$^{6}$'])
    # ax.plot(S[0:-1], sum(ni_s.T), color='k', ls='--')
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(S[0:-1], ne_s, c='k', label='e-')
    # ax.plot(S[0:-1], sum(ZN_s.T), c='r', ls='--', label=r'$\Sigma N_i Z_i$')
    # ax.xaxis.set_minor_locator(MultipleLocator(1))
    # ax.set_xlabel('S [R$_J$]')
    # ax.set_ylabel('n [m$^{-3}$]')
    # ax.legend()
    # ax.set_xlim(0, 14.7)
    # ax.set_yticks([0, 1e6, 2e6, 3e6, 4e6, 5e6, 6e6, 7e6, 8e6])
    # ax.set_yticklabels(['0', '', '2x10$^{6}$', '', '4x10$^{6}$', '', '6x10$^{6}$', '', '8x10$^{6}$'])
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(S[0:-1], pot_s, c='k')
    # ax.xaxis.set_minor_locator(MultipleLocator(1))
    # ax.set_xlabel('S [R$_J$]')
    # ax.set_ylabel(r'$\Phi$ [V]')
    # ax.set_xlim(0, 14.7)

    # =============================================================================
    #       WE LOOK AT HOW THE CORRECTION TERM VARIES ACROSS THE FIELD LINE:
    #
    cor_t = []
    cor_e = []
    for i in range(len(pot_s)):
        t_cor = []
        for j in range(len(ni[i])):
            cor = correction(Z_no[j], T_ions[j], 0, pot_s[i])
            t_cor.append(cor)
        cor_e.append(correction(-1, TE, 0, pot_s[i]))
        cor_t.append(t_cor)

    cor_t = np.array(cor_t)[:, :, 0]
    fig = plt.figure()
    gs = GridSpec(3, 1)
    #
    # ax1 = fig.add_subplot(gs[0])
    # for i in range(len(Z_no)):
    #     ax1.plot(S[0:-1], ni.T[i], color=colours[i], label=labels[i])
    # ax1.legend(loc=[1, -1])
    #
    # ax2 = fig.add_subplot(gs[1])
    # for i in range(len(Z_no)):
    #     ax2.plot(S[0:-1], cor_t.T[i], color=colours[i], label=labels[i])
    #
    # ax3 = fig.add_subplot(gs[2])
    # for i in range(len(Z_no)):
    #     ax3.plot(S[0:-1], cor_t.T[i] * ni.T[i], color=colours[i], label=labels[i])
    #
    # ax1.set_xlim(0, 14.7)
    # ax2.set_xlim(0, 14.7)
    # ax3.set_xlim(0, 14.7)
    # ax3.set_xlabel('S [R$_J$]')
    # ax1.set_ylabel('n [m$^{-3}$]')
    # ax2.set_ylabel('cor')
    # ax3.set_ylabel('n x cor [m$^{-3}$]')
    # ax1.grid(which='both')
    # ax2.grid(which='both')
    # ax3.grid(which='both')
    # ax1.xaxis.set_minor_locator(MultipleLocator(1))
    # ax2.xaxis.set_minor_locator(MultipleLocator(1))
    # ax3.xaxis.set_minor_locator(MultipleLocator(1))
    # ax1.set_ylim(-500, 2500000)
    # ax3.set_ylim(-500, 2500000)
    #
    # ax1.yaxis.set_minor_locator(MultipleLocator(250000))
    # ax2.yaxis.set_minor_locator(MultipleLocator(10))
    # ax3.yaxis.set_minor_locator(MultipleLocator(250000))

    # =============================================================================

    # Save density profiles in csv format
    #
    # file = open(savepath, 'w+')
    # file.write('S,nS+,nS++,nS+++,nO+,nO++,nNa+,nHot+,nH+,ne-,cS+,cS++,cS+++,cO+,cO++,cNa+,cHot+,cH+,ce-\n')
    # for i in range(len(ni)):
    #     file.write(str(S[0:-1][i]) + ',' + str(ni[i][0]) + ',' + str(ni[i][1]) + ',' + str(ni[i][2]) + \
    #                ',' + str(ni[i][3]) + ',' + str(ni[i][4]) + ',' + str(ni[i][5]) \
    #                + ',' + str(ni[i][6]) + ',' + str(ni[i][7]) + ',' + str(ne[i][0]) + ',' + \
    #                str(cor_t[i][0]) + ',' + str(cor_t[i][1]) + ',' + str(cor_t[i][2]) \
    #                + ',' + str(cor_t[i][3]) + ',' + str(cor_t[i][4]) + ',' + str(cor_t[i][5]) \
    #                + ',' + str(cor_t[i][6]) + ',' + str(cor_t[i][7]) + ',' + str(cor_e[i][0]) + '\n')
    # file.close()
    nAverage = TotalNumberDensity(ni, ne)
    massDensity = totalMassDensity(ni, ne, m_ions, ME)
    print(len(nAverage))
    print(len(ni))

    np.savetxt('output/postFieldLine/radius%0.0ftheta%0.0f.txt' % (int(np.amax(x)), int(np.arctan2(y[0], z[0]))), np.c_[x, y, z, B, massDensity], delimiter=',')
#
#
#
# n_ions = np.array([NS1,NS2,NS3,NO1,NO2,NNa1,NHO1,NH1])

