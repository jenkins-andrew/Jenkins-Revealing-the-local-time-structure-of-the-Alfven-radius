"""
The main method to allow the whole plotting and tracing to work
"""

import numpy as np
import FieldTrace
import diffusive_equilibrium_code
import PlotFieldLines
import FieldandDensityGridGenerator

majorRunChoice = 0


# plotChoice = 0


def plotChoiceInput():
    choice = 0
    while True:
        try:
            choice = int(input("\nWould you like to plot?: \n"
                               "(1) Yes, plot the field lines\n"
                               "(2) No \n"))
        except ValueError:
            print("Not a valid input:")
            continue
        if (choice > 2) | (choice < 1):
            print("Not a valid input:")
        else:
            break

    return choice


while True:
    try:
        majorRunChoice = int(input("(1) Generate field lines \n"
                                   "(2) Generate field lines and total mass density along the lines\n"
                                   "(3) Something else\n"))
    except ValueError:
        print("Not a valid input:")
        continue
    if (majorRunChoice > 3) | (majorRunChoice < 1):
        print("Not a valid input:")
    else:
        break

if (majorRunChoice == 1) | (majorRunChoice == 2):
    rmin = float(input("Enter minimum radius:\n"))
    rmax = float(input("Enter maximum radius:\n"))
    pmin = float(input("Enter starting phi in degrees:\n"))
    pmax = float(input("Enter final phi in degrees:\n"))
    currentSheet = input("Current sheet on? Y or N:\n")
    if (currentSheet == "N") | (currentSheet == "n"):
        currentSheet = False
    elif (currentSheet == "Y") | (currentSheet == "y"):
        currentSheet = True
    FieldTrace.produceTraceArrays(rmin, rmax, pmin * np.pi / 180, pmax * np.pi / 180, currentSheet)

    if majorRunChoice == 1:
        plotChoice = plotChoiceInput()
        if plotChoice == 1:
            if pmax == pmin:
                path = 'newoutput/radius%0.2fto%0.2fphi%0.2fCurrentOn=%s.txt' % (
                    rmin, rmax, pmax * np.pi / 180, currentSheet)
                PlotFieldLines.plotOnePhiSet(path)
            else:
                for phi0 in np.arange(pmin, pmax + 0.001, 0.25 * np.pi):
                    path = 'newoutput/radius%0.2fto%0.2fphi%0.2fCurrentOn=%s.txt' % (
                        rmin, rmax, phi0 * np.pi / 180, currentSheet)
                    PlotFieldLines.plotOnePhiSet(path)

    elif majorRunChoice == 2:
        if pmax == pmin:
            path = 'newoutput/radius%0.2fto%0.2fphi%0.2fCurrentOn=%s.txt' % (
                rmin, rmax, pmax * np.pi / 180, currentSheet)
            diffusive_equilibrium_code.runDiffusiveCode(path)
            FieldandDensityGridGenerator.generateAlfvenAndRadial(path)
        else:
            for phi0 in np.arange(pmin, pmax + 0.001, 0.25 * np.pi):
                path = 'newoutput/radius%0.2fto%0.2fphi%0.2fCurrentOn=%s.txt' % (
                    rmin, rmax, phi0 * np.pi / 180, currentSheet)
                diffusive_equilibrium_code.runDiffusiveCode(path)
                FieldandDensityGridGenerator.generateAlfvenAndRadial(path)
        plotChoice = plotChoiceInput()
        if plotChoice == 1:
            path = 'newoutput/radius%0.2fto%0.2fphi%0.2fCurrentOn=%s.txt' % (
                rmin, rmax, pmax * np.pi / 180, currentSheet)
            PlotFieldLines.plotCorotation(path)
else:
    print("Not ready yet... Sorry")
