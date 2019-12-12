"""
The main method to allow the whole plotting and tracing to work
"""

import numpy as np
import FieldTrace
import diffusive_equilibrium_code
import PlotFieldLines
import FieldandDensityGridGenerator
import os

majorRunChoice = 0


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
                                   "(3) Just print\n"))
    except ValueError:
        print("Not a valid input:")
        continue
    if (majorRunChoice > 3) | (majorRunChoice < 1):
        print("Not a valid input:")
    else:
        break

if (majorRunChoice == 1) | (majorRunChoice == 2):
    rmin = float(input("Enter starting radius:\n"))
    rmax = float(input("Enter final radius:\n"))
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
    files = [f for f in os.listdir('newoutput/')]
    print("What would you like to print?:")
    for i in range(len(files)):
        print("(%i) %s" % (i, files[i]))
    fileNo = int(input("\n"))
    path = "newoutput/"+files[fileNo]
    print(path)
    size = len(np.loadtxt(path)[0])

    if size == 4:
        PlotFieldLines.plotOnePhiSet(path)
    elif size == 7:
        PlotFieldLines.plotCorotation(path)
    else:
        print("Cannot currently print file.")

