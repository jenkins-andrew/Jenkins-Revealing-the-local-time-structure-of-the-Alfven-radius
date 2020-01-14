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
    """
    Asks the user if they would like to plot field lines and returns the result
    :return: 1 = Plot, 2 = No plot
    """
    choice = 0
    while True:
        try:
            choice = int(input("\nWould you like to plot?: \n"
                               "(1) Yes \n"
                               "(2) No \n"))
        except ValueError:
            print("Not a valid input:")
            continue
        if (choice > 2) | (choice < 1):
            print("Not a valid input:")
        else:
            break

    return choice


def pathString(rminIn, rmaxIn, pIn, currentSheetIn):
    """
    Returns a string to the path of the text files. This follows the format all txt files are saved as.
    :param rminIn: Minimum radius
    :param rmaxIn: Maximum radius
    :param pIn: The phi value for the trace
    :param currentSheetIn: If the current sheet is on or not
    :return: The string in the form 'newoutput/radius%0.2fto%0.2fphi%0.2fCurrentOn=%s.txt'
    """
    return 'newoutput/radius%0.2fto%0.2fphi%0.2fCurrentOn=%s.txt' % (rminIn, rmaxIn, pIn * np.pi / 180,
                                                                     currentSheetIn)


def printChoiceListAndOption(folder):
    """
    # List all the files in newoutput
    :return:
    """
    files = [f for f in os.listdir(folder)]
    for i in range(len(files)):
        print("(%i) %s" % (i, files[i]))
    fileNo = int(input("\n"))
    path = "newoutput/"+files[fileNo]
    print(path)
    return path


while True:
    try:
        majorRunChoice = int(input("(1) Generate field lines \n"
                                   "(Under Repair) Generate field lines and total mass density along the lines\n"
                                   "(3) Just print\n"
                                   "(4) Generate total mass density along pre-made field lines\n"))
    except ValueError:
        print("Not a valid input:")
        continue
    if (majorRunChoice > 4) | (majorRunChoice < 1):
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
    # Creates a field line trace between rmin and rmax and pmin to pmax, with or without a current sheet
    FieldTrace.produceTraceArrays(rmin, rmax, pmin * np.pi / 180, pmax * np.pi / 180, currentSheet)

    if majorRunChoice == 1:
        # Ask if the user wants to plot the field line trace
        plotChoice = plotChoiceInput()
        if plotChoice == 1:
            if pmax == pmin:
                path = pathString(rmin, rmax, pmax, currentSheet)
                PlotFieldLines.plotOnePhiSet(path)
            else:
                for phi0 in np.arange(pmin, pmax + 0.001, 0.25 * np.pi):
                    # Get a plot for each phi value
                    path = pathString(rmin, rmax, phi0, currentSheet)
                    PlotFieldLines.plotOnePhiSet(path)

    elif majorRunChoice == 2:
        if pmax == pmin:
            path = pathString(rmin, rmax, pmax, currentSheet)
            # Also get the plasma mass density along the field lines
            diffusive_equilibrium_code.runDiffusiveCode(path)
            # Find the Alfven and radial velocities along the field lines
            FieldandDensityGridGenerator.generateAlfvenAndRadialFromDefusive(path)
        else:
            for phi0 in np.arange(pmin, pmax + 0.001, 0.25 * np.pi):
                path = pathString(rmin, rmax, phi0, currentSheet)
                # Also get the plasma mass density along the field lines
                diffusive_equilibrium_code.runDiffusiveCode(path)
                # Find the Alfven and radial velocities along the field lines
                FieldandDensityGridGenerator.generateAlfvenAndRadialFromDefusive(path)
        plotChoice = plotChoiceInput()
        if plotChoice == 1:
            path = pathString(rmin, rmax, pmax, currentSheet)
            # Plot the field lines with the plasma that is said to be in corotation
            PlotFieldLines.plotCorotation(path)

elif majorRunChoice == 3:
    print("What would you like to print?:")
    path = printChoiceListAndOption('newoutput/')
    size = len(np.loadtxt(path)[0])
    # Allows the user to select one of these files and then plot them. Checks what type of file it is and runs the
    # correct plotting method on it
    if size == 4:
        PlotFieldLines.plotOnePhiSet(path)
    elif size == 7:
        PlotFieldLines.plotCorotation(path)
    else:
        print("Cannot currently print file.")

elif majorRunChoice == 4:
    print("Which trace would you like to have lines found for??:")
    path = printChoiceListAndOption('newoutput/')
    size = len(np.loadtxt(path)[0])
    if size != 4:
        print("\nCannot generate a total mass density for this file. May have already been done.")

    else:
        FieldandDensityGridGenerator.generateAlfvenAndRadial(path)

    plotChoice = plotChoiceInput()
    if plotChoice == 1:
        # Plot the field lines with the plasma that is said to be in corotation
        PlotFieldLines.plotCorotation(path)
