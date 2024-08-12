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
    return 'newoutput/radius%0.2fto%0.2fphi%0.2fCurrentOn=%s.npy' % (rminIn, rmaxIn, pIn * np.pi / 180,
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
    path = folder+files[fileNo]
    print(path)
    return path


while True:
    try:
        majorRunChoice = int(input("(1) Generate field lines \n"
                                   "(2) Generate field lines and total mass density\n"
                                   "(3) Plot the field lines or plasma sheet with field lines (file dependent)\n"
                                   "(4) Generate total mass density along pre-made field lines\n"
                                   "(5) Generate txt file for one field line\n"
                                   "(6) Calculate Alfven travel times\n"
                                   "(7) Produce orbital trace plot\n"
                                   "(8) Find radial distance for an orbital angle limit\n"))
    except ValueError:
        print("Not a valid input:")
        continue
    if (majorRunChoice > 8) | (majorRunChoice < 1):
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
                for phi0 in np.arange(pmin, pmax + 0.001, 45):
                    # Get a plot for each phi value
                    path = pathString(rmin, rmax, phi0, currentSheet)
                    PlotFieldLines.plotOnePhiSet(path)

    elif majorRunChoice == 2:
        if pmax == pmin:
            path = pathString(rmin, rmax, pmax, currentSheet)
            # Find the Alfven and radial velocities along the field lines
            FieldandDensityGridGenerator.generateAlfvenAndRadial(path)
        else:
            for phi0 in np.arange(pmin, pmax + 0.001, 45):
                path = pathString(rmin, rmax, phi0, currentSheet)
                # Find the Alfven and radial velocities along the field lines
                FieldandDensityGridGenerator.generateAlfvenAndRadial(path)
        plotChoice = plotChoiceInput()
        if plotChoice == 1:
            path = pathString(rmin, rmax, pmax, currentSheet)
            # Plot the field lines with the plasma that is said to be in corotation
            PlotFieldLines.plotCorotation(path)

elif majorRunChoice == 3:
    print("What would you like to print?:")
    path = printChoiceListAndOption('newoutput/')
    size = len(np.load(path, allow_pickle=True)[0][0])
    # Allows the user to select one of these files and then plot them. Checks what type of file it is and runs the
    # correct plotting method on it
    printtype = int(input("What type of print?\n"
                          "1) Field Lines\n"
                          "2) Plasma Sheet\n"))
    if printtype == 1:
        PlotFieldLines.plotOnePhiSet(path)
    elif printtype == 2:
        PlotFieldLines.plotCorotation(path)
    else:
        print("Cannot currently print file.")

elif majorRunChoice == 4:
    print("Which trace would you like to have lines found for?:")
    path = printChoiceListAndOption('newoutput/')
    size = len(np.load(path, allow_pickle=True)[0][0])
    if size != 4:
        print("\nCannot generate a total mass density for this file. May have already been done.")

    else:
        FieldandDensityGridGenerator.generateAlfvenAndRadial(path)

    plotChoice = plotChoiceInput()
    if plotChoice == 1:
        # Plot the field lines with the plasma that is said to be in corotation
        PlotFieldLines.plotCorotation(path)
elif majorRunChoice == 5:
    print("Which file contains a trace you would like?:")
    path = printChoiceListAndOption('newoutput/')
    newstr = ''.join((ch if ch in '0123456789.' else ' ') for ch in path)
    pathstring = [i for i in newstr.split()]
    start = float(pathstring[0])
    end = float(pathstring[1])
    phi = float(pathstring[2])
    output = np.load(path, allow_pickle=True)
    fieldLineStep = int((end - start) / len(output) + 1)

    fieldLineNumber = int(input("What field line number would you like the information for? Note it must be between "
                                "%0.2f and %0.2f RJ and be a multiple of %d.\n" % (start, end, fieldLineStep)))

    arrayNumber = int((fieldLineNumber-start) / fieldLineStep)
    print("Field trace starting at %0.2f for phi = %0.2f" % (fieldLineNumber, phi))
    print(arrayNumber, path)
    np.savetxt('fieldtrace%0.2fphi%0.2f.txt' % (fieldLineNumber, phi), np.c_[output[arrayNumber]], delimiter='\t')

elif majorRunChoice == 6:
    folder = input("Which folder of files would you like to find Alfven travel times for?\n")
    files = [f for f in os.listdir(folder+'/')]
    print("Finding Alfven travel times for files in folder: %s" % folder)
    for i in range(len(files)):
        path = folder + '/' + files[i]
        FieldandDensityGridGenerator.generateAlfvenTravelTimes(path)

elif majorRunChoice == 7:
    folder = input("Which folder of files would you like to find Alfven travel times for?\n")
    print("Which file for the trace?:")
    path = printChoiceListAndOption(folder)
    PlotFieldLines.betterOrbitalTrace(path)

elif majorRunChoice == 8:
    limit = float(input("Angle limit (degrees)\n"))
    percentageError = float(input("Percentage error in limit\n"))
    guess = float(input("Guess of radial starting distance\n"))
    phi = 112
    path = pathString(guess, guess, phi, False)
    angle = 0
    radial = guess
    outOfLimit = True
    while (angle < limit) | (angle > (1+percentageError*.01)*limit):
        FieldTrace.produceTraceArrays(radial, radial, phi * np.pi / 180, phi * np.pi / 180, False)
        FieldandDensityGridGenerator.generateAlfvenAndRadial(path)
        FieldandDensityGridGenerator.generateAlfvenTravelTimes(path)
        angle = PlotFieldLines.angleTravelledThrough('travelTimes/alfvenTravelTimesfor%0.2fto%0.2fatPhi%0.2f.txt' % (radial, radial, phi*np.pi/180))
        print(angle)
        dif = abs(angle - limit)
        if angle > (1+percentageError*.01)*limit:
            radial -= dif/1.12
        elif angle < limit:
            radial += dif/1.12
        path = pathString(radial, radial, phi, False)

    print("Angle is %0.2f at radial distance %0.2f" %(angle, radial))