"""
The main method to allow the whole plotting and tracing to work
"""

import numpy as np
import FieldTrace
import diffusive_equilibrium_code
majorRunChoice = 0

while True:
    try:
        majorRunChoice = int(input("(1) Generate field lines \n"
                                   "(2) Generate field lines and total mass density along the lines\n"
                                   "(3) Just plot\n"))
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
    currentSheet = input("Current sheet on? True or False:\n")
    if currentSheet == "False":
        currentSheet = False
    FieldTrace.produceTraceArrays(rmin, rmax, pmin*np.pi/180, pmax*np.pi/180, currentSheet)
elif majorRunChoice == 2:
    diffusive_equilibrium_code.runDiffusiveCode()
else:
    print("Not ready yet... Sorry")
