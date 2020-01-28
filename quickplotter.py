import numpy as np
import matplotlib.pyplot as plt
import os

plt.figure()

files = [f for f in os.listdir('travelTimes/')]
for i in range(len(files)):
    path = "travelTimes/" + files[i]
    r, t = np.loadtxt(path, delimiter='\t', unpack=True)
    phi = float(path[48:52])
    phi = round(phi * 180/np.pi)

    plt.plot(r, t, label='Phi = %d' % phi)

plt.legend(fontsize=18)
plt.xticks(size=18)
plt.yticks(size=18)
plt.xlabel('Equatorial starting Radius $(R_J)$', fontsize=18)
plt.ylabel('Alfven Travel Time $(s)$', fontsize=18)
plt.tight_layout()
plt.show()
