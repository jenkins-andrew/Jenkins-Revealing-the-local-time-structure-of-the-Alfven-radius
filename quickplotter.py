import numpy as np
import matplotlib.pyplot as plt
import os

plt.figure()

files = [f for f in os.listdir('travelTimes/')]
for i in range(len(files)):
    path = "travelTimes/" + files[i]
    r, t = np.loadtxt(path, delimiter='\t', unpack=True)
    phi = (path[48:52])

    plt.plot(r, t, label='Phi = %s' % phi)

plt.legend(fontsize=18)
plt.xticks(size=18)
plt.yticks(size=18)
plt.xlabel('Equatorial starting Radius $(R_J)$', fontsize=18)
plt.ylabel('Alfven Travel Time $(s)$', fontsize=18)
plt.tight_layout()
plt.show()
