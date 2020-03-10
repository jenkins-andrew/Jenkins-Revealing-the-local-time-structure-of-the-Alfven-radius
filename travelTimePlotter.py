import numpy as np
import matplotlib.pyplot as plt
import os

folder = 'traveltimes/'
files = [f for f in os.listdir(folder)]
for i in range(len(files)):
    path = folder + files[i]
    r, t, ft, t2, ft2 = np.loadtxt(path, delimiter='\t', unpack=True)
    phi = float(path[49:53])
    phi = round(phi * 180/np.pi)
    plt.figure(1)
    plt.plot(r, t, label='Phi = %d' % phi)
    plt.figure(2)
    plt.plot(r, t2, label='Phi = %d' % phi)
    plt.figure(3)
    plt.plot(r, ft, label='Phi = %d' % phi)
    plt.figure(4)
    plt.plot(r, ft2, label='Phi = %d' % phi)
    plt.figure(5)
    plt.plot(r, t+t2, label='Phi = %d' % phi)

plt.figure(1)
plt.legend(fontsize=18)
plt.xticks(size=18)
plt.yticks(size=18)
plt.xlabel('Intersection with magnetic equator $(R_J)$', fontsize=18)
plt.ylabel(r'Alfv\'en Travel Time $(s)$', fontsize=18)
plt.title("Northern hemipshere", fontsize=18)
plt.tight_layout()
plt.tick_params(right=True, which='both', labelsize=18)

plt.figure(2)
plt.legend(fontsize=18)
plt.xticks(size=18)
plt.yticks(size=18)
plt.title("Southern hemipshere", fontsize=18)
plt.xlabel('Intersection with magnetic equator $(R_J)$', fontsize=18)
plt.ylabel(r'Alfv$\'e$n Travel Time $(s)$', fontsize=18)
plt.tick_params(right=True, which='both', labelsize=18)
plt.tight_layout()

plt.figure(3)
plt.legend(fontsize=18)
plt.xticks(size=18)
plt.yticks(size=18)
plt.title("Northern hemipshere", fontsize=18)
plt.ylabel('Fractional time spent in plasma sheet ', fontsize=18)
plt.xlabel('Intersection with magnetic equator $(R_J)$', fontsize=18)
plt.tight_layout()

plt.figure(4)
plt.legend(fontsize=18)
plt.xticks(size=18)
plt.yticks(size=18)
plt.title("Southern hemipshere", fontsize=18)
plt.xlabel('Intersection with magnetic equator $(R_J)$', fontsize=18)
plt.ylabel('Fractional time spent in plasma sheet ', fontsize=18)
plt.tight_layout()

plt.figure(1)
plt.legend(fontsize=18)
plt.xticks(size=18)
plt.yticks(size=18)
plt.xlabel('Intersection with magnetic equator $(R_J)$', fontsize=18)
plt.ylabel(r'Alfv\'en Travel Time $(s)$', fontsize=18)
plt.title("Northern hemipshere", fontsize=18)
plt.tight_layout()
plt.tick_params(right=True, which='both', labelsize=18)

plt.show()