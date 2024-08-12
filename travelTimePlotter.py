import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as tick

folder = 'traveltimes/'
files = [f for f in os.listdir(folder)]
for i in range(len(files)):
    path = folder + files[i]
    r, t, ft, t2, ft2, fl = np.loadtxt(path, delimiter='\t', unpack=True)
    mask = (r < 60)
    r = r[mask]
    t = t[mask]
    ft = ft[mask]
    t2 = t2[mask]
    ft2 = ft2[mask]
    fl = fl[mask]
    phi = float(path[49:53])
    phi = round(phi * 180/np.pi)
    plt.figure(1)
    plt.plot(r, t/3600, label='Phi = %d' % phi)
    plt.figure(2)
    plt.plot(r, t2/3600, label='Phi = %d' % phi)
    plt.figure(3)
    plt.plot(r, ft, label='Phi = %d' % phi)
    plt.figure(4)
    plt.plot(r, ft2, label='Phi = %d' % phi)
    plt.figure(5)
    plt.plot(r, (t+t2)/3600, label='Phi = %d' % phi)
    plt.figure(6)
    plt.plot(r, fl, label='Phi = %d' % phi)

plt.figure(1)
plt.legend(fontsize=18)
plt.xticks(size=18)
plt.yticks(size=18)
plt.xlabel('Intersection with magnetic equator $(R_J)$', fontsize=18)
plt.ylabel(r'Alfv\'en Travel Time (hrs)', fontsize=18)
plt.title("Northern hemisphere", fontsize=18)
plt.tight_layout()
plt.tick_params(right=True, which='both', labelsize=18)

plt.figure(2)
plt.legend(fontsize=18)
plt.xticks(size=18)
plt.yticks(size=18)
plt.title("Southern hemisphere", fontsize=18)
plt.xlabel('Intersection with magnetic equator $(R_J)$', fontsize=18)
plt.ylabel(r'Alfvén Travel Time (hrs)', fontsize=18)
plt.tick_params(right=True, which='both', labelsize=18)
plt.tight_layout()

plt.figure(3)
plt.legend(fontsize=18)
plt.xticks(size=18)
plt.yticks(size=18)
plt.title("Northern hemisphere", fontsize=18)
plt.ylabel('Fractional time spent in plasma sheet ', fontsize=18)
plt.xlabel('Intersection with magnetic equator $(R_J)$', fontsize=18)
plt.tight_layout()

plt.figure(4)
plt.legend(fontsize=18)
plt.xticks(size=18)
plt.yticks(size=18)
plt.title("Southern hemisphere", fontsize=18)
plt.xlabel('Intersection with magnetic equator $(R_J)$', fontsize=18)
plt.ylabel('Fractional time spent in plasma sheet ', fontsize=18)
plt.tight_layout()

plt.figure(5)
ax = plt.axes()
plt.legend(fontsize=18)
plt.xticks(size=18)
plt.yticks(size=18)
plt.xlabel('Magnetic equator intersection distance $(R_J)$', fontsize=18)
plt.ylabel(r'Alfvén Travel Time (hrs)', fontsize=18)
# plt.title("Northern to Southern hemisphere", fontsize=18)
ax.tick_params(axis='both', which='major', size=6)
ax.tick_params(axis='both', which='minor', size=4)
ax.xaxis.set_major_locator(tick.MultipleLocator(10))
ax.xaxis.set_minor_locator(tick.MultipleLocator(2))
ax.yaxis.set_major_locator(tick.MultipleLocator(0.5))
ax.yaxis.set_minor_locator(tick.MultipleLocator(0.1))
plt.tight_layout()
plt.tick_params(right=True, which='both', labelsize=18)

plt.figure(6)
ax = plt.axes()
plt.legend(fontsize=18)
plt.xticks(size=18)
plt.yticks(size=18)
plt.xlabel('Magnetic equator intersection distance $(R_J)$', fontsize=18)
plt.ylabel(r'Fractional of line in Plasma Sheet', fontsize=18)
# plt.title("Northern to Southern hemisphere", fontsize=18)
ax.tick_params(axis='both', which='major', size=6)
ax.tick_params(axis='both', which='minor', size=4)
ax.xaxis.set_major_locator(tick.MultipleLocator(10))
ax.xaxis.set_minor_locator(tick.MultipleLocator(2))
ax.yaxis.set_major_locator(tick.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(tick.MultipleLocator(0.05))
plt.tight_layout()
plt.tick_params(right=True, which='both', labelsize=18)

plt.show()