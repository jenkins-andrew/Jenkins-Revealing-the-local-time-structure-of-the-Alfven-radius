import numpy as np
from Magnetic_field_models import field_models

fieldGenerator = field_models()

rInRJ = []
thetaInRadians = []
phiInRadians = []
Bradius, Btheta, Bphi = [], [], []

for r in np.arange(1, 10, 1):
    for theta in np.arange(0.001, 0.5*np.pi, 0.1):
        for phi in np.arange(0.001, 2*np.pi, 0.1):
            rInRJ.append(r)
            thetaInRadians.append(theta)
            phiInRadians.append(phi)
            try:
                Br, Bt, Bp = fieldGenerator.Internal_Field(r, theta, phi, 'JRM09')
            except:
                Br, Bt, Bp = np.NaN
            Bradius.append(Br)
            Btheta.append(Bt)
            Bphi.append(Bp)


np.savetxt('magfieldvalues.txt', np.c_[rInRJ, thetaInRadians, phiInRadians,
                             Bradius, Btheta, Bphi], delimiter='\t', header='r\ttheta\tphi\tBr\tBtheta\tBphi')

