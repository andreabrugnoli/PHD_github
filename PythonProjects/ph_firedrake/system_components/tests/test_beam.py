import numpy as np
import scipy.linalg as la
from math import pi
from system_components.beams import FreeEB, ClampedEB, draw_allbending, draw_bending
from modules_phdae.classes_phsystem import SysPhdaeRig
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
from firedrake import *
from system_components.tests.manipulator_constants import *


fntsize = 15
n_el = 30
L = 1
frac = 2
per = 1/frac
L1 = per * L
L2 = (1-per) * L

n_el1 = int(n_el/frac)
n_el2 = n_el - n_el1

beamCC = ClampedEB(n_el1, L1, 1, 1, 1, 1)
beamFF = FreeEB(n_el2, L2, 1, 1, 1, 1)

npC = beamCC.n_p
npF = beamFF.n_p

beamCF = SysPhdaeRig.gyrator_ordered(beamCC, beamFF, [2, 3], [0, 1], np.eye(2))

npCF = beamCF.n_p

# beamCC = ClampedTruss(n_el1, L1, 1, 1, 1)
# beamFF = FreeTruss(n_el2, L2, 1, 1, 1)
#
# npC = beamCC.n_p
# npF = beamFF.n_p
#
# beamCF = SysPhdaeRig.gyrator_ordered(beamCC, beamFF, [1], [0], np.array([[1]]))
#
# npCF = beamCF.n_p

eigenvalues, eigvectors = la.eig(beamCF.J, beamCF.E)

omega_all = np.imag(eigenvalues)

index = omega_all > 0

omega = omega_all[index]
eigvec_omega = eigvectors[:, index]
perm = np.argsort(omega)
eigvec_omega = eigvec_omega[:, perm]

omega.sort()

#
# print("Smallest positive normalized eigenvalues computed: ")
for i in range(4):
    # Cantilever truss frequency: (2*k-1)/2*pi for k=1,2,...
    print(np.sqrt(omega[i]))
    #
    # real_eig = np.real(np.concatenate((eigvec_omega[:npC-2, i], eigvec_omega[npC:npCF, i]), axis=0))
    # imag_eig = np.imag(np.concatenate((eigvec_omega[:npC-2, i], eigvec_omega[npC:npCF, i]), axis=0))
    #
    real_eig1 = np.real(eigvec_omega[:npC, i])
    imag_eig1 = np.imag(eigvec_omega[:npC, i])

    real_eig2 = np.real(eigvec_omega[npC:npCF, i])
    imag_eig2 = np.imag(eigvec_omega[npC:npCF, i])
    #
    if np.linalg.norm(real_eig1) > np.linalg.norm(imag_eig1):
        eig1 = real_eig1
        eig2 = real_eig2
    else:
        eig1 = imag_eig1
        eig2 = imag_eig2
    #
    x1, u1, w1 = draw_allbending(50, [0, 0, 0], eig1, L1)
    x2, u2, w2 = draw_allbending(50, [0, 0, 0], eig2, L2)
    plt.figure()

    plt.plot(x1, w1, 'r', x2+L1, w2, 'b')
    plt.legend(("Beam1", "Beam2"))

plt.show()

    # real_eig = np.real(eigvec_omega[:npCF, i])
    # imag_eig = np.imag(eigvec_omega[:npCF, i])

    # if np.linalg.norm(real_eig) > np.linalg.norm(imag_eig):
    #     eig = real_eig
    # else:
    #     eig = imag_eig

    # x1, u1, w1 = draw_allbending(50, [0, 0, 0], eig, L)
    # x2, u2, w2 = draw_bending(50, [0, 0, 0], eig[2:], L)

    # plt.plot(x1, w1, 'r', x2, w2, 'b')
    # plt.legend(("With w0", "Without w0"))
    # plt.show()

# beam = FloatTruss(n_el, 1, 1, 1, 1)
#
# eigenvalues, eigvectors = la.eig(beam.J[n_rig:, n_rig:], beam.E[n_rig:, n_rig:])
#
# omega_all = np.imag(eigenvalues)
#
# index = omega_all > 0
#
# omega = omega_all[index]
# eigvec_omega = eigvectors[:, index]
# perm = np.argsort(omega)
# eigvec_omega = eigvec_omega[:, perm]
#

# omega.sort()
#
# print("Smallest positive normalized eigenvalues computed: ")
# for i in range(4):
#     print(omega[i])


# beam = FloatingPlanarEB(n_el, 1, 1, 1)
#
# eigenvalues, eigvectors = la.eig(beam.J[n_rig:,n_rig:], beam.E[n_rig:, n_rig:])
#
# omega_all = np.imag(eigenvalues)
#
# index = omega_all > 0
#
# omega = omega_all[index]
# eigvec_omega = eigvectors[:, index]
# perm = np.argsort(omega)
# eigvec_omega = eigvec_omega[:, perm]
#
# omega.sort()
#
# print("Smallest positive normalized eigenvalues computed: ")
# for i in range(4):
#     print(np.sqrt(omega[i]))