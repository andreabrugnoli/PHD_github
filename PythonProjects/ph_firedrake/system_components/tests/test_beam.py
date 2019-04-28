import numpy as np
import scipy.linalg as la
from system_components.beams import FreeEB, ClampedEB
from modules_phdae.classes_phsystem import SysPhdae
import matplotlib.pyplot as plt
from scipy.io import savemat
from system_components.tests.manipulator_constants import *

n_el = 1

# Cantilever truss frequency: (2*k-1)/2*pi for k=1,2,...
per = 0.5
beamC = ClampedEB(n_el, per*L1, rho1, 1, EI1, 1)
beamF = FreeEB(n_el, (1-per)*L1, rho1, 1, EI1, 1)

beamCantilever = SysPhdae.gyrator(beamC, beamF, [2, 3], [0, 1], np.eye(2))

plt.spy(beamCantilever.E)

eigenvalues, eigvectors = la.eig(beamCantilever.J, beamCantilever.E)
# eigenvalues, eigvectors = la.eig(beamC.J, beamC.E)

omega_all = np.imag(eigenvalues)

index = omega_all > 0

omega = omega_all[index]
eigvec_omega = eigvectors[:, index]
perm = np.argsort(omega)
eigvec_omega = eigvec_omega[:, perm]

omega.sort()

k_n = omega**(0.5)*L1*(rho1/(EI1))**(0.25)
print("Smallest positive normalized eigenvalues computed: ")
for i in range(n_el*2):
    print(k_n[i])

#
# pathout = '/home/a.brugnoli/GitProjects/MatlabProjects/PH/PH_TITOP/EulerBernoulliBeam/Matrices_ClampedEB/'
# Edae_file = 'E_dae'; Jdae_file = 'J_dae'; Bdae_file = 'B_dae'
# savemat(pathout + Edae_file, mdict={Edae_file: beamCantilever.E})
# savemat(pathout + Jdae_file, mdict={Jdae_file: beamCantilever.J})
# savemat(pathout + Bdae_file, mdict={Bdae_file: beamCantilever.B})
#
# Jode_file = 'J_ode'; Qode_file = 'Q_ode'; Bode_file = 'B_ode'
# savemat(pathout + Jode_file, mdict={Jode_file: beamCantilever.J})
# savemat(pathout + Qode_file, mdict={Qode_file: la.inv(beamCantilever.E)})
# savemat(pathout + Bode_file, mdict={Bode_file: beamCantilever.B})


#
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