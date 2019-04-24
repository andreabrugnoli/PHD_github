import numpy as np
import scipy.linalg as la
from system_components.beams import FloatingPlanarEB, FloatFlexBeam, FloatTruss

n_el = 2
n_rig = 3

# Cantilever truss frequency: (2*k-1)/2*pi for k=1,2,...

beam = FloatFlexBeam(n_el, 1, 1, 1, 1, 1)

eigenvalues, eigvectors = la.eig(beam.J, beam.E)

omega_all = np.imag(eigenvalues)

index = omega_all > 0

omega = omega_all[index]
eigvec_omega = eigvectors[:, index]
perm = np.argsort(omega)
eigvec_omega = eigvec_omega[:, perm]

omega.sort()

print("Smallest positive normalized eigenvalues computed: ")
for i in range(4):
    print((omega[i]))

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