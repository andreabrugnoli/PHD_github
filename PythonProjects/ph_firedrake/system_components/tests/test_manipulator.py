import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import sys
sys.path.append("/home/a.brugnoli/GitProjects/PythonProjects/modules_phdae")
sys.path.append("/home/a.brugnoli/GitProjects/PythonProjects/ph_firedrake")
from classes_phsystem import SysPhdaeRig
from system_components.classes_beam import FloatingEB

n_el = 2

J_joint1 = 0.1  # kg/m^2
J_joint2 = 0.1  # kg/m^2

m_joint2 = 1
rho1 = 0.2  # kg/m
rho2 = 0.2  # kg/m

EI1 = 1  # N m^2
EI2 = 1  # N m^2

L1 = 0.5  # m
L2 = 0.5  # m

m_payload = 0.1  # kg
J_payload = 0.5 * 10**-3  # kg/m^2

n_rig = 3

beam1 = FloatingEB(n_el, rho1, EI1, L1, J_joint=J_joint1)
E_hinged = beam1.E[2:, 2:]
J_hinged = beam1.J[2:, 2:]
B_hinged = beam1.B[2:, 2:]
beam1_hinged = SysPhdaeRig(len(E_hinged), 0, 1, beam1.n_p, beam1.n_q,
                           E=E_hinged, J=J_hinged, B=B_hinged)

beam2 = FloatingEB(n_el, rho2, EI2, L2, m_joint=m_joint2, J_joint=J_joint2)

M_payload = la.block_diag(m_payload, m_payload, J_payload)
J_payload = np.zeros((n_rig, n_rig))
B_payload = np.eye(n_rig)

payload = SysPhdaeRig(n_rig, 0, n_rig, 0, 0, E=M_payload, J=J_payload, B=B_payload)

alpha2 = 0
R = np.array([[np.cos(alpha2), np.sin(alpha2)],
              [-np.sin(alpha2), np.cos(alpha2)]])

ind1 = np.array([1, 2], dtype=int)
ind2_int1 = np.array([0, 1], dtype=int)

# sys_int1 = SysPhdae.transformer(beam1, beam2, ind1, ind2_int1, R)
# sys_all = SysPhdae.transformer(sys_int1, payload, ind2_int2, ind3, np.eye(3))

sys_int1 = SysPhdaeRig.transformer_ordered(beam1_hinged, beam2, ind1, ind2_int1, R)
n_int = len(sys_int1.B.T)
print(n_int)
ind2_int2 = np.array([n_int-3, n_int-2, n_int-1] , dtype=int)
ind3 = np.array([0, 1, 2], dtype=int)

sys_all = SysPhdaeRig.transformer_ordered(sys_int1, payload, ind2_int2, ind3, np.eye(3))

plt.figure(); plt.spy(sys_int1.E)
plt.figure(); plt.spy(sys_int1.J)
plt.figure(); plt.spy(sys_all.E)
plt.figure(); plt.spy(sys_all.J)
plt.show()

J_all = sys_all.J
M_all = sys_all.E

# plt.figure()
# plt.spy(J_all)
# plt.figure()
# plt.spy(M_all)
# plt.show()

eigenvalues, eigvectors = la.eig(J_all, M_all)

omega_all = np.imag(eigenvalues)

index = omega_all > 0

omega = omega_all[index]
eigvec_omega = eigvectors[:, index]
perm = np.argsort(omega)
eigvec_omega = eigvec_omega[:, perm]

omega.sort()

print(omega)