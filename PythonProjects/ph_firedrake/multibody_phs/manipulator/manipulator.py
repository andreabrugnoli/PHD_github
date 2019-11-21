import numpy as np
import scipy.linalg as la
from modules_ph.classes_phsystem import SysPhdaeRig
from system_components.beams import FloatingPlanarEB
from multibody_phs.manipulator.manipulator_constants import n_el, rho1, EI1, L1, rho2, EI2, L2, n_rig, J_joint1, J_joint2, J_payload, m_joint2, m_payload


beam1 = FloatingPlanarEB(n_el, L1, rho1, 1, EI1, 1, J_joint=J_joint1)
E_hinged = beam1.E[2:, 2:]
J_hinged = beam1.J[2:, 2:]
B_hinged = beam1.B[2:, 2:]
beam1_hinged = SysPhdaeRig(len(E_hinged), 0, 1, beam1.n_p, beam1.n_q,
                           E=E_hinged, J=J_hinged, B=B_hinged)

beam2 = FloatingPlanarEB(n_el, L2, rho2, 1, EI2, 1, m_joint=m_joint2, J_joint=J_joint2)

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

sys_int1.pivot(2, 1)
n_int = len(sys_int1.B.T)
ind2_int2 = np.array([n_int-3, n_int-2, n_int-1] , dtype=int)
ind3 = np.array([0, 1, 2], dtype=int)

sys_dae = SysPhdaeRig.transformer_ordered(sys_int1, payload, ind2_int2, ind3, np.eye(3))

J_dae = sys_dae.J
E_dae = sys_dae.E
B_dae = sys_dae.B

sys_ode, T = sys_dae.dae_to_odeE()

J_ode = sys_ode.J
Q_ode = sys_ode.Q
B_ode = sys_ode.B
#
# pathout = '/home/a.brugnoli/GitProjects/MatlabProjects/PH/PH_TITOP/TwoLinks_Manipulator/Matrices_manipulator/'
# Qode_file = 'Q_ode'; Jode_file = 'J_ode'; Bode_file = 'B_ode'
# savemat(pathout + Qode_file, mdict={Qode_file: Q_ode})
# savemat(pathout + Jode_file, mdict={Jode_file: J_ode})
# savemat(pathout + Bode_file, mdict={Bode_file: B_ode})
#
# Edae_file = 'E_dae'; Jdae_file = 'J_dae'; Bdae_file = 'B_dae'
# savemat(pathout + Edae_file, mdict={Edae_file: E_dae})
# savemat(pathout + Jdae_file, mdict={Jdae_file: J_dae})
# savemat(pathout + Bdae_file, mdict={Bdae_file: B_dae})

# plt.figure(); plt.spy(sys_int1.E)
# plt.figure(); plt.spy(sys_int1.J)
# plt.figure(); plt.spy(sys_all.E)
# plt.figure(); plt.spy(sys_all.J)
# plt.show()

# plt.figure()
# plt.spy(J_all)
# plt.figure()
# plt.spy(M_all)
# plt.show()
#
eigenvalues, eigvectors = la.eig(sys_dae.J, sys_dae.E)
omega_all = np.imag(eigenvalues)
index = omega_all >= 0
omega = omega_all[index]
eigvec_omega = eigvectors[:, index]
perm = np.argsort(omega)
eigvec_omega = eigvec_omega[:, perm]
omega.sort()

print(omega)