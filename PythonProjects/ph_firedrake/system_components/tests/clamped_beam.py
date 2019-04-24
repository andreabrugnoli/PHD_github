# EB beam written with the port Hamiltonian approach
import numpy as np
import scipy.linalg as la
from modules_phdae.classes_phsystem import SysPhdaeRig
from system_components.beams import FloatingPlanarEB
from scipy.io import savemat
from system_components.tests.manipulator_constants import n_el, rho1, EI1, L1, n_rig, J_joint1, m_joint1

beam = FloatingPlanarEB(n_el, rho1, EI1, L1, m_joint=m_joint1, J_joint=J_joint1)

J = beam.J_e
M = beam.M_e
B = beam.B_e
n_e = beam.n
n_lmb = 3
G = np.zeros((n_e, n_lmb))
G[:n_rig, :n_rig] = np.eye(n_lmb)

Z_lmb = np.zeros((n_lmb, n_lmb))
Z_al_lmb = np.zeros((n_e, n_lmb))
Z_u_lmb = np.zeros((n_lmb, n_e))

J_aug = np.vstack([np.hstack([J, G]),
                    np.hstack([-G.T, Z_lmb])])

E_aug = np.vstack([np.hstack([M, Z_al_lmb]),
                    np.hstack([Z_u_lmb, Z_lmb])])
B_C = B[:, n_rig:]

B_e = np.concatenate((np.zeros_like(B_C), B_C), axis=1)
B_lmb = np.concatenate((np.eye(n_lmb), Z_lmb), axis=1)

B_aug = np.concatenate((B_e, B_lmb))

n_aug = n_e + n_lmb

beam_dae = SysPhdaeRig(n_aug, n_lmb, n_rig, beam.n_p, beam.n_q, E=E_aug, J=J_aug, B=B_aug)
beam_ode, T = beam_dae.dae_to_ode()

pathout = '/home/a.brugnoli/GitProjects/MatlabProjects/PH/PH_TITOP/SimpleBeam/Matrices_ClampedEB/'
Edae_file = 'E_dae'; Jdae_file = 'J_dae'; Bdae_file = 'B_dae'
savemat(pathout + Edae_file, mdict={Edae_file: beam_dae.E})
savemat(pathout + Jdae_file, mdict={Jdae_file: beam_dae.J})
savemat(pathout + Bdae_file, mdict={Bdae_file: beam_dae.B})

Jode_file = 'J_ode'; Qode_file = 'Q_ode'; Bode_file = 'B_ode'
savemat(pathout + Jode_file, mdict={Jode_file: beam_ode.J})
savemat(pathout + Qode_file, mdict={Qode_file: beam_ode.Q})
savemat(pathout + Bode_file, mdict={Bode_file: beam_ode.B[:, 3:]})


# tol = 1e-6
# eigenvalues, eigvectors = la.eig(J_all, M_all) # la.eig(J_all[2:, 2:], M_all[2:, 2:]) #
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
# k_n = np.sqrt(omega)*coeff_norm
# print("Smallest positive normalized eigenvalues computed: ")
# for i in range(2*n):
#     print(k_n[i])

#
# eigvec_w = eigvec_omega[:n_Vp, :]
# eigvec_w_real = np.real(eigvec_w)
# eigvec_w_imag = np.imag(eigvec_w)
#
# eig_funH2 = Function(Vp)
# Vp_4proj = FunctionSpace(mesh, "CG", 2)
# eig_funH1 = Function(Vp_4proj)
#
# n_fig = 3
# plot_eigenvector = True
# if plot_eigenvector:
#
#     for i in range(n_fig):
#         z_real = eigvec_w_real[:, i]
#         z_imag = eigvec_w_imag[:, i]
#
#         tol = 1e-6
#         fntsize = 20
#
#         eig_funH2.vector()[:] = z_imag
#         eig_funH1.assign(project(eig_funH2, Vp_4proj))
#         plot(eig_funH1)
#         plt.xlabel('$x$', fontsize=fntsize)
#         plt.title('Eigenvector $e_p$', fontsize=fntsize)
#
#         # if i<4:
#         #     plt.figure()
#         #     plt.plot([0, L], z_real[:n_rig])
#         #     plt.xlabel('$x$', fontsize=fntsize)
#         #     plt.title('Eigenvector $e_p$', fontsize=fntsize)
#         # else:
#         #     eig_funH2.vector()[:] = np.concatenate((np.array([0, 0]), z_real[n_rig:]))
#         #     eig_funH1.assign(project(eig_funH2, Vp_4proj))
#         #     plot(eig_funH1)
#         #     plt.xlabel('$x$', fontsize=fntsize)
#         #     plt.title('Eigenvector $e_p$', fontsize=fntsize)
#
#     plt.show()