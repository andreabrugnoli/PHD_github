from firedrake import *
from system_components.plates import FloatingKP, FloatingBellKP, FloatingMP
from system_components.tests.kirchhoff_constants import *
import numpy as np
import scipy.linalg as la
from scipy.io import savemat
from modules_phdae.classes_phsystem import SysPhdaeRig, check_positive_matrix


pointP = np.array([0, Ly/6])
pointC1 = np.array([Lx, Ly/3]).reshape((-1, 2))
pointC2 = np.array([Lx, 2*Ly/3]).reshape((-1, 2))

pointsC = np.vstack((pointC1, pointC2))
plate = FloatingMP(Lx, Ly, h, rho, E, nu, nx, ny, pointP)
# plate = FloatingBellKP(Lx, Ly, h, rho, E, nu, nx, ny, pointP, pointC1, modes=False)

# pl_red = plate.reduce_system(0.001, 6)
# J = pl_red.J
# M = pl_red.E
# assert check_positive_matrix(M)
# B = pl_red.B
# n_e = pl_red.n
# n_lmb = 9
# n_rig = pl_red.n_r
# G = B #np.zeros((n_e, n_lmb))
# # G[:n_rig, :n_rig] = np.eye(n_lmb)
# Z_lmb = np.zeros((n_lmb, n_lmb))
# Z_al_lmb = np.zeros((n_e, n_lmb))
# Z_u_lmb = np.zeros((n_lmb, n_e))
#
# J_aug = np.vstack([np.hstack([J, G]),
#                     np.hstack([-G.T, Z_lmb])])
#
# E_aug = la.block_diag(M, Z_lmb)
#
# print(len(J_aug))
# # B_C = B[:, n_rig:]
# #
# # B_aug = np.concatenate((B_C, Z_lmb))
#
# n_aug = n_e + n_lmb
#
# eigenvalues, eigvectors = la.eig(plate.J_f, plate.M_f)
# # eigenvalues, eigvectors = la.eig(J_aug, E_aug)
# omega_all = np.imag(eigenvalues)
#
# index = omega_all > 1e-9
#
# omega = omega_all[index]
# eigvec_omega = eigvectors[:, index]
# perm = np.argsort(omega)
# eigvec_omega = eigvec_omega[:, perm]
#
# omega.sort()
#
# print(omega[:5])
