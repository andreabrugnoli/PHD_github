from firedrake import *
from system_components.plates import FloatingKP, FloatingBellKP, FloatingMP
from system_components.tests.kirchhoff_constants import *
from math import pi
import numpy as np
import scipy.linalg as la
from scipy.io import savemat
from modules_ph.classes_phsystem import SysPhdaeRig, check_positive_matrix


pointP = np.array([0.1, 0])
nC_point = 5
pointC1 = np.array([0.1*np.cos(pi/(2*nC_point)), 0.1*np.sin(pi/(2*nC_point))]).reshape((-1, 2))
pointC2 = np.array([0.1*np.cos(2*pi/(2*nC_point)), 0.1*np.sin(2*pi/(2*nC_point))]).reshape((-1, 2))
pointC3 = np.array([0.1*np.cos(3*pi/(2*nC_point)), 0.1*np.sin(3*pi/(2*nC_point))]).reshape((-1, 2))
pointC4 = np.array([0.1*np.cos(4*pi/(2*nC_point)), 0.1*np.sin(4*pi/(2*nC_point))]).reshape((-1, 2))
pointC5 = np.array([0.1*np.cos(5*pi/(2*nC_point)), 0.1*np.sin(5*pi/(2*nC_point))]).reshape((-1, 2))

pointsC = np.vstack((pointC1, pointC2, pointC3, pointC4, pointC5))

plate = FloatingKP(Lx, Ly, h, rho, E, nu, nx, ny, pointP, coord_C=pointsC, modes=False)
print(plate.n_f)
# plate = FloatingBellKP(Lx, Ly, h, rho, E, nu, nx, ny, pointP, pointC1, modes=False)

# pl_red = plate.reduce_system(0.001, 6)
pl_red = plate
J = pl_red.J
M = pl_red.E
B = pl_red.B

n_e = pl_red.n
n_lmb = pl_red.m
print(n_lmb, n_e)
n_rig = pl_red.n_r
G = B #np.zeros((n_e, n_lmb))
# G[:n_rig, :n_rig] = np.eye(n_lmb)
Z_lmb = np.zeros((n_lmb, n_lmb))
Z_al_lmb = np.zeros((n_e, n_lmb))
Z_u_lmb = np.zeros((n_lmb, n_e))

J_aug = np.vstack([np.hstack([J, G]),
                    np.hstack([-G.T, Z_lmb])])

E_aug = la.block_diag(M, Z_lmb)

# B_C = B[:, n_rig:]
#
# B_aug = np.concatenate((B_C, Z_lmb))

n_aug = n_e + n_lmb

# eigenvalues, eigvectors = la.eig(plate.J_f, plate.M_f)
eigenvalues, eigvectors = la.eig(J_aug, E_aug)
omega_all = np.imag(eigenvalues)

index = omega_all > 1e-9

omega = omega_all[index]
eigvec_omega = eigvectors[:, index]
perm = np.argsort(omega)
eigvec_omega = eigvec_omega[:, perm]

omega.sort()

print(omega[:8]/(2*pi))
