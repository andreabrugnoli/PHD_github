from firedrake import *
from system_components.plates import FloatingKP, FloatingBellKP
from system_components.tests.kirchhoff_constants import *
import numpy as np
import scipy.linalg as la
from scipy.io import savemat
from modules_phdae.classes_phsystem import SysPhdaeRig

pointP = np.array([0, Ly/2])
pointC1 = np.array([0, 5*Ly/6]).reshape((-1, 2))

plate = FloatingKP(Lx, Ly, h, rho, E, nu, nx, ny, pointP, pointC1, modes=True)
# plate = FloatingBellKP(Lx, Ly, h, rho, E, nu, nx, ny, pointP, pointC1, modes=True)

J = plate.J
M = plate.E
B = plate.B
n_e = plate.n
n_lmb = 3
n_rig = plate.n_r
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

B_aug = np.concatenate((B_C, Z_lmb))

n_aug = n_e + n_lmb

eigenvalues, eigvectors = la.eig(J_aug, E_aug)
omega_all = np.imag(eigenvalues)
index = omega_all > 0
omega = omega_all[index]
eigvec_omega = eigvectors[:, index]
perm = np.argsort(omega)
eigvec_omega = eigvec_omega[:, perm]
omega.sort()

print(omega[:10])

# plate_dae = SysPhdaeRig(n_aug, n_lmb, n_rig, plate.n_p, plate.n_q, E=E_aug, J=J_aug, B=B_aug)
#
# pathout = '/home/a.brugnoli/GitProjects/MatlabProjects/PH/PH_TITOP/KirchhoffPlate/Matrices_Clamped/'
# Edae_file = 'E_dae'; Jdae_file = 'J_dae'; Bdae_file = 'B_dae'
# savemat(pathout + Edae_file, mdict={Edae_file: plate_dae.E})
# savemat(pathout + Jdae_file, mdict={Jdae_file: plate_dae.J})
# savemat(pathout + Bdae_file, mdict={Bdae_file: plate_dae.B})
# #
# # plate_ode, T = plate_dae.dae_to_ode()
# # Jode_file = 'J_ode'; Qode_file = 'Q_ode'; Bode_file = 'B_ode'
# # savemat(pathout + Jode_file, mdict={Jode_file: plate_ode.J})
# # savemat(pathout + Qode_file, mdict={Qode_file: plate_ode.Q})
# # savemat(pathout + Bode_file, mdict={Bode_file: plate_ode.B[:, 3:]})