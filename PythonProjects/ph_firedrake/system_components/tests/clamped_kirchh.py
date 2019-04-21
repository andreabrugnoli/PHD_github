from firedrake import *
from system_components.thin_components import FloatingKP
from system_components.tests.kirchhoff_constants import rho, E, Lx, Ly, h, nu, nx, ny
import numpy as np
from scipy.io import savemat
from modules_phdae.classes_phsystem import SysPhdaeRig

mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=False)
tab_coord = mesh.coordinates.dat.data

pointP = np.array([0, 0])
pointC1 = np.array([Lx, 0]).reshape((-1, 2))


plate = FloatingKP(Lx, Ly, h, rho, E, nu, nx, ny, pointP, pointC1, modes=False)

J = plate.J_e
M = plate.M_e
B = plate.B_e
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

B_e = np.concatenate((np.zeros_like(B_C), B_C), axis=1)
B_lmb = np.concatenate((np.eye(n_lmb), Z_lmb), axis=1)

B_aug = np.concatenate((B_e, B_lmb))

n_aug = n_e + n_lmb

plate_dae = SysPhdaeRig(n_aug, n_lmb, n_rig, plate.n_p, plate.n_q, E=E_aug, J=J_aug, B=B_aug)

pathout = '/home/a.brugnoli/GitProjects/MatlabProjects/PH/PH_TITOP/KirchhoffPlate/Matrices_Clamped/'
Edae_file = 'E_dae'; Jdae_file = 'J_dae'; Bdae_file = 'B_dae'
savemat(pathout + Edae_file, mdict={Edae_file: plate_dae.E})
savemat(pathout + Jdae_file, mdict={Jdae_file: plate_dae.J})
savemat(pathout + Bdae_file, mdict={Bdae_file: plate_dae.B})
#
# plate_ode, T = plate_dae.dae_to_ode()
# Jode_file = 'J_ode'; Qode_file = 'Q_ode'; Bode_file = 'B_ode'
# savemat(pathout + Jode_file, mdict={Jode_file: plate_ode.J})
# savemat(pathout + Qode_file, mdict={Qode_file: plate_ode.Q})
# savemat(pathout + Bode_file, mdict={Bode_file: plate_ode.B[:, 3:]})