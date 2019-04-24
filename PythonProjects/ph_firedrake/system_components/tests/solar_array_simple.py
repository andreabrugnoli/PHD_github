from system_components.plates import FloatingKP
from system_components.tests.kirchhoff_constants import rho, E, Lx, Ly, h, nu, nx, ny
import numpy as np
from scipy.io import savemat
import scipy.linalg as la
import matplotlib.pyplot as plt

pointP1 = np.array([0, Ly/2])

pointC1 = np.array([Lx, 0]).reshape((-1, 2))
pointC2 = np.array([Lx, Ly]).reshape((-1, 2))
pointsC_1 = np.vstack((pointC1, pointC2))

plate_1 = FloatingKP(Lx, Ly, h, rho, E, nu, 1, 2, pointP1, coord_C=pointsC_1, modes=False)

pointP2 = np.array([0, 0])

pointC3 = np.array([0, Ly]).reshape((-1, 2))
pointC4 = np.array([Lx, 0]).reshape((-1, 2))
pointC5 = np.array([Lx, Ly]).reshape((-1, 2))
pointsC_2 = np.vstack((pointC3, pointC4, pointC5))

plate_2 = FloatingKP(Lx, Ly, h, rho, E, nu, 1, 1, pointP2, coord_C=pointsC_2, modes=False)

pointP3 = np.array([0, 0])

pointC6 = np.array([0, Ly]).reshape((-1, 2))
pointsC_3 = pointC6

plate_3 = FloatingKP(Lx, Ly, h, rho, E, nu, 1, 1, pointP3, coord_C=pointsC_3, modes=False)


ind1 = list(range(3, 9))
ind2 = list(range(6))
plate_12 = plate_1.transformer_ordered(plate_2, ind1, ind2, np.eye(6))

solar_array = plate_12.transformer_ordered(plate_3, ind1, ind2, np.eye(6))

G_int = solar_array.G_e
print(np.linalg.matrix_rank(G_int), solar_array.n_lmb)
nrig_tot = solar_array.n_r
G_r = G_int[:nrig_tot]
print(G_r)
print(np.linalg.matrix_rank(G_r))


sys_SA = solar_array.dae_to_ode()[0]

pathout = '/home/a.brugnoli/GitProjects/MatlabProjects/PH/PH_TITOP/KirchhoffPlate/MatricesSASimple/'
Qode_file = 'Q_pH'; Jode_file = 'J_pH'; Bode_file = 'B_pH'
savemat(pathout + Qode_file, mdict={Qode_file: sys_SA.Q})
savemat(pathout + Jode_file, mdict={Jode_file: sys_SA.J})
savemat(pathout + Bode_file, mdict={Bode_file: sys_SA.B})
