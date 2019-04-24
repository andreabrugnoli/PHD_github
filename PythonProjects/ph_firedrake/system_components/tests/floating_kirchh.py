from system_components.plates import FloatingKP
from system_components.tests.kirchhoff_constants import rho, E, Lx, Ly, h, nu, nx, ny
import numpy as np
from scipy.io import savemat
import scipy.linalg as la
import matplotlib.pyplot as plt


pointP = np.array([0, Ly/2])
pointC1 = np.array([Lx, Ly/2]).reshape((-1, 2))

plate = FloatingKP(Lx, Ly, h, rho, E, nu, 1, 2, pointP, coord_C=pointC1, modes=False)
plate.reduce_system(0.001, 4)

print(plate.n, plate.n_r, plate.n_p, plate.n_q, plate.m)

M = plate.M_e
print(np.linalg.matrix_rank(M), plate.n_e)
assert np.linalg.matrix_rank(M) == plate.n_e
J = plate.J_e
B = plate.B_e

# plt.figure(); plt.spy(M)
# plt.figure(); plt.spy(J)
# plt.figure(); plt.spy(B)
# plt.show()


Q_ode = la.inv(M)
J_ode = J
B_ode = B

pathout = '/home/a.brugnoli/GitProjects/MatlabProjects/PH/PH_TITOP/KirchhoffPlate/Matrices_Free/'
Qode_file = 'Q_pH'; Jode_file = 'J_pH'; Bode_file = 'B_pH'
savemat(pathout + Qode_file, mdict={Qode_file: Q_ode})
savemat(pathout + Jode_file, mdict={Jode_file: J_ode})
savemat(pathout + Bode_file, mdict={Bode_file: B_ode})

