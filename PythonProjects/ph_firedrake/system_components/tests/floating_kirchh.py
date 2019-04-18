from firedrake import *
from system_components.thin_components import find_point, FloatingKP
from system_components.tests.kirchhoff_constants import rho, E, Lx, Ly, h, nu, nx, ny
import numpy as np
from scipy.io import savemat
import scipy.linalg as la

mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=False)
tab_coord = mesh.coordinates.dat.data

pointP = np.array([0, 3*Ly/4])
pointC1 = np.array([Lx, Ly]).reshape((-1, 2))

pointC2 = np.array([Lx, 5*Ly/6]).reshape((-1, 2))

pointsC = np.vstack((pointC1, pointC2))

i_P, dist_P = find_point(tab_coord, pointP)
i_C1, dist_C1 = find_point(tab_coord, pointC1)
i_C2, dist_C2 = find_point(tab_coord, pointC2)

print("Point P found" + str(tab_coord[i_P]) + " with distance " + str(dist_P))
print("Point C1 found" + str(tab_coord[i_C1]) + " with distance " + str(dist_C1))
print("Point C2 found" + str(tab_coord[i_C2]) + " with distance " + str(dist_C2))

plate = FloatingKP(Lx, Ly, h, rho, E, nu, nx, ny, pointP, pointC1, modes=True)

print(plate.n, plate.n_r, plate.n_p, plate.n_q, plate.m)

M = plate.M_e
J = plate.J_e
B = plate.B_e

Q_ode = la.inv(M)
J_ode = J
B_ode = B

pathout = '/home/a.brugnoli/GitProjects/MatlabProjects/PH/PH_TITOP/KirchhoffPlate/Matrices/'
Qode_file = 'Q_pH'; Jode_file = 'J_pH'; Bode_file = 'B_pH'
savemat(pathout + Qode_file, mdict={Qode_file: Q_ode})
savemat(pathout + Jode_file, mdict={Jode_file: J_ode})
savemat(pathout + Bode_file, mdict={Bode_file: B_ode})

