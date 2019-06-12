from firedrake import *
from system_components.plates import FloatingKP
from system_components.tests.kirchhoff_constants import rho, E, Lx, Ly, h, nu, nx, ny
import numpy as np
from scipy.io import savemat
import scipy.linalg as la
import matplotlib.pyplot as plt

mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=False)
tab_coord = mesh.coordinates.dat.data

pointP = np.array([0, 0])
pointC1 = np.array([Lx, 0]).reshape((-1, 2))
pointC2 = np.array([Lx, Ly/4]).reshape((-1, 2))
pointC3 = np.array([Lx, 3*Ly/4]).reshape((-1, 2))
pointC4 = np.array([Lx, Ly]).reshape((-1, 2))

pointsC = np.vstack((pointC1, pointC2, pointC3, pointC4))

plate = FloatingKP(Lx, Ly, h, rho, E, nu, nx, ny, pointP, coord_C=pointC1, modes=False)

pathout = '/home/a.brugnoli/GitProjects/MatlabProjects/PH/ReductionPHDAE/KP_Matrices/'
Mode_file = 'M'; Jode_file = 'J'; Bode_file = 'B'
# savemat(pathout + Mode_file, mdict={Mode_file: plate.M_e})
# savemat(pathout + Jode_file, mdict={Jode_file: plate.J_e})
# savemat(pathout + Bode_file, mdict={Bode_file: plate.B_e})

plate = plate.reduce_system(0.001, 10)[0]
Mode_file = 'Mr'; Jode_file = 'Jr'; Bode_file = 'Br'
# savemat(pathout + Mode_file, mdict={Mode_file: plate.M_e})
# savemat(pathout + Jode_file, mdict={Jode_file: plate.J_e})
# savemat(pathout + Bode_file, mdict={Bode_file: plate.B_e})


