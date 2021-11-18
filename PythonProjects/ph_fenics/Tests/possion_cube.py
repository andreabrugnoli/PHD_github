from fenics import *

import numpy as np
np.set_printoptions(threshold=np.inf)

from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from tools_plotting import setup

import os
os.environ["OMP_NUM_THREADS"] = "1"

# from vedo.dolfin import plot

tol = 1e-10

L = 1
n_el = 1
deg = 3

mesh = Mesh("/home/andrea/cube.xml")

# mesh_plot = plot(mesh) # mode="mesh", interactive=0
# plt.show()

f = Expression("10*(x[0]+x[1]-1+x[2]-1)", degree = 2)

# f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree = 2)
V = FunctionSpace(mesh, "CG", 2)
u = Function(V)
u.interpolate(f)
c = plot(u)
plt.colorbar(c)
plt.show()
