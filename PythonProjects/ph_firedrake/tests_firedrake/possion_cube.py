from firedrake import *

import numpy as np
np.set_printoptions(threshold=np.inf)

from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from tools_plotting import setup

import os
os.environ["OMP_NUM_THREADS"] = "1"

tol = 1e-10

L = 1
n_el = 1
deg = 3

mesh = Mesh("/home/andrea/Meshes/cube.msh")
triplot(mesh)
x = SpatialCoordinate(mesh)

# V = FunctionSpace(mesh, "CG", 2)
# u = Function(V)
#
# f = sin(x[0]*pi)*sin(2*x[1]*pi)
# u.assign(f)

# triplot(u)

plt.show()