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

# mesh = Mesh("/home/andrea/cube.msh")
mesh = CubeMesh(n_el, n_el, n_el, L)
# triplot(mesh)
# plt.show()

P_0 = FiniteElement("CG", tetrahedron, deg)
P_1 = FiniteElement("N1curl", tetrahedron, deg, variant='point')
P_2 = FiniteElement("RT", tetrahedron, deg, variant='integral')
P_3 = FiniteElement("DG", tetrahedron, deg-1)

V_0 = FunctionSpace(mesh, P_0)
V_1 = FunctionSpace(mesh, P_1)
V_2 = FunctionSpace(mesh, P_2)
V_3 = FunctionSpace(mesh, P_3)


u_0 = Function(V_0)
u_1 = Function(V_1)
u_2 = Function(V_2)
u_3 = Function(V_3)

x, y, z = SpatialCoordinate(mesh)

f_0 = sin(x)
f_1 = as_vector([sin(x), sin(y), sin(z)])
interpolate(f_1, u_1)
interpolate(curl(u_1), u_2)

# interpolate(u_2, u_1)


# interpolate(sin(x), u_0)
# u_00 = Function(V_0)
# u_00.assign(u_0)
# interpolate(grad(sin(x)), u_1)
# interpolate(grad(u_0), u_1)
# u_1.assign(grad(u_0))

