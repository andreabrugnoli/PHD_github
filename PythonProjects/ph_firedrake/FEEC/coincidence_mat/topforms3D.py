from firedrake import *

import numpy as np
np.set_printoptions(threshold=np.inf)

from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

tol = 1e-10

L = 1
n_el = 2
deg = 1

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

print("number of vertices " + str(V_0.dim()))
print("number of edges " + str(V_1.dim()))
print("number of faces " + str(V_2.dim()))
print("number of volumes " + str(V_3.dim()))


u_3 = interpolate(Constant(1), V_3)

print(u_3.vector().get_local())