from firedrake import *

import numpy as np
np.set_printoptions(threshold=np.inf)

from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

tol = 1e-10

L = 1
n_el = 1
deg = 1

mesh = RectangleMesh(n_el, n_el, L, L)
triplot(mesh)
plt.show()
# triplot(mesh)
# plt.show()

def curl2D(u):
    return as_vector([- u.dx(1), u.dx(0)])

def rot2D(u_vec):
    return u_vec[1].dx(0) - u_vec[0].dx(1)


P_0 = FiniteElement("CG", triangle, deg)
P_1 = FiniteElement("N1curl", triangle, deg)
# P_1 = FiniteElement("RT", triangle, deg)
P_2 = FiniteElement("DG", triangle, deg-1)

V_0 = FunctionSpace(mesh, P_0)
V_1 = FunctionSpace(mesh, P_1)
V_2 = FunctionSpace(mesh, P_2)


vec = Constant((1, .5))

vec_h = interpolate(vec, V_1)

print(vec_h.vector().get_local())


u_2 = interpolate(Constant(1), V_2)

print(u_2.vector().get_local())
