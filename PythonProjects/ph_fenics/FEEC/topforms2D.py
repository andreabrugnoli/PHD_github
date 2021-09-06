from fenics import *
import mshr

import numpy as np
np.set_printoptions(threshold=np.inf)

from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

tol = 1e-10

L = 1
n_el = 1
deg = 1

mesh = RectangleMesh(Point(0, 0), Point(L, L), n_el, n_el)

# domain = mshr.Square(Point(0,0,0), Point(L,L,L))
# mesh = mshr.generate_mesh(domain, n_el)

# plot(mesh)
# plt.show()


def curl2D(u):
    return as_vector([- u.dx(1), u.dx(0)])

def rot2D(u_vec):
    return u_vec[1].dx(0) - u_vec[0].dx(1)

# V_0 = FunctionSpace(mesh, "P- Lambda", deg, 0)
# V_1 = FunctionSpace(mesh, "P- Lambda", deg, 1)  # Equivalent to Hcurl
# V_2 = FunctionSpace(mesh, "P- Lambda", deg, 2)
#
# P_0 = FiniteElement("CG", triangle, deg, variant='point')
# # P_1 = FiniteElement("N1curl", triangle, deg, variant='integral')
# P_1 = FiniteElement("RT", triangle, deg, variant='integral')
# P_2 = FiniteElement("DG", triangle, deg-1, variant='point')


P_0 = FiniteElement("CG", triangle, deg, variant='feec')
# P_1 = FiniteElement("N1curl", triangle, deg, variant='feec')
P_1 = FiniteElement("RT", triangle, deg, variant='feec')
P_2 = FiniteElement("DG", triangle, deg-1, variant='feec')

V_0 = FunctionSpace(mesh, P_0)
V_1 = FunctionSpace(mesh, P_1)
V_2 = FunctionSpace(mesh, P_2)

u_2 = interpolate(Constant(1), V_2)

print(u_2.vector().get_local())
