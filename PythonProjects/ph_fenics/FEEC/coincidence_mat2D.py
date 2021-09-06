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

# mesh = RectangleMesh(Point(0, 0), Point(L, L), n_el, n_el)
domain = mshr.Rectangle(Point(0,0,0), Point(L,L,L))
mesh = mshr.generate_mesh(domain, n_el)

plot(mesh)
plt.show()


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

u_0 = TrialFunction(V_0)

v_1 = TestFunction(V_1)
u_1 = TrialFunction(V_1)

v_2 = TestFunction(V_2)
u_2 = TrialFunction(V_2)


# Construction of the D_0 co-incidence matrix
m_1 = dot(v_1, u_1) * dx
# d_10 = dot(v_1, grad(u_0)) * dx
d_10 = dot(v_1, curl2D(u_0)) * dx

M1_petsc = PETScMatrix()
D10_petsc = PETScMatrix()

assemble(m_1, M1_petsc)
assemble(d_10, D10_petsc)

M1_mat = csr_matrix(M1_petsc.mat().getValuesCSR()[::-1])
D10_mat = csr_matrix(D10_petsc.mat().getValuesCSR()[::-1])

M1_mat.tocsc()
D10_mat.tocsc()

D_0 = spsolve(M1_mat, D10_mat)
D_0.tolil()

D_0[abs(D_0) < tol] = 0.0
print("D_0")
print(D_0)
# print(D_0[abs(D_0)>tol])

# Construction of the D_1 co-incidence matrix
m_2 = dot(v_2, u_2) * dx
# d_21 = dot(v_2, rot2D(u_1)) * dx
d_21 = dot(v_2, div(u_1)) * dx

M2_petsc = PETScMatrix()
D21_petsc = PETScMatrix()

assemble(m_2, M2_petsc)
assemble(d_21, D21_petsc)

M2_mat = csr_matrix(M2_petsc.mat().getValuesCSR()[::-1])
D21_mat = csr_matrix(D21_petsc.mat().getValuesCSR()[::-1])

M2_mat.tocsc()
D21_mat.tocsc()

D_1 = spsolve(M2_mat, D21_mat)
D_1.tolil()

D_1[abs(D_1) < tol] = 0.0
print("D_1")
# print(D_1)
print(D_1[abs(D_1)>tol])
