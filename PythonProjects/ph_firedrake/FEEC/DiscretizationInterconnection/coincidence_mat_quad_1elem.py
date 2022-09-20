from firedrake import *

import numpy as np
np.set_printoptions(threshold=np.inf)

from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

tol = 1e-10

L_x = 1
L_y = 1

deg = 1
quad = False
mesh = RectangleMesh(1, 1, L_x, L_y, quadrilateral=quad)

triplot(mesh)

def curl2D(u):
    return as_vector([- u.dx(1), u.dx(0)])

def rot2D(u_vec):
    return u_vec[1].dx(0) - u_vec[0].dx(1)

if quad:
    P_0 = FiniteElement("CG", quadrilateral, deg)
    P_1 = FiniteElement("RTCE", quadrilateral, deg)
    P_1til = FiniteElement("RTCF", quadrilateral, deg)
    P_2 = FiniteElement("DG", quadrilateral, deg - 1)
else:
    P_0 = FiniteElement("CG", triangle, deg)
    P_1 = FiniteElement("N1curl", triangle, deg)
    P_1til = FiniteElement("RT", triangle, deg)
    P_2 = FiniteElement("DG", triangle, deg - 1)



V_0 = FunctionSpace(mesh, P_0)
V_1 = FunctionSpace(mesh, P_1)
V_1til = FunctionSpace(mesh, P_1til)
V_2 = FunctionSpace(mesh, P_2)


u_0 = TrialFunction(V_0)

v_1 = TestFunction(V_1)
u_1 = TrialFunction(V_1)

v_1til = TestFunction(V_1til)
u_1til = TrialFunction(V_1til)

v_2 = TestFunction(V_2)
u_2 = TrialFunction(V_2)


# Construction of the co-incidence matrices for the curl2D div complex (outer oriented)
# The D_0 case
m_1til = dot(v_1til, u_1til) * dx
d_10til = dot(v_1til, curl2D(u_0)) * dx

M1til_petsc = assemble(m_1til, mat_type='aij')
D10til_petsc = assemble(d_10til, mat_type='aij')

M1til_mat = csr_matrix(M1til_petsc.M.handle.getValuesCSR()[::-1])
D10til_mat = csr_matrix(D10til_petsc.M.handle.getValuesCSR()[::-1])

M1til_mat.tocsc()
D10til_mat.tocsc()

Dtil_0 = spsolve(M1til_mat, D10til_mat)
Dtil_0.tolil()


print("Dtil_0")
print(Dtil_0.shape)

Dtil_0[abs(Dtil_0) < tol] = 0.0
# print(Dtil_0[abs(Dtil_0)>tol])
print(Dtil_0)

# The D_1 case
m_2 = dot(v_2, u_2) * dx
d_21til = dot(v_2, div(u_1til)) * dx

M2_petsc = assemble(m_2, mat_type='aij' )
D21til_petsc = assemble(d_21til, mat_type='aij' )

M2_mat = csr_matrix(M2_petsc.M.handle.getValuesCSR()[::-1])
D21til_mat = csr_matrix(D21til_petsc.M.handle.getValuesCSR()[::-1])

M2_mat.tocsc()
D21til_mat.tocsc()

Dtil_1 = spsolve(M2_mat, D21til_mat)
Dtil_1.tolil()

print("Dtil_1")
print(Dtil_1.shape)

Dtil_1[abs(Dtil_1) < tol] = 0.0
# print(Dtil_1[abs(Dtil_1)>tol])
print(Dtil_1)


# # Construction of the co-incidence matrices for the grad rot2D complex (inner oriented)
# # The D_0 case
# m_1 = dot(v_1, u_1) * dx
# d_10 = dot(v_1, grad(u_0)) * dx
#
# M1_petsc = assemble(m_1, mat_type='aij' )
# D10_petsc = assemble(d_10, mat_type='aij' )
#
# M1_mat = csr_matrix(M1_petsc.M.handle.getValuesCSR()[::-1])
# D10_mat = csr_matrix(D10_petsc.M.handle.getValuesCSR()[::-1])
#
# M1_mat.tocsc()
# D10_mat.tocsc()
#
# D_0 = spsolve(M1_mat, D10_mat)
# D_0.tolil()
#
# print("D_0")
# print(D_0.shape)
# print(D_0)
#
# # D_0[abs(D_0) < tol] = 0.0
# # print(D_0[abs(D_0)>tol])
#
# # The D_1 case
# d_21 = dot(v_2, rot2D(u_1)) * dx
#
# D21_petsc = assemble(d_21, mat_type='aij' )
#
# D21_mat = csr_matrix(D21_petsc.M.handle.getValuesCSR()[::-1])
#
# D21_mat.tocsc()
#
# D_1 = spsolve(M2_mat, D21_mat)
# D_1.tolil()
#
# print("D_1")
# print(D_1.shape)
# print(D_1)
#
# # D_1[abs(D_1) < tol] = 0.0
# # print(D_1[abs(D_1)>tol])
#
#
# # Plot unitary finite element
#
#
#


