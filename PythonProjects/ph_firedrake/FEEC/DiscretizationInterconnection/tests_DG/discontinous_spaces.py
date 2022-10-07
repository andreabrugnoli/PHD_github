from firedrake import *
from tools_plotting import setup
import numpy as np
np.set_printoptions(threshold=np.inf)
import os
os.environ["OMP_NUM_THREADS"] = "1"

from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

from FEEC.DiscretizationInterconnection.triangle_mesh import create_reference_triangle
from matplotlib import pyplot as plt

L_x = 1
L_y = 1

tol = 1e-10

deg = 1


def curl2D(u):
    return as_vector([- u.dx(1), u.dx(0)])

def rot2D(u_vec):
    return u_vec[1].dx(0) - u_vec[0].dx(1)

mesh = RectangleMesh(1, 1, L_x, L_y)
x, y = SpatialCoordinate(mesh)

n_ver = FacetNormal(mesh)

# triplot(mesh)
# plt.show()
P0 = FiniteElement("CG", triangle, deg)
P1 = FiniteElement("N1curl", triangle, deg)
P1til = FiniteElement("RT", triangle, deg)
# P1 = FiniteElement("N1curl", triangle, deg, variant='integral')
# P1til = FiniteElement("RT", triangle, deg, variant='integral')
P2 = FiniteElement("DG", triangle, deg - 1)


P0_b = BrokenElement(P0)
P1_b = BrokenElement(P1)
P1til_b = BrokenElement(P1til)
P2_b = BrokenElement(P2)

V0_b = FunctionSpace(mesh, P0_b)
V1_b = FunctionSpace(mesh, P1_b)
V1til_b = FunctionSpace(mesh, P1til_b)
V2_b = FunctionSpace(mesh, P2_b)

V0 = FunctionSpace(mesh, P0)
V1 = FunctionSpace(mesh, P1)
V1til = FunctionSpace(mesh, P1til)
V2 = FunctionSpace(mesh, P2)

dx = Measure('dx')
ds = Measure('ds')

v0_b = TestFunction(V0_b)
u0_b = TrialFunction(V0_b)

v1_b = TestFunction(V1_b)
u1_b = TrialFunction(V1_b)

v1til_b = TestFunction(V1til_b)
u1til_b = TrialFunction(V1til_b)

v2_b = TestFunction(V2_b)
u2_b = TrialFunction(V2_b)

f0_b = Function(V0_b)
f1_b = Function(V1_b)
f1til_b = Function(V1til_b)
f2_b = Function(V2_b)

f2_b.project(x*y**2)
f0_b.project(x*y**2)

# * 1: plane x == 0
# * 2: plane x == Lx
# * 3: plane y == 0
# * 4: plane y == Ly

# b_L = v0_b * dot(u1til_b, n_ver) * ds(1)
# petsc_BL = assemble(b_L, mat_type='aij').M.handle
# B_L = np.array(petsc_BL.convert("dense").getDenseArray())
# B_L[abs(B_L) < tol] = 0.0
# dofs2_L = np.where(B_L.any(axis=0))[0]
#
# print("Left boundary matrix")
# B_L = B_L[:, dofs2_L]
# print(B_L)
#
# b_R = v0_b * dot(u1til_b, n_ver) * ds(2)
# petsc_BR = assemble(b_R, mat_type='aij').M.handle
# B_R = np.array(petsc_BR.convert("dense").getDenseArray())
# B_R[abs(B_R) < tol] = 0.0
# dofs2_R = np.where(B_R.any(axis=0))[0]
#
# print("Right boundary matrix")
# B_R = B_R[:, dofs2_R]
# print(B_R)
#
#
#
# n0_b = V0_b.dim()
# n1_b = V1_b.dim()
# n1til_b = V1til_b.dim()
#
# print('Dim V0_b')
# print(n0_b)
# print('Dim V1_b')
# print(n1_b)
# print('Dim V0_b')
# print(n1til_b)


# for i in range(n0_b):
#     zeros_n0 = np.zeros((n0_b,))
#     zeros_n0[i] =1
#
#     f0_b.vector().set_local(zeros_n0)
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     trisurf(f0_b, axes=ax)
# plt.show()

# for i in range(n1_b):
#     zeros_n1 = np.zeros((n1_b,))
#     zeros_n1[i] = 1
#
#     f1_b.vector().set_local(zeros_n1)
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_xlabel('x')
#     ax.set_xlim(-2*L_x, 3* L_x)
#     ax.set_ylim(-2*L_y, 3*L_y)
#
#     quiver(f1_b, axes=ax)
# plt.show()
#
#
# for i in range(n1til_b):
#     zeros_n1til = np.zeros((n1til_b,))
#     zeros_n1til[i] = 1
#
#     f1til_b.vector().set_local(zeros_n1til)
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#
#     ax.set_xlim(-2*L_x, 3* L_x)
#     ax.set_ylim(-2*L_y, 3*L_y)
#     quiver(f1til_b, axes=ax)
#
# plt.show()

# # Construction of the co-incidence matrices for the curl2D div complex (outer oriented) on discontinous spaces
# # The D_0 case
# m1til_b = dot(v1til_b, u1til_b) * dx
# d10til_b = dot(v1til_b, curl2D(u0_b)) * dx
#
# M1til_b_petsc = assemble(m1til_b, mat_type='aij')
# D10til_b_petsc = assemble(d10til_b, mat_type='aij')
#
# M1til_b_mat = csr_matrix(M1til_b_petsc.M.handle.getValuesCSR()[::-1])
# D10til_b_mat = csr_matrix(D10til_b_petsc.M.handle.getValuesCSR()[::-1])
#
# M1til_b_mat.tocsc()
# D10til_b_mat.tocsc()
#
# D0til_b = spsolve(M1til_b_mat, D10til_b_mat)
# D0til_b.tolil()
#
#
# print("D0til_b")
# print(D0til_b.shape)
#
# D0til_b[abs(D0til_b) < tol] = 0.0
# # print(Dtil_0[abs(Dtil_0)>tol])
# print(D0til_b)
#
# # The D_1 case
# m2_b = dot(v2_b, u2_b) * dx
# d21til_b = dot(v2_b, div(u1til_b)) * dx
#
# M2_b_petsc = assemble(m2_b, mat_type='aij' )
# D21til_b_petsc = assemble(d21til_b, mat_type='aij' )
#
# M2_b_mat = csr_matrix(M2_b_petsc.M.handle.getValuesCSR()[::-1])
# D21til_b_mat = csr_matrix(D21til_b_petsc.M.handle.getValuesCSR()[::-1])
#
# M2_b_mat.tocsc()
# D21til_b_mat.tocsc()
#
# D1til_b = spsolve(M2_b_mat, D21til_b_mat)
# D1til_b.tolil()
#
# print("D1til_b")
# print(D1til_b.shape)
#
# D1til_b[abs(D1til_b) < tol] = 0.0
# # print(Dtil_1[abs(Dtil_1)>tol])
# print(D1til_b)


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


