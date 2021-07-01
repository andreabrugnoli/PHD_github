from firedrake import *

import numpy as np
np.set_printoptions(threshold=np.inf)

from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

tol = 1e-10

L = 10
n_el = 2
deg = 1

mesh_int = IntervalMesh(n_el, L)
mesh = ExtrudedMesh(mesh_int, n_el)


def curl2D(u):
    return as_vector([- u.dx(1), u.dx(0)])

def rot2D(u_vec):
    return u_vec[1].dx(0) - u_vec[0].dx(1)


CG_deg1 = FiniteElement("CG", interval, deg)
DG_deg1 = FiniteElement("DG", interval, deg)

DG_deg = FiniteElement("DG", interval, deg - 1)

P_CG1_DG = TensorProductElement(CG_deg1, DG_deg)
P_DG_CG1 = TensorProductElement(DG_deg, CG_deg1)

RT_horiz = HDivElement(P_CG1_DG)
RT_vert = HDivElement(P_DG_CG1)
RT_quad = RT_horiz + RT_vert

Ned_horiz = HCurlElement(P_CG1_DG)
Ned_vert = HCurlElement(P_DG_CG1)
Ned_quad = Ned_horiz + Ned_vert

# P_CG1_DG1 = TensorProductElement(CG_deg1, DG_deg1)
# P_DG1_CG1 = TensorProductElement(DG_deg1, CG_deg1)
#
# BDM_horiz = HDivElement(P_CG1_DG1)
# BDM_vert = HDivElement(P_DG1_CG1)
# BDM_quad = BDM_horiz + BDM_vert


V_0 = FunctionSpace(mesh, "CG", deg)

# V_1 = FunctionSpace(mesh, RT_quad, deg)
V_1 = FunctionSpace(mesh, Ned_quad, deg)

V_2 = FunctionSpace(mesh, "DG", deg-1)


u_0 = TrialFunction(V_0)

v_1 = TestFunction(V_1)
u_1 = TrialFunction(V_1)

v_2 = TestFunction(V_2)
u_2 = TrialFunction(V_2)


# Construction of the D_0 co-incidence matrix
m_1 = dot(v_1, u_1) * dx
d_10 = dot(v_1, grad(u_0)) * dx
# d_10 = dot(v_1, curl2D(u_0)) * dx

M1_petsc = assemble(m_1, mat_type='aij' )
D10_petsc = assemble(d_10, mat_type='aij' )

M1_mat = csr_matrix(M1_petsc.M.handle.getValuesCSR()[::-1])
D10_mat = csr_matrix(D10_petsc.M.handle.getValuesCSR()[::-1])

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
d_21 = dot(v_2, rot2D(u_1)) * dx
# d_21 = dot(v_2, div(u_1)) * dx

M2_petsc = assemble(m_2, mat_type='aij' )
D21_petsc = assemble(d_21, mat_type='aij' )

M2_mat = csr_matrix(M2_petsc.M.handle.getValuesCSR()[::-1])
D21_mat = csr_matrix(D21_petsc.M.handle.getValuesCSR()[::-1])

M2_mat.tocsc()
D21_mat.tocsc()

D_1 = spsolve(M2_mat, D21_mat)
D_1.tolil()

D_1[abs(D_1) < tol] = 0.0
print("D_1")
# print(D_1)
print(D_1[abs(D_1)>tol])