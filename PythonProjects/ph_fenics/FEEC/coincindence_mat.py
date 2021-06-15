from fenics import *
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from vedo.dolfin import plot

mesh = UnitCubeMesh(1, 1, 1)

# mesh_plot = plot(mesh, mode="mesh", interactive=0)
#
# vmesh = mesh_plot.actors[0].lineWidth(0)
# vmesh.cutWithPlane(origin=(0,0,0), normal=(1,-1,0))
# plot(vmesh, interactive=1)

V_0 = FunctionSpace(mesh, "CG", 1)
V_1 = FunctionSpace(mesh, "N1curl", 1)
V_2 = FunctionSpace(mesh, "RT", 1)
V_3 = FunctionSpace(mesh, "DG", 0)

u_0 = TrialFunction(V_0)

v_1 = TestFunction(V_1)
u_1 = TrialFunction(V_1)

v_2 = TestFunction(V_2)
u_2 = TrialFunction(V_2)

v_3 = TestFunction(V_3)
u_3 = TrialFunction(V_3)

# Construction of the D_0 co-incidence matrix
m_1 = dot(v_1, u_1) * dx
d_10 = dot(v_1, grad(u_0)) * dx

M1_petsc = assemble(m_1)
D10_petsc = assemble(d_10)

M1_mat = M1_petsc.array()
D10_mat = D10_petsc.array()

D_0 = la.solve(M1_mat, D10_mat)

tol = 1e-14
D_0[abs(D_0) < tol] = 0.0

print(D_0.shape)
print(D_0)

# Construction of the D_1 co-incidence matrix
m_2 = dot(v_2, u_2) * dx
d_21 = dot(v_2, curl(u_1)) * dx

M2_petsc = assemble(m_2)
D21_petsc = assemble(d_21)

M2_mat = M2_petsc.array()
D21_mat = D21_petsc.array()

D_1 = la.solve(M2_mat, D21_mat)

D_1[abs(D_1) < tol] = 0.0

print(D_1.shape)
print(D_1/2)

# Construction of the D_3 co-incidence matrix
m_3 = v_3 * u_3 * dx
d_32 = v_3 * div(u_2) * dx

M3_petsc = assemble(m_3)
D32_petsc = assemble(d_32)

M3_mat = M3_petsc.array()
D32_mat = D32_petsc.array()

D_2 = la.solve(M3_mat, D32_mat)

D_2[abs(D_2) < tol] = 0.0

print(D_2.shape)
print(D_2/3)