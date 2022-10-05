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


def j_p_32(v3, p3, v2, u2):
    j_p_32 = dot(v3, div(u2)) * dx - dot(div(v2), p3) * dx

    return j_p_32

def j_d_10(v1, u1, v0, p0):
    j_d_10 = dot(v1, grad(p0)) * dx - dot(grad(v0), u1) * dx

    return j_d_10

def j_int(v2_b, p0_b, v0_b, u2_b):
    j_int_20 = (dot(v2_b('+'), n_ver('+')) * p0_b('-') + dot(v2_b('-'), n_ver('-')) * p0_b('+'))*dS
    j_int_02 = - (v0_b('+') * dot(u2_b('-'), n_ver('-')) + v0_b('-') * dot(u2_b('+'), n_ver('+')))*dS

    return j_int_20 + j_int_02

def j_int_p(v2_b, p0_b, v0_b, u2_b):
    j_int_p = (dot(v2_b('+'), n_ver('+')) * p0_b('-') - v0_b('-') * dot(u2_b('+'), n_ver('+'))) * dS

    return j_int_p


def j_int_d(v2_b, p0_b, v0_b, u2_b):
    j_int_d = (-v0_b('+') * dot(u2_b('-'), n_ver('-')) + dot(v2_b('-'), n_ver('-')) * p0_b('+')) * dS

    return j_int_d

mesh = RectangleMesh(1, 1, L_x, L_y)
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
V2_b = FunctionSpace(mesh, P1til_b)
V3_b = FunctionSpace(mesh, P2_b)

V_b = V0_b * V1_b * V3_b * V2_b

dx = Measure('dx')
ds = Measure('ds')
dS = Measure('dS')

v_b = TestFunction(V_b)
v0_b, v1_b, v3_b, v2_b = split(v_b)

e_b = TrialFunction(V_b)
e0_b, e1_b, e3_b, e2_b = split(e_b)

f_b = Function(V_b)
f0_b, f1_b, f3_b, f2_b = split(f_b)

jform_p_32 = j_p_32(v3_b, e3_b, v2_b, e2_b)
jform_d_10 = j_d_10(v1_b, e1_b, v0_b, e0_b)
jform_int_20 = j_int(v2_b, e0_b, v0_b, e2_b)

jform_int_p_20 = j_int_p(v2_b, e0_b, v0_b, e2_b)
jform_int_d_20 = j_int_d(v2_b, e0_b, v0_b, e2_b)

j_pd = jform_p_32 + jform_d_10
j_pd_int = jform_p_32 + jform_d_10 + jform_int_20

j_pd_int_p = jform_p_32 + jform_d_10 + jform_int_p_20
j_pd_int_d = jform_p_32 + jform_d_10 + jform_int_d_20

J_pd_petsc = assemble(j_pd, mat_type='aij').M.handle
J_pd = np.array(J_pd_petsc.convert("dense").getDenseArray())

print(np.linalg.norm(J_pd + np.transpose(J_pd)))

J_pd_int_petsc = assemble(j_pd_int, mat_type='aij').M.handle
J_pd_int = np.array(J_pd_int_petsc.convert("dense").getDenseArray())

print(np.linalg.norm(J_pd_int + np.transpose(J_pd_int)))

J_pd_int_p_petsc = assemble(j_pd_int_p, mat_type='aij').M.handle
J_pd_int_p = np.array(J_pd_int_p_petsc.convert("dense").getDenseArray())

print(np.linalg.norm(J_pd_int_p + np.transpose(J_pd_int_p)))

J_pd_int_d_petsc = assemble(j_pd_int_d, mat_type='aij').M.handle
J_pd_int_d = np.array(J_pd_int_d_petsc.convert("dense").getDenseArray())

print(np.linalg.norm(J_pd_int_d + np.transpose(J_pd_int_d)))

# print(J_pd_int[-6:-3, 3:6])
J_pd[abs(J_pd) < tol] = 0.0
J_pd_int_p[abs(J_pd_int_p) < tol] = 0.0
J_pd_int_d[abs(J_pd_int_d) < tol] = 0.0
J_pd_int[abs(J_pd_int) < tol] = 0.0

plt.figure()
plt.spy(J_pd)
plt.title('No coupling')

plt.figure()
plt.spy(J_pd_int_p)
plt.title('Primal coupling')

plt.figure()
plt.spy(J_pd_int_d)
plt.title('Dual coupling')

plt.figure()
plt.spy(J_pd_int)
plt.title('All coupling')

plt.show()


# J_int_petsc = assemble(int_form_10 + int_form_01, mat_type='aij').M.handle
# J_int = np.array(J_int_petsc.convert("dense").getDenseArray())

# print(np.linalg.norm(J_int + np.transpose(J_int)))
#
# int_form_primal = (dot(v1til_b('+'), n_ver('+')) * u0_b('-') - v0_b('-') * dot(u1til_b('+'), n_ver('+')))*dS
# int_form_dual = (-v0_b('+') * dot(u1til_b('-'), n_ver('-')) + dot(v1til_b('-'), n_ver('-')) * u0_b('+'))*dS
#
# J_int_p_petsc = assemble(int_form_primal, mat_type='aij').M.handle
# J_int_p = np.array(J_int_p_petsc.convert("dense").getDenseArray())
#
# print(np.linalg.norm(J_int_p + np.transpose(J_int_p)))
#
# J_int_d_petsc = assemble(int_form_dual, mat_type='aij').M.handle
# J_int_d = np.array(J_int_d_petsc.convert("dense").getDenseArray())
#
# print(np.linalg.norm(J_int_d + np.transpose(J_int_d)))
