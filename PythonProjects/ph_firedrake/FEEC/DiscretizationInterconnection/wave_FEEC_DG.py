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
n_ver = FacetNormal(mesh)

triplot(mesh)
plt.show()
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

V_1til_0_b = V1til_b * V0_b

dx = Measure('dx')
ds = Measure('ds')
dS = Measure('dS')

v_1til_0_b = TestFunction(V_1til_0_b)
v1til_b, v0_b = split(v_1til_0_b)

u_1til_0_b = TrialFunction(V_1til_0_b)
u1til_b, u0_b = split(u_1til_0_b)

# v0_b = TestFunction(V0_b)
# u0_b = TrialFunction(V0_b)
#
# v1_b = TestFunction(V1_b)
# u1_b = TrialFunction(V1_b)
#
# v1til_b = TestFunction(V1til_b)
# u1til_b = TrialFunction(V1til_b)
#
# v2_b = TestFunction(V2_b)
# u2_b = TrialFunction(V2_b)
#
# f0_b = Function(V0_b)
# f1_b = Function(V1_b)
# f1til_b = Function(V1til_b)
# f2_b = Function(V2_b)

int_form_10 = (dot(v1til_b('+'), n_ver('+')) * u0_b('-') + dot(v1til_b('-'), n_ver('-')) * u0_b('+'))*dS
int_form_01 = - (v0_b('+') * dot(u1til_b('-'), n_ver('-')) + v0_b('-') * dot(u1til_b('+'), n_ver('+')))*dS

J_int_petsc = assemble(int_form_10 + int_form_01, mat_type='aij').M.handle
J_int = np.array(J_int_petsc.convert("dense").getDenseArray())

print(np.linalg.norm(J_int + np.transpose(J_int)))

int_form_primal = (dot(v1til_b('+'), n_ver('+')) * u0_b('-') - v0_b('-') * dot(u1til_b('+'), n_ver('+')))*dS
int_form_dual = (-v0_b('+') * dot(u1til_b('-'), n_ver('-')) + dot(v1til_b('-'), n_ver('-')) * u0_b('+'))*dS

J_int_p_petsc = assemble(int_form_primal, mat_type='aij').M.handle
J_int_p = np.array(J_int_p_petsc.convert("dense").getDenseArray())

print(np.linalg.norm(J_int_p + np.transpose(J_int_p)))

J_int_d_petsc = assemble(int_form_dual, mat_type='aij').M.handle
J_int_d = np.array(J_int_d_petsc.convert("dense").getDenseArray())

print(np.linalg.norm(J_int_d + np.transpose(J_int_d)))
