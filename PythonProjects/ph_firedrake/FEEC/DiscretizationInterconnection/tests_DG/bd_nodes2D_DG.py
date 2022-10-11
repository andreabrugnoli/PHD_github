from firedrake import *
from tools_plotting import setup
import numpy as np
np.set_printoptions(threshold=np.inf)


import os
os.environ["OMP_NUM_THREADS"] = "1"

tol = 1e-10

import matplotlib.pyplot as plt

n_el = 2
L_x = 1
L_y = 1

deg = 2

mesh = RectangleMesh(n_el, n_el, L_x, L_y)
n_ver = FacetNormal(mesh)

P0 = FiniteElement("CG", triangle, deg)
P1 = FiniteElement("N1curl", triangle, deg, variant="integral")
P2 = FiniteElement("RT", triangle, deg, variant="integral")
P3 = FiniteElement("DG", triangle, deg - 1)

V0 = FunctionSpace(mesh, P0)
V1 = FunctionSpace(mesh, P1)
V2 = FunctionSpace(mesh, P2)
V3 = FunctionSpace(mesh, P3)

V0132 = V0 * V1 * V3 * V2

v0132 = TestFunction(V0132)
v0, v1, v3, v2 = split(v0132)

e0132 = TrialFunction(V0132)
p0, u1, p3, u2 = split(e0132)

bc_D = DirichletBC(V0132.sub(0), Constant(0), "on_boundary")
bc_N = DirichletBC(V0132.sub(3), Constant((0.0, 0.0)), "on_boundary")

print('D nodes DirichletBC')
print(bc_D.nodes)
print(len(bc_D.nodes))

print('N nodes DirichletBC')
print(V0.dim() + V1.dim() + V3.dim() + bc_N.nodes)
print(len(bc_N.nodes))


# P0nc = FiniteElement("CR", triangle, deg)
# P0nc_b = BrokenElement(P0nc)
# V0nc_b = FunctionSpace(mesh, P0nc_b)

P0_b = BrokenElement(P0)

P1_b = BrokenElement(P1)
P2_b = BrokenElement(P2)
P3_b = BrokenElement(P3)

V0_b = FunctionSpace(mesh, P0_b)
V1_b = FunctionSpace(mesh, P1_b)
V2_b = FunctionSpace(mesh, P2_b)
V3_b = FunctionSpace(mesh, P3_b)

V0132_b = V0_b * V1_b * V3_b * V2_b

v0132_b = TestFunction(V0132_b)
v0_b, v1_b, v3_b, v2_b = split(v0132_b)

e0132_b = TrialFunction(V0132_b)
p0_b, u1_b, p3_b, u2_b = split(e0132_b)

dx = Measure('dx')
ds = Measure('ds')
dS = Measure('dS')

# Routine for obtaining the degrees of freedom for different spaces

bform_0 = v0 * Constant(1) * ds

bvec_0 = assemble(bform_0).vector().get_local()

bvec_0[abs(bvec_0) < tol]=0

dofs_V0 = list(bvec_0.nonzero()[0])

# D Dofs from boundary integral evaluation
print('D dofs boundary evaluation')
print(dofs_V0)
print(len(dofs_V0))
bform_2 = dot(v2, n_ver) * ds
# bform_2 = dot(v2, n_ver) * Constant(1) * ds

bvec_2 = assemble(bform_2).vector().get_local()
bvec_2[abs(bvec_2) < tol]=0
# print(bvec_2)

dofs_V2 = list(bvec_2.nonzero()[0])
# N Dofs from boundary integral evaluation
print('N dofs boundary evaluation')
print(dofs_V2)
print(len(dofs_V2))

