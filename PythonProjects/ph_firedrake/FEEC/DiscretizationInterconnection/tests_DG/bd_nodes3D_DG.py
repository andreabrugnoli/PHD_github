from firedrake import *
from tools_plotting import setup
import numpy as np
np.set_printoptions(threshold=np.inf)


import os
os.environ["OMP_NUM_THREADS"] = "1"

tol = 1e-10

import matplotlib.pyplot as plt

n_el = 1
L_x = 1
L_y = 1
L_z = 1

deg = 1

mesh = BoxMesh(n_el, n_el, n_el, L_x, L_y, L_z)
n_ver = FacetNormal(mesh)

P0 = FiniteElement("CG", tetrahedron, deg)
P1 = FiniteElement("N1curl", tetrahedron, deg)
P2 = FiniteElement("RT", tetrahedron, deg)
P3 = FiniteElement("DG", tetrahedron, deg - 1)

V0 = FunctionSpace(mesh, P0)
V1 = FunctionSpace(mesh, P1)
V2 = FunctionSpace(mesh, P2)
V3 = FunctionSpace(mesh, P3)

V0132 = V0 * V1 * V3 * V2


bc_D = DirichletBC(V0132.sub(0), Constant(0), "on_boundary")
bc_N = DirichletBC(V0132.sub(3), Constant((0.0, 0.0, 0.0)), "on_boundary")

# print('D nodes')
# print(bc_D.nodes)
#
# print('N nodes')
# print(bc_N.nodes)

P0_b = BrokenElement(P0)
P1_b = BrokenElement(P1)
P2_b = BrokenElement(P2)
P3_b = BrokenElement(P3)

V0_b = FunctionSpace(mesh, P0_b)
V1_b = FunctionSpace(mesh, P1_b)
V2_b = FunctionSpace(mesh, P2_b)
V3_b = FunctionSpace(mesh, P3_b)

V0132_b = V0_b * V1_b * V3_b * V2_b

# print(V0.boundary_nodes(1))


bc_D_b = DirichletBC(V0132_b.sub(0), Constant(0), "on_boundary", method='topological')
bc_N_b = DirichletBC(V0132_b.sub(3), Constant((0.0, 0.0, 0.0)), "on_boundary")

print('D nodes broken')
print(bc_D_b.nodes)

print('N nodes broken')
print(bc_N_b.nodes)
