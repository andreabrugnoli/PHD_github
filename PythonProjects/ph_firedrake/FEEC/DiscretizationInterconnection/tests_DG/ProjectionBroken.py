from firedrake import *
from tools_plotting import setup
import numpy as np
np.set_printoptions(threshold=np.inf)
import os
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt

n_el = 10
L = 1

mesh = UnitSquareMesh(n_el, n_el)
n_ver = FacetNormal(mesh)

deg = 1

P0 = FiniteElement("CG", triangle, deg)
P1 = FiniteElement("N1curl", triangle, deg, variant="integral")

P1_b = BrokenElement(P1)

V0 = FunctionSpace(mesh, P0)
V1 = FunctionSpace(mesh, P1)

V1_b = FunctionSpace(mesh, P1_b)

x, y = SpatialCoordinate(mesh)

omega_x = 1
omega_y = 1

exp0 = sin(omega_x*x)*sin(omega_y*y)

# grad_exp0 = grad(exp0)
grad_exp0 = as_vector([omega_x*cos(omega_x*x)*sin(omega_y*y),
                       omega_y*sin(omega_x*x)*cos(omega_y*y)])

exp0_h = project(exp0, V0)
grad_exp0_h = grad(exp0_h)
# trisurf(exp0_h)
# plt.show()

u1_b_prex = project(grad_exp0, V1_b)
u1_b_prh = project(grad_exp0_h, V1_b)

# quiver(u1_b_prex)
# quiver(u1_b_prh)
#
# plt.show()

# print(np.sqrt(assemble(dot(u1_b_prex-u1_b_prh, u1_b_prex-u1_b_prh)*dx)))
print(errornorm(grad_exp0_h, u1_b_prex, norm_type="L2"))
print(errornorm(u1_b_prh, u1_b_prex, norm_type="L2"))

u1_prex = project(grad_exp0, V1)
u1_prh = project(grad_exp0_h, V1)

# print(np.sqrt(assemble(dot(u1_prex-u1_prh, u1_prex-u1_prh)*dx)))
print(errornorm(grad_exp0_h, u1_prex, norm_type="L2"))
print(errornorm(u1_prh, u1_prex, norm_type="L2"))