## This is a first test to solve the wave equation in 2D domains using the dual filed method

# from warnings import simplefilter
# simplefilter(action='ignore', category=DeprecationWarning)

import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from firedrake import *

bd_cond=input("Enter bc: ")

n_el = 1
deg= 1

L = 1/2
mesh = BoxMesh(n_el, n_el, n_el, 1, 1/2, 1/2)
n_ver = FacetNormal(mesh)

P_0 = FiniteElement("CG", tetrahedron, deg)
# P_1 = FiniteElement("N1curl", tetrahedron, deg, variant='integral')
P_1 = FiniteElement("N1curl", tetrahedron, deg)
P_2 = FiniteElement("RT", tetrahedron, deg)
# Integral evaluation on Raviart-Thomas for deg=3 completely freezes interpolation
# P_2 = FiniteElement("RT", tetrahedron, deg, variant='integral')
P_3 = FiniteElement("DG", tetrahedron, deg - 1)

V_3 = FunctionSpace(mesh, P_3)
V_1 = FunctionSpace(mesh, P_1)

V_0 = FunctionSpace(mesh, P_0)
V_2 = FunctionSpace(mesh, P_2)

print(V_0.dim())
print(V_1.dim())
print(V_2.dim())
print(V_3.dim())

V_32 = V_3 * V_2
V_10 = V_1 * V_0

x, y, z = SpatialCoordinate(mesh)

om_x = 1
om_y = 1
om_z = 1

om_t = np.sqrt(om_x ** 2 + om_y ** 2 + om_z ** 2)
phi_x = 0
phi_y = 0
phi_z = 0
phi_t = 0

t = Constant(0.0)

ft = 2 * sin(om_t * t + phi_t) + 3 * cos(om_t * t + phi_t)
dft = om_t * (2 * cos(om_t * t + phi_t) - 3 * sin(om_t * t + phi_t))  # diff(dft_t, t)

gxyz = cos(om_x * x + phi_x) * sin(om_y * y + phi_y) * sin(om_z * z + phi_z)

dgxyz_x = - om_x * sin(om_x * x + phi_x) * sin(om_y * y + phi_y) * sin(om_z * z + phi_z)
dgxyz_y = om_y * cos(om_x * x + phi_x) * cos(om_y * y + phi_y) * sin(om_z * z + phi_z)
dgxyz_z = om_z * cos(om_x * x + phi_x) * sin(om_y * y + phi_y) * cos(om_z * z + phi_z)

grad_gxyz = as_vector([dgxyz_x,
                       dgxyz_y,
                       dgxyz_z]) # grad(gxyz)


p_ex = gxyz * dft
u_ex = grad_gxyz * ft

dx = Measure('dx')
ds = Measure('ds')


if bd_cond=="D":
    bc_D = [DirichletBC(V_10.sub(1), p_ex, "on_boundary")]
    bc_D_nat = None

    bc_N = None
    bc_N_nat = [DirichletBC(V_32.sub(1), u_ex, "on_boundary")]

elif bd_cond=="N":
    bc_N = [DirichletBC(V_32.sub(1), u_ex, "on_boundary")]
    bc_N_nat = None

    bc_D = None
    bc_D_nat = [DirichletBC(V_10.sub(1), p_ex, "on_boundary")]

else:
    bc_D = [DirichletBC(V_10.sub(1), p_ex, 1), \
            DirichletBC(V_10.sub(1), p_ex, 3), \
            DirichletBC(V_10.sub(1), p_ex, 5)]

    bc_D_nat = [DirichletBC(V_10.sub(1), p_ex, 2), \
                DirichletBC(V_10.sub(1), p_ex, 4), \
                DirichletBC(V_10.sub(1), p_ex, 6)]

    bc_N = [DirichletBC(V_32.sub(1), u_ex, 2), \
            DirichletBC(V_32.sub(1), u_ex, 4), \
            DirichletBC(V_32.sub(1), u_ex, 6)]

    bc_N_nat = [DirichletBC(V_32.sub(1), u_ex, 1), \
                DirichletBC(V_32.sub(1), u_ex, 3), \
                DirichletBC(V_32.sub(1), u_ex, 5)]

dofs10_D = []
dofs32_D = []

if bc_D is not None:
    for ii in range(len(bc_D)):
        print("Dirichlet bc " + str(ii))
        nodes10_D = V_1.dim() + bc_D[ii].nodes
        nodes32_D = V_3.dim() + bc_N_nat[ii].nodes

        dofs10_D = dofs10_D + list(nodes10_D)
        dofs32_D = dofs32_D + list(nodes32_D)
        print("Nodes in bc_D")
        print(bc_D[ii].nodes)
        print("Nodes in bc_N_nat")
        print(bc_N_nat[ii].nodes)


else: print("None D value")

dofs10_D = list(set(dofs10_D))
dofs32_D = list(set(dofs32_D))

print("10 dofs on Gamma_D")
print(dofs10_D)
print("32 dofs on Gamma_D")
print(dofs32_D)


dofs10_N = []
dofs32_N = []


if bc_N is not None:
    for ii in range(len(bc_N)):
        print("Neumann bc " + str(ii))
        nodes32_N = V_3.dim() + bc_N[ii].nodes
        nodes10_N = V_1.dim() + bc_D_nat[ii].nodes

        dofs32_N = dofs32_N + list(nodes32_N)
        dofs10_N = dofs10_N + list(nodes10_N)

        print("Nodes in bc_N")
        print(bc_N[ii].nodes)
        print("Nodes in bc_D_nat")
        print(bc_D_nat[ii].nodes)
else: print("None N value")

dofs32_N = list(set(dofs32_N))
dofs10_N = list(set(dofs10_N))

print("32 dofs on Gamma_N")
print(dofs32_N)
print("10 dofs on Gamma_N")
print(dofs10_N)

