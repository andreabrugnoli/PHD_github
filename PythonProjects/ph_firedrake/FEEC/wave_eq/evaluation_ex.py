import os
os.environ["OMP_NUM_THREADS"] = "1"

# import numpy as np
from firedrake import *
import  matplotlib.pyplot as plt

n_el = 10
L = 1
mesh = RectangleMesh(n_el, n_el, L, L, quadrilateral=False)
n_ver = FacetNormal(mesh)


dx = Measure('dx')
ds = Measure('ds')

x, y = SpatialCoordinate(mesh)

t = Constant(0.0)
om_x = pi
om_y = pi

om_t = np.sqrt(om_x ** 2 + om_y ** 2)
phi_x = 1
phi_y = 10
phi_t = 5

w_ex = sin(om_x * x + phi_x) * sin(om_y * y + phi_y) * sin(om_t * t + phi_t)
p_ex = om_t * sin(om_x * x + phi_x) * sin(om_y * y + phi_y) * cos(om_t * t + phi_t)
u_ex = as_vector([om_x * cos(om_x * x + phi_x) * sin(om_y * y + phi_y) * sin(om_t * t + phi_t),
                  om_y * sin(om_x * x + phi_x) * cos(om_y * y + phi_y) * sin(om_t * t + phi_t)])

bdflow_ex_n1 = dot(u_ex, n_ver) * ds(domain=mesh)
bdflow_ex_n2 = p_ex * ds(domain=mesh)
bdflow_ex_n3 = p_ex * dot(u_ex, n_ver) * ds(domain=mesh)

H_ex_n1 = 0.5 * p_ex*2 *dx(domain=mesh) + 0.5*dot(u_ex, u_ex) * dx(domain=mesh)
H_ex_n2 = 0.5 * p_ex*2 *dx(domain=mesh) + 0.5*dot(grad(w_ex), grad(w_ex)) * dx(domain=mesh)



for ii in range(10):
    t.assign(ii)

    bd_flow1 = assemble(bdflow_ex_n1)
    bd_flow2 = assemble(bdflow_ex_n2)
    bd_flow3 = assemble(bdflow_ex_n3)

    print("Boundary flow 1:")
    print(bd_flow1)
    print("Boundary flow 2:")
    print(bd_flow2)
    print("Boundary flow 3:")
    print(bd_flow3)

for ii in range(10):
    t.assign(ii)
    H_n1 = assemble(H_ex_n1)
    H_n2 = assemble(H_ex_n2)

    print("Energy1")
    print(H_n1)
    print("Energy2")
    print(H_n2)