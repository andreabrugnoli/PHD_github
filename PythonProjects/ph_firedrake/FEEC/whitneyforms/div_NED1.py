from firedrake import *

n_el = int(input("Enter number of elements : "))
pol_deg = int(input("Enter polynomial degree : "))
# mesh = PeriodicUnitCubeMesh(n_el, n_el, n_el)
mesh = UnitCubeMesh(n_el, n_el, n_el)
x, y, z = SpatialCoordinate(mesh)

v_ex = as_vector([sin(x)*cos(y)*cos(z), -cos(x)*sin(y)*cos(z), 0])

V_1 = FunctionSpace(mesh, "N1curl", pol_deg)
V_2 = FunctionSpace(mesh, "RT", pol_deg)


u_1 = Function(V_1)
u_1.assign(interpolate(v_ex, V_1))
divu1_L2 = assemble(div(u_1)**2*dx)

u_2 = Function(V_2)
u_2.assign(interpolate(v_ex, V_2))
divu2_L2 = assemble(div(u_2)**2*dx)


print("Nedelec L2 div norm")
print(divu1_L2)
print("Raviart thomas L2 div norm")
print(divu2_L2)
