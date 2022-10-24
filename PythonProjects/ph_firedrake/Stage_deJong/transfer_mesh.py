from firedrake import *
import matplotlib.pyplot as plt

def expr(x, y):
    return sin(2*pi*x)*cos(2*pi*y)

n_el1 = 4
mesh1 = UnitSquareMesh(n_el1, n_el1)

n_el2 = 32
mesh2 = UnitSquareMesh(n_el2, n_el2)

V1 = FunctionSpace(mesh1, "CG", 1)
V2 = FunctionSpace(mesh2, "CG", 2)

f1 = Function(V1)
f2 = Function(V2)
f2_int = Function(V2)

x1, y1  = SpatialCoordinate(mesh1)
expr1 = expr(x1, y1)

f1.assign(interpolate(expr1, V1))

x2, y2 = SpatialCoordinate(mesh2)
expr2 = expr(x2, y2)
f2.assign(interpolate(expr2, V2))

f2_int.assign(project(f1, V2))


trisurf(f1)
trisurf(f2)
trisurf(f2_int)


plt.show()

