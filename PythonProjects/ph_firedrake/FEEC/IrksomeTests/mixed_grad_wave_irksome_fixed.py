from firedrake import *
from irksome import GaussLegendre, Dt, TimeStepper
import numpy

from warnings import simplefilter
simplefilter(action='ignore', category=DeprecationWarning)

N = 10

msh = UnitSquareMesh(N, N)
V = FunctionSpace(msh, "N1curl", 1)
W = FunctionSpace(msh, "CG", 1)
Z = V*W

x, y = SpatialCoordinate(msh)
up0 = project(as_vector([0, 0, sin(pi*x)*sin(pi*y)]), Z)
u0, p0 = split(up0)

v, w = TestFunctions(Z)
F = inner(Dt(u0), v)*dx - inner(u0, grad(w)) * dx + inner(Dt(p0), w)*dx + inner(grad(p0), v) * dx

E = 0.5 * (inner(u0, u0)*dx + inner(p0, p0)*dx)

bc = DirichletBC(Z.sub(1), 0, "on_boundary")

t = Constant(0.0)
dt = Constant(1.0/N)

butcher_tableau = GaussLegendre(2)
params = {"mat_type": "aij",
          "snes_type": "ksponly",
          "ksp_type": "preonly",
          "pc_type": "lu"}

stepper = TimeStepper(F, butcher_tableau, t, dt, up0,
                      bcs=bc, solver_parameters=params)

print("Time    Energy")
print("==============")

while (float(t) < 1.0):
    if float(t) + float(dt) > 1.0:
        dt.assign(1.0 - float(t))

    stepper.advance()

    t.assign(float(t) + float(dt))
    print("{0:1.1e} {1:5e}".format(float(t), assemble(E)))
