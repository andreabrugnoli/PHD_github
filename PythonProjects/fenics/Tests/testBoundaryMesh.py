from dolfin import *
import numpy as np

parameters["allow_extrapolation"] = True

mesh = UnitSquareMesh(20, 20)

class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0)
leftboundary = LeftBoundary()

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
leftboundary.mark(boundaries, 1)
ds = Measure("ds", domain=mesh, subdomain_data = boundaries)

boundarymesh = BoundaryMesh(mesh, 'exterior')
left_mesh = SubMesh(boundarymesh, leftboundary)
L = FunctionSpace(left_mesh, 'CG', 1)
bf = Function(L)
# here goes some more code to find the correct bf
# but here a dummy function will do
bf.vector()[:] = np.ones(21, dtype=float)



V = FunctionSpace(mesh, 'CG', 1)
u = TrialFunction(V)
v = TestFunction(V)
a = inner(grad(u), grad(v)) * dx
L = interpolate(bf, V) * v * ds(1)
assemble(L)