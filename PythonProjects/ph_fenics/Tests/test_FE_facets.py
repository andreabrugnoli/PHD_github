from fenics import *
parameters["allow_extrapolation"] = True

import numpy as np

"Returns an error"

mesh = UnitSquareMesh(20, 20)
bd_mesh = BoundaryMesh(mesh, 'exterior')

class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0)

leftboundary = LeftBoundary()

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
leftboundary.mark(boundaries, 1)
ds = Measure('ds')[boundaries]

boundarymesh = BoundaryMesh(mesh, 'exterior')
left_mesh = SubMesh(boundarymesh, leftboundary)
V_bd = FunctionSpace(left_mesh, 'CG', 1)
vf = TestFunction(V_bd)
bf = TrialFunction(V_bd)

# here goes some more code to find the correct bf
# but here a dummy function will do
# bf.vector()[:] = np.ones(21, dtype=float)

# V = FunctionSpace(mesh, 'CG', 1)
# u = TrialFunction(V)
# v = TestFunction(V)


b_form = vf * bf * ds(subdomain_data=left_mesh)
assemble(b_form)
