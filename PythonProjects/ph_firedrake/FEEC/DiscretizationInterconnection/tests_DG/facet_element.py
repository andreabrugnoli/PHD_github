from firedrake import *

n_el = 1
deg = 3
mesh = UnitSquareMesh(n_el, n_el)

P0 = FiniteElement('CG', mesh.ufl_cell(), deg)

P0_f = FacetElement(P0)
V0_tan = FunctionSpace(mesh, P0_f)

P0_f_b = BrokenElement(P0_f)
V0_nor = FunctionSpace(mesh, P0_f_b)

V0 = FunctionSpace(mesh, P0)


print(V0_tan.dim(), V0_nor.dim(), V0.dim())

print(V0_tan.boundary_nodes("on_boundary"))


