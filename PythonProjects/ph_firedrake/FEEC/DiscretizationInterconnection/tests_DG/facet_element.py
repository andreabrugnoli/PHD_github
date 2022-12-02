from firedrake import *

n_el = 3
deg = 1
mesh = UnitSquareMesh(n_el, n_el, quadrilateral=True)
n_ver = FacetNormal(mesh)

P0 = FiniteElement('CG', mesh.ufl_cell(), deg)
P0_b = BrokenElement(P0)
P0_f = FacetElement(P0)
P0_f_b = BrokenElement(P0_f)

# P1til = FiniteElement('RT', mesh.ufl_cell(), deg+1)
P1til = FiniteElement('RTCF', mesh.ufl_cell(), deg+1)

P1til_f = FacetElement(P1til)
P1til_f_b = BrokenElement(P1til_f)

V0_tan = FunctionSpace(mesh, P0_f)

W0_nor = FunctionSpace(mesh, P0_f_b)
W0 = FunctionSpace(mesh, P0_b)


W1til_nor = FunctionSpace(mesh, P1til_f_b)


v0 = TestFunction(V0_tan)
v0f_b = TestFunction(W0_nor)
v0_b = TestFunction(W0)

e1til_b = TrialFunction(W1til_nor)

# ## Duality for 0 and 1til tan broken
# dl_form_tb = v0 * dot(e1til_b, n_ver)
#
# form_tb = (dl_form_tb('+') + dl_form_tb('+'))*dS + dl_form_tb*ds
#
# petsc_form_tb = assemble(form_tb, mat_type='aij').M.handle
# G_0t_1tilb = np.array(petsc_form_tb.convert("dense").getDenseArray())
#
# print("Boundary matrix duality")
#
# print(G_0t_1tilb.shape, np.linalg.matrix_rank(G_0t_1tilb))

## Duality for 0 and 1til
dl_form_bb = v0_b * dot(e1til_b, n_ver)

form_bb = (dl_form_bb('+') + dl_form_bb('+'))*dS + dl_form_bb*ds

petsc_form_bb = assemble(form_bb, mat_type='aij').M.handle
G_0b_1tilb = np.array(petsc_form_bb.convert("dense").getDenseArray())

print("Boundary matrix duality")

print(G_0b_1tilb.shape, np.linalg.matrix_rank(G_0b_1tilb))




