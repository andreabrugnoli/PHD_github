# Test to try to understand with hybrid multiplier
from firedrake import *
from FEEC.DiscretizationInterconnection.slate_syntax.solve_hybrid_system import solve_hybrid, solve_hybrid_2constr
from matplotlib import pyplot as plt
from tools_plotting import setup

def laplace_form(v_0, u_0):
    return dot(grad(v_0), grad(u_0))*dx


def constr_loc(v_0, u_0, v_0_nor, lam_0_nor):
    form = (v_0('+') * lam_0_nor('+') + v_0('-') * lam_0_nor('-')) * dS + v_0 * lam_0_nor * ds \
           - ((v_0_nor('+') * u_0('+') + v_0_nor('-') * u_0('-')) * dS + v_0_nor * u_0 * ds)
    return form


def constr_global(v_0_nor, lam_0_nor, v_0_tan, u_0_tan):
    form = (v_0_nor('+') * u_0_tan('+') + v_0_nor('-') * u_0_tan('-')) * dS + v_0_nor * u_0_tan * ds \
           - ((v_0_tan('+') * lam_0_nor('+') + v_0_tan('-') * lam_0_nor('-')) * dS + v_0_tan * lam_0_nor * ds)

    return form


# def constr_bd(v_0, lam_0_nor):
#     form = (v_0('+') * lam_0_nor('+') + v_0('-') * lam_0_nor('-')) * dS + v_0 * lam_0_nor * ds
#     return form
#
#
# def cont_u(v_0_nor, u_0, u_0_tan):
#     form = (v_0_nor('+') * u_0_tan('+') + v_0_nor('-') * u_0_tan('-')) * dS + v_0_nor * u_0_tan * ds \
#         - ((v_0_nor('+') * u_0('+') + v_0_nor('-') * u_0('-')) * dS + v_0_nor * u_0 * ds)
#     return form
#
#
# def cont_lam(v_0_tan, lam_0_nor):
#     form = (v_0_tan('+') * lam_0_nor('+') + v_0_tan('-') * lam_0_nor('-')) * dS + v_0_tan * lam_0_nor * ds
#     # form = avg(v_0_tan)*jump(lam_0_nor) * dS + v_0_tan * lam_0_nor * ds
#     return form


def f_form(v_0, f):
    return v_0 * f * dx


n_el = 10
deg = 3

msh = UnitSquareMesh(n_el, n_el)
dx = Measure('dx')
ds = Measure('ds')
dS = Measure('dS')

n_ver = FacetNormal(msh)

P0 = FiniteElement("CG", triangle, deg)
P0f = FacetElement(P0)

V0 = FunctionSpace(msh, P0)

P0_b = BrokenElement(P0)

P0f_b = BrokenElement(P0f)

W0 = FunctionSpace(msh, P0_b)
W0_nor = FunctionSpace(msh, P0f_b)
V0_tan = FunctionSpace(msh, P0f)


W_loc = W0 * W0_nor

V_grad = W_loc * V0_tan

v_grad = TestFunction(V_grad)
v0, v0_nor, v0_tan = split(v_grad)

e_grad = TrialFunction(V_grad)
u0, lam0_nor, u0_tan = split(e_grad)

print(W0.dim())
print(W0_nor.dim())
print(V0_tan.dim())

omega_x = 1
omega_y = 1
x, y = SpatialCoordinate(msh)
u_ex = sin(omega_x*x)*sin(omega_y*y)

gradu_ex = grad(u_ex)

# f_ex = -(omega_x**2 + omega_y**2)*sin(omega_x*x)*sin(omega_y*y)
f_ex = -div(grad(u_ex))

bc_D = [DirichletBC(V0_tan, u_ex, "on_boundary")]

dofs_D = []

for ii in range(len(bc_D)):
    nodesD = bc_D[ii].nodes

    dofs_D = dofs_D + list(nodesD)

dofs_D = list(set(dofs_D))
dofs_N = list(set(V0_tan.boundary_nodes("on_boundary")).difference(set(dofs_D)))

# print('dofs D')
# print(dofs_D)
#
# print('dofs N')
# print(dofs_N)

dofsV0_tan_D = W0.dim() + W0_nor.dim() + np.array(dofs_D)
dofsV0_tan_N = W0.dim()  + W0_nor.dim() + np.array(dofs_N)

dofsV0_tan_NoD = list(set(np.arange(V_grad.dim())).difference(set(dofsV0_tan_D)))

bcD_mixed = DirichletBC(V_grad.sub(2), u_ex, "on_boundary")

a_form = laplace_form(v0, u0) - constr_loc(v0, u0, v0_nor, lam0_nor) - constr_global(v0_nor, lam0_nor, v0_tan, u0_tan)
# a_form = laplace_form(v0, u0) - constr_bd(v0, lam0_nor) + cont_u(v0_nor, u0, u0_tan) + cont_lam(v0_tan, lam0_nor)
b_form = f_form(v0, f_ex)

# sol = Function(V_grad)
# solve(a_form == b_form, sol, bcs=bcD_mixed)

# petsc_a = assemble(a_form, mat_type='aij').M.handle
# A = np.array(petsc_a.convert("dense").getDenseArray())
#
# plt.spy(A)

sol = solve_hybrid(a_form, b_form, bc_D, V0_tan, W_loc)
u_h, lamnor_h, utan_h = sol.split()



Pgradn_uex_ = Function(W0_nor)
wtan = TestFunction(W0_nor)
Pgradn_u = TrialFunction(W0_nor)

A_Pgradn_uex = (wtan('+') * Pgradn_u('+') + wtan('-') * Pgradn_u('-')) * dS + wtan * Pgradn_u * ds

b_Pgradn_uex = (wtan('+') * dot(grad(u_ex)('+'), n_ver('+')) \
              + wtan('-') * dot(grad(u_ex)('-'), n_ver('-'))) * dS \
              + wtan * dot(grad(u_ex), n_ver) * ds

solve(A_Pgradn_uex == b_Pgradn_uex, Pgradn_uex_)


DeltaH = assemble(dot(grad(u_h), grad(u_h))*dx(domain=msh))

Work_lam = assemble((lamnor_h('+')*u_h('+') + lamnor_h('-')*u_h('-')) * dS(domain=msh)
                    + lamnor_h * u_h * ds(domain=msh))

Work_F = assemble(u_h * f_ex * dx(domain=msh))

# Constraint_u = assemble((lamnor_h('+') * u_h('+') + lamnor_h('-') * u_h('-')) * dS \
# - (lamnor_h('+') * utan_h('+') + lamnor_h('-') * utan_h('-')) * dS)
#
# print("Constraint u " + str(Constraint_u))

Constraint_lam_vec = assemble((v0_tan('+') * lamnor_h('+') + v0_tan('-') * lamnor_h('-'))*dS).vector().get_local()[dofsV0_tan_NoD]

Constraint_lam = np.dot(sol.vector().get_local()[dofsV0_tan_NoD], Constraint_lam_vec)
print("Constraint lam " + str(Constraint_lam))

y_nmid_D = (v0_tan('+') * lamnor_h('+') +  v0_tan('-') * lamnor_h('-')) * dS \
                    + v0_tan * lamnor_h * ds
yess_D = assemble(y_nmid_D).vector().get_local()[dofsV0_tan_D]
uess_D = assemble(sol).vector().get_local()[dofsV0_tan_D]
Work_lam = np.dot(yess_D, uess_D) #assemble(lamnor_h * u_h * ds(domain=msh))

Work= Work_F + Work_lam
tol = 1e-12
print("Power preservation " + str(DeltaH-Work))
assert abs(DeltaH - Work) < tol

fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
contours = trisurf(interpolate(u_ex, V0), axes=axes, cmap="inferno")
axes.set_aspect("auto")
axes.set_title("u ex")
fig.colorbar(contours)
fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
contours = trisurf(u_h, axes=axes, cmap="inferno")
axes.set_aspect("auto")
axes.set_title("u h")
fig.colorbar(contours)

fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
contours = trisurf(utan_h, axes=axes, cmap="inferno")
axes.set_aspect("auto")
axes.set_title("utanh")
fig.colorbar(contours)

triplot(msh)

fig = plt.figure()
axes = fig.add_subplot(111)
contours = tricontourf(lamnor_h, axes=axes, cmap="inferno")
axes.set_aspect("auto")
axes.set_title("lam nor")
fig.colorbar(contours)

fig = plt.figure()
axes = fig.add_subplot(111)
contours = tricontourf(Pgradn_uex_, axes=axes, cmap="inferno")
axes.set_aspect("auto")
axes.set_title("gradn uex")
fig.colorbar(contours)

err_gradn_uex = Function(W0_nor)
err_gradn_uex.assign(Pgradn_uex_-lamnor_h)

fig = plt.figure()
axes = fig.add_subplot(111)
contours = tricontourf(err_gradn_uex, axes=axes, cmap="inferno")
axes.set_aspect("auto")
axes.set_title("Error lambda nor")
fig.colorbar(contours)

plt.show()