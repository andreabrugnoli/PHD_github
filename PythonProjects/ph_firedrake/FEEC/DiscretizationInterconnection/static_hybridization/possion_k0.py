# Test to try to understand with hybrid multiplier
from firedrake import *
from FEEC.DiscretizationInterconnection.slate_syntax.solve_hybrid_system import solve_hybrid, solve_hybrid_2constr
from matplotlib import pyplot as plt


def laplace_form(v_0, u_0):
    return dot(grad(v_0), grad(u_0))*dx


# def constr_loc(v_0, u_0, v_0_nor, lam_0_nor):
#     form = (v_0('+') * lam_0_nor('+') + v_0('-') * lam_0_nor('-')) * dS + v_0 * lam_0_nor * ds \
#            - ((v_0_nor('+') * u_0('+') + v_0_nor('-') * u_0('-')) * dS + v_0_nor * u_0 * ds)
#     return form
#
#
# def constr_global(v_0_nor, lam_0_nor, v_0_tan, u_0_tan):
#     form = (v_0_nor('+') * u_0_tan('+') + v_0_nor('-') * u_0_tan('-')) * dS + v_0_nor * u_0_tan * ds \
#            - ((v_0_tan('+') * lam_0_nor('+') + v_0_tan('-') * lam_0_nor('-')) * dS + v_0_tan * lam_0_nor * ds)
#     return form


def constr_bd(v_0, lam_0_nor):
    form = (v_0('+') * lam_0_nor('+') - v_0('-') * lam_0_nor('-')) * dS + v_0 * lam_0_nor * ds
    return form


def cont_u(v_0_nor, u_0, u_0_tan):
    form = (v_0_nor('+') * u_0_tan('+') - v_0_nor('-') * u_0_tan('-')) * dS + v_0_nor * u_0_tan * ds \
        - ((v_0_nor('+') * u_0('+') + v_0_nor('-') * u_0('-')) * dS + v_0_nor * u_0 * ds)
    return form


def cont_lam(v_0_tan, lam_0_nor):
    form = (v_0_tan('+') * lam_0_nor('+') - v_0_tan('-') * lam_0_nor('-')) * dS + v_0_tan * lam_0_nor * ds
    # form = avg(v_0_tan)*jump(lam_0_nor) * dS + v_0_tan * lam_0_nor * ds
    return form


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


x, y = SpatialCoordinate(msh)
u_ex = sin(pi*x)*sin(pi*y)

gradu_ex = grad(u_ex)

# f_ex = 2*pi**2*sin(pi*x)*sin(pi*y)
f_ex = -div(grad(u_ex))


bc_D = DirichletBC(V0_tan, Constant(0), "on_boundary")

bcD_mixed = DirichletBC(V_grad.sub(2), Constant(0), "on_boundary")

# a_form = laplace_form(v0, u0) - constr_loc(v0, u0, v0_nor, lam0_nor) - constr_global(v0_nor, lam0_nor, v0_tan, u0_tan)

a_form = laplace_form(v0, u0) - constr_bd(v0, lam0_nor) + cont_u(v0_nor, u0, u0_tan) + cont_lam(v0_tan, lam0_nor)
b_form = f_form(v0, f_ex)

# sol = Function(V_grad)
# solve(a_form == b_form, sol, bcs=bcD_mixed)

# petsc_a = assemble(a_form, mat_type='aij').M.handle
# A = np.array(petsc_a.convert("dense").getDenseArray())
#
# plt.spy(A)

sol = solve_hybrid(a_form, b_form, bc_D, V0_tan, W_loc)
u_h, lamnor_h, utan_h = sol.split()


fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
contours = trisurf(interpolate(u_ex, V0), axes=axes, cmap="inferno")
axes.set_aspect("auto")
axes.set_title("u_ex")
fig.colorbar(contours)
fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
contours = trisurf(u_h, axes=axes, cmap="inferno")
axes.set_aspect("auto")
axes.set_title("u_h")
fig.colorbar(contours)

fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
contours = trisurf(utan_h, axes=axes, cmap="inferno")
axes.set_aspect("auto")
axes.set_title("utan_h")
fig.colorbar(contours)

triplot(msh)

fig = plt.figure()
axes = fig.add_subplot(111)
contours = tricontourf(lamnor_h, axes=axes, cmap="inferno")
axes.set_aspect("auto")
axes.set_title("lam_nor")
fig.colorbar(contours)

fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
contours = trisurf(lamnor_h, axes=axes, cmap="inferno")
axes.set_aspect("auto")
axes.set_title("lam_nor")
fig.colorbar(contours)

plt.show()