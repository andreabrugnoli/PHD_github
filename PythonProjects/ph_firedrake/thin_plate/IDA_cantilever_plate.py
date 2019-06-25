# Mindlin plate written with the port Hamiltonian approach

from firedrake import *
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt

import scipy.linalg as la

import matplotlib
import matplotlib.pyplot as plt
from assimulo.solvers import IDA
from assimulo.implicit_ode import Implicit_Problem
from firedrake.plot import _two_dimension_triangle_func_val
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from tools_plotting.animate_surf import animate2D
matplotlib.rcParams['text.usetex'] = True

n = 2
r = 2 #int(input('Degree for FE: '))

E = 2e11 # Pa
rho = 8000  # kg/m^3
# E = 1
# rho = 1  # kg/m^3

nu = 0.3
h = 0.01

plot_eigenvector = 'y'

bc_input = input('Select Boundary Condition: ')

L = 1

D = E * h ** 3 / (1 - nu ** 2) / 12
fl_rot = 12 / (E * h ** 3)
norm_coeff = L ** 2 * np.sqrt(rho*h/D)
# Useful Matrices

# Operators and functions
def gradSym(u):
    return 0.5 * (nabla_grad(u) + nabla_grad(u).T)
    # return sym(nabla_grad(u))

def bending_curv(momenta):
    kappa = fl_rot * ((1+nu)*momenta - nu * Identity(2) * tr(momenta))
    return kappa


# The unit square mesh is divided in :math:`N\times N` quadrilaterals::
n_x, n_y = n, n
mesh = UnitSquareMesh(n_x, n_y, quadrilateral=False)

# domain = mshr.Rectangle(Point(0, 0), Point(l_x, l_y))
# mesh = mshr.generate_mesh(domain, n, "cgal")

# plot(mesh)
# plt.show()


# Finite element defition

Vp = FunctionSpace(mesh, 'CG', r+1)
Vq = FunctionSpace(mesh, 'HHJ', r)
V = Vp * Vq

n_Vp = V.sub(0).dim()
n_Vq = V.sub(1).dim()
n_V = V.dim()
print(n_V)

v = TestFunction(V)
v_p, v_q = split(v)

e = TrialFunction(V)
e_p, e_q = split(e)

al_p = rho * h * e_p
al_q = bending_curv(e_q)

dx = Measure('dx')
ds = Measure('ds')
dS = Measure("dS")

m_form = inner(v_p, al_p) * dx + inner(v_q, al_q) * dx

n_ver = FacetNormal(mesh)
s_ver = as_vector([-n_ver[1], n_ver[0]])

e_mnn = inner(e_q, outer(n_ver, n_ver))
v_mnn = inner(v_q, outer(n_ver, n_ver))

e_mns = inner(e_q, outer(n_ver, s_ver))
v_mns = inner(v_q, outer(n_ver, s_ver))

j_1 = - inner(grad(grad(v_p)), e_q) * dx \
      + jump(grad(v_p), n_ver) * dot(dot(e_q('+'), n_ver('+')), n_ver('+')) * dS \
      + dot(grad(v_p), n_ver) * dot(dot(e_q, n_ver), n_ver) * ds

j_2 = + inner(v_q, grad(grad(e_p))) * dx \
      - dot(dot(v_q('+'), n_ver('+')), n_ver('+')) * jump(grad(e_p), n_ver) * dS \
      - dot(dot(v_q, n_ver), n_ver) * dot(grad(e_p), n_ver) * ds


j_form = j_1 + j_2

bc_1, bc_2, bc_3, bc_4 = bc_input

bc_dict = {1: bc_1, 3: bc_2, 2: bc_3, 4: bc_4}

# The boundary edges in this mesh are numbered as follows:

# 1: plane x == 0
# 2: plane x == 1
# 3: plane y == 0
# 4: plane y == 1

bcs = []
boundary_dofs = []

for key, val in bc_dict.items():

    if val == 'C':
        bc_p = DirichletBC(Vp, Constant(0.0), key)
        for node in bc_p.nodes:
            boundary_dofs.append(node)
        bcs.append(bc_p)

    elif val == 'S':
        bc_p = DirichletBC(Vp, Constant(0.0), key)
        bc_q = DirichletBC(Vq, Constant(((0.0, 0.0), (0.0, 0.0))), key)

        for node in bc_p.nodes:
            boundary_dofs.append(node)
        bcs.append(bc_p)

        for node in bc_q.nodes:
            boundary_dofs.append(n_Vp + node)
        bcs.append(bc_q)

    elif val == 'F':
        bc_q = DirichletBC(Vq, Constant(((0.0, 0.0), (0.0, 0.0))), key)
        for node in bc_q.nodes:
            boundary_dofs.append(n_Vp + node)
        bcs.append(bc_q)

boundary_dofs = sorted(boundary_dofs)
n_lmb = len(boundary_dofs)

G = np.zeros((n_V, n_lmb))
for (i, j) in enumerate(boundary_dofs):
    G[j, i] = 1

J = assemble(j_form, mat_type='aij')
M = assemble(m_form, mat_type='aij')

petsc_j = J.M.handle
petsc_m = M.M.handle

JJ = np.array(petsc_j.convert("dense").getDenseArray())
MM = np.array(petsc_m.convert("dense").getDenseArray())

Z_lmb = np.zeros((n_lmb, n_lmb))

J_aug = np.vstack([np.hstack([JJ, G]),
                   np.hstack([-G.T, Z_lmb])
                ])

E_aug = la.block_diag(MM, Z_lmb)

v_omn = dot(grad(v_p), n_ver)
V_mnn = FunctionSpace(mesh, 'Lagrange', 1)
M_nn = TrialFunction(V_mnn)
B_f = assemble(v_omn * M_nn * ds(2), mat_type='aij')
petsc_b_u = B_f.M.handle
B_u = np.array(petsc_b_u.convert("dense").getDenseArray())
u_dofs = np.where(B_u.any(axis=0))[0]  # np.where(~np.all(B_in == 0, axis=0) == True) #
B_u = B_u[:, u_dofs]
n_u = len(u_dofs)

# B_u = assemble(v_p * ds(2), mat_type='aij').vector().get_local()
B_aug = np.concatenate((B_u, np.zeros((n_lmb, n_u))), axis=0)
order = []

om_f = 3.4699366066454016/norm_coeff

# invMM = la.inv(MM)

# Simulate
t_final = 1
n_ev = 200
t_ev = np.linspace(0, t_final, n_ev)


def dae_closed_phs(t, y, yd):

    u = sin(om_f*t) * (t > 0.01*t_final)

    res = E_aug @ yd - J_aug @ y - B_aug @ np.ones((n_u,)) * u
    # res = E_aug @ yd - J_aug @ y - B_aug * u

    return res

    # res_e = E_aug[:n_e, :] @ yd - J_aug[:n_e, :] @ y - B_aug[:n_e] * u
    # res_lmb = G.T @ invMM @ (J_aug[:n_e, :] @ y + B_aug[:n_e] * u)
    #
    # return np.concatenate((res_e, res_lmb))


def handle_result(solver, t, y, yd):

    order.append(solver.get_last_order())

    solver.t_sol.extend([t])
    solver.y_sol.extend([y])
    solver.yd_sol.extend([yd])

    # The initial conditons

y0 = np.zeros(n_V + n_lmb)  # Initial conditions
yd0 = np.zeros(n_V + n_lmb)  # Initial conditions

# Create an Assimulo implicit problem
imp_mod = Implicit_Problem(dae_closed_phs, y0, yd0, name='dae_closed_pHs')
imp_mod.handle_result = handle_result

# Set the algebraic components
imp_mod.algvar = list(np.concatenate((np.ones(n_V), np.zeros(n_lmb))))

# Create an Assimulo implicit solver (IDA)
imp_sim = IDA(imp_mod)  # Create a IDA solver

# Sets the paramters
imp_sim.atol = 1e-6  # Default 1e-6
imp_sim.rtol = 1e-6  # Default 1e-6
imp_sim.suppress_alg = True  # Suppres the algebraic variables on the error test
imp_sim.report_continuously = True
# imp_sim.maxh = 1e-6

# Let Sundials find consistent initial conditions by use of 'IDA_YA_YDP_INIT'
imp_sim.make_consistent('IDA_YA_YDP_INIT')


t_sol, y_sol, yd_sol = imp_sim.simulate(t_final, 0, t_ev)
e_sol = y_sol[:, :n_V].T
lmb_sol = y_sol[:, n_V:].T

ep_sol = e_sol[:n_Vp, :]
w0 = np.zeros((n_Vp,))
w_sol = np.zeros(ep_sol.shape)
w_sol[:, 0] = w0
w_old = w_sol[:, 0]
n_ev = len(t_sol)
dt_vec = np.diff(t_sol)


for i in range(1, n_ev):

    w_sol[:, i] = w_old + 0.5 * (ep_sol[:, i - 1] + ep_sol[:, i]) * dt_vec[i-1]
    w_old = w_sol[:, i]

wi_fun = Function(Vp)
w_vec = []


for i in range(n_ev):
    wi_fun.vector()[:] = w_sol[:, i]
    w_vec.append(interpolate(wi_fun, Vp))
    # w_vec.append(wi_fun)

maxZ = np.max(w_sol)
minZ = np.min(w_sol)


fntsize = 16
# H_vec = np.zeros((n_ev,))
#
# for i in range(n_ev):
#     H_vec[i] = 0.5 * (e_sol[:, i].T @ MM @ e_sol[:, i])
#
# fig, ax = plt.subplots()
# ax.yaxis.set_major_formatter(FormatStrFormatter('%1.2g'))
# plt.plot(t_ev, H_vec, 'b-', label='Hamiltonian Plate (J)')
# plt.xlabel(r'{Time} (s)', fontsize=fntsize)
# plt.ylabel(r'{Hamiltonian} (J)', fontsize=fntsize)
# plt.title(r"Hamiltonian trend",
#           fontsize=fntsize)
# plt.legend(loc='upper left')
# path_out = "/home/a.brugnoli/Plots_Videos/Plots/Kirchhoff_plots/Simulations/Article_CDC/DampingInjection/"
# plt.savefig(path_out + "Hamiltonian.eps", format="eps")

anim = animate2D(minZ, maxZ, w_vec, t_ev, xlabel = '$x[m]$', ylabel = '$y [m]$', \
                         zlabel='$w [m]$', title = 'Vertical Displacement')

plt.show()

