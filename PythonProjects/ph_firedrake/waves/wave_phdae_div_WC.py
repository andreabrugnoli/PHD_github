from firedrake import *
import numpy as np
import scipy.linalg as la
np.set_printoptions(threshold=np.inf)
from matplotlib import pyplot as plt
from tools_plotting.animate_surf import animate2D
from assimulo.solvers import IDA
from assimulo.implicit_ode import Implicit_Problem
from matplotlib import animation
plt.rc('text', usetex=True)
# Finite element defition

# The unit square mesh is divided in :math:`N\times N` quadrilaterals::
# ind = 9
path_mesh = "/home/a.brugnoli/GitProjects/PythonProjects/ph_firedrake/waves/meshes_ifacwc/"
mesh = Mesh(path_mesh + "duct_test" + ".msh")

figure = plt.figure()
ax = figure.add_subplot(111)
plot(mesh, axes=ax)
plt.show()

q_1 = 0.8163  # m^3 kg ^-1   1/mu_0
mu_0 = 1/q_1

q_2 = 1.4161 * 10**5  # Pa   1/xsi
xi_s = 1/q_2
# c_0 = 340  # 340 m/s

# rho = 0.1
# T = as_tensor([[10, 0], [0, 10]])

deg_p = 1
deg_q = 2
Vp = FunctionSpace(mesh, "CG", deg_p)
Vq = FunctionSpace(mesh, "RT", deg_q)
# Vq = VectorFunctionSpace(mesh, "Lagrange", deg_q)

V = Vp * Vq

n_V = V.dim()
print(n_V)

v = TestFunction(V)
v_p, v_q = split(v)

e_v = TrialFunction(V)
e_p, e_q = split(e_v)

al_p = xi_s * e_p
al_q = mu_0 * e_q

x, r = SpatialCoordinate(mesh)
R_ext = 1
L_duct = 2
tol_geo = 1e-6

dx = Measure('dx')
ds = Measure('ds')

m_p = v_p * al_p * r * dx
m_q = dot(v_q, al_q) * r * dx
m_form = m_p + m_q

j_div = v_p * (div(e_q) + e_q[1]/r) * r * dx
j_divIP = -(div(v_q) + v_q[1]/r) * e_p * r * dx

j_form = j_div + j_divIP
petsc_j = assemble(j_form, mat_type='aij').M.handle
petsc_m = assemble(m_form, mat_type='aij').M.handle

JJ = np.array(petsc_j.convert("dense").getDenseArray())
MM = np.array(petsc_m.convert("dense").getDenseArray())

n_p = Vp.dim()
n_q = Vq.dim()

n_e = V.dim()

n_ver = FacetNormal(mesh)

# B matrices based on Lagrange
V_bc = FunctionSpace(mesh, 'CG', 1)
lmb_N = TrialFunction(V_bc)
v_N = TestFunction(V_bc)
u_D = TrialFunction(V_bc)

is_uD = conditional(And(gt(r, R_ext - tol_geo), And(gt(x, L_duct/3), lt(x, 2*L_duct/3))), 1, 0)
b_D = dot(v_q, n_ver) * u_D * is_uD * ds

petsc_bD = assemble(b_D, mat_type='aij').M.handle

B_D = np.array(petsc_bD.convert("dense").getDenseArray())
controlD_dofs = np.where(B_D.any(axis=0))[0]

B_D = B_D[:, controlD_dofs]

is_lmbN = conditional(lt(r, R_ext - tol_geo), 1, 0)
g_N = dot(v_q, n_ver) * lmb_N * is_lmbN * ds

petsc_gN = assemble(g_N, mat_type='aij').M.handle
G_N = np.array(petsc_gN.convert("dense").getDenseArray())

neumann_dofs = np.where(G_N.any(axis=0))[0]
G_N = G_N[:, neumann_dofs]

is_uN = conditional(lt(x, tol_geo), 1, 0)
u_Nxy = sin(pi*r/R_ext)
b_N = v_N * u_Nxy * is_uN * r * ds

B_N = assemble(b_N).vector().get_local()[neumann_dofs]

n_lmb = G_N.shape[1]
n_uD = B_D.shape[1]
t_final = 0.01

Z = 1000
Amp_uNxy = 0.1

t_diss = 0.05*t_final

def dae_closed_phs(t, y, yd):

    print(t/t_final*100)

    e_var = y[:n_e]
    lmb_var = y[n_e:]
    ed_var = yd[:n_e]

    res_e = MM @ ed_var - JJ @ e_var - G_N @ lmb_var + Z * B_D @ B_D.T @ e_var * (t>t_diss)
    res_lmb = - G_N.T @ e_var + B_N * Amp_uNxy * (1 - np.cos(pi*t/t_diss)) # *(t<t_diss)

    return np.concatenate((res_e, res_lmb))


order = []


def handle_result(solver, t, y, yd):

    order.append(solver.get_last_order())

    solver.t_sol.extend([t])
    solver.y_sol.extend([y])
    solver.yd_sol.extend([yd])


# invMM = la.inv(MM)
ep_0 = np.zeros(n_p)
eq_0 = np.zeros(n_q)

e_0 = np.concatenate((ep_0, eq_0))
y0 = np.zeros(n_e + n_lmb)  # Initial conditions

# de_0 = invMM @ (JJ @ e_0 + G_N @ lmb_0 + B_Dxy)
# dlmb_0 = la.solve(- G_N.T @ invMM @ G_N, G_N.T @ invMM @ (JJ @ de_0 + B_Dt))

y0[:n_p] = ep_0
y0[n_p:n_V] = eq_0
# y0[n_V:] = lmb_0

yd0 = np.zeros(n_e + n_lmb)  # Initial conditions
# yd0[:n_V] = de_0
# yd0[n_V:] = dlmb_0

# Create an Assimulo implicit problem
imp_mod = Implicit_Problem(dae_closed_phs, y0, yd0, name='dae_closed_pHs')
imp_mod.handle_result = handle_result

# Set the algebraic components
imp_mod.algvar = list(np.concatenate((np.ones(n_e), np.zeros(n_lmb))))

# Create an Assimulo implicit solver (IDA)
imp_sim = IDA(imp_mod)  # Create a IDA solver

# Sets the paramters
imp_sim.atol = 1e-6  # Default 1e-6
imp_sim.rtol = 1e-6  # Default 1e-6
imp_sim.suppress_alg = True  # Suppress the algebraic variables on the error test
imp_sim.report_continuously = True
# imp_sim.maxh = 1e-6

# Let Sundials find consistent initial conditions by use of 'IDA_YA_YDP_INIT'
imp_sim.make_consistent('IDA_YA_YDP_INIT')

# Simulate
n_ev = 500
t_ev = np.linspace(0, t_final, n_ev)
t_sol, y_sol, yd_sol = imp_sim.simulate(t_final, 0, t_ev)
e_sol = y_sol[:, :n_e].T
lmb_sol = y_sol[:, n_e:].T

ep_sol = e_sol[:n_p, :]

n_ev = len(t_sol)
dt_vec = np.diff(t_sol)

maxZ = np.max(ep_sol)
minZ = np.min(ep_sol)
wfun_vec = []
w_fun = Function(Vp)

for i in range(n_ev):
    w_fun.vector()[:] = ep_sol[:, i]
    wfun_vec.append(interpolate(w_fun, Vp))

H_vec = np.zeros((n_ev,))

for i in range(n_ev):
    H_vec[i] = 0.5 * (e_sol[:, i].T @ MM @ e_sol[:, i])

# # np.save("t_dae.npy", t_sol)
# # np.save("H_dae.npy", H_vec)
fntsize = 16

fig = plt.figure()
plt.plot(t_ev, H_vec, 'b-')
plt.xlabel(r'{Time} (s)', fontsize=fntsize)
plt.ylabel(r'{Hamiltonian} (J)', fontsize=fntsize)
plt.title(r"Hamiltonian trend",
          fontsize=fntsize)
# plt.legend(loc='upper left')
#
#
# anim = animate2D(minZ, maxZ, wfun_vec, t_ev, xlabel = '$x[m]$', ylabel = '$r [m]$', \
#                           zlabel='$p$', title='pressure')

plt.show()

# rallenty = 10
# fps = 20
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
# path_out = "./"
# anim.save(path_out + 'wave_dae.mp4', writer=writer)

