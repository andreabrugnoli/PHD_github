
from firedrake import *
import numpy as np
import scipy.linalg as la

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from assimulo.solvers import IDA
from assimulo.implicit_ode import Implicit_Problem
from system_components.beams import draw_allbending

plt.close('all')
matplotlib.rcParams['text.usetex'] = True

E = 2e11
rho = 7900  # kg/m^3
nu = 0.3

b = 0.05
h = 0.01
A = b * h

I = 1./12 * b * h**3

EI = E * I
L = 1


n_elem = 1
deg = 3


# The unit square mesh is divided in :math:`N\times N` quadrilaterals::
L = 1

mesh = IntervalMesh(n_elem, L)

# plot(mesh)
# plt.show()


# Finite element defition

V_p = FunctionSpace(mesh, "Hermite", deg)
V_q = FunctionSpace(mesh, "Hermite", deg)

V = V_p * V_q
n_e = V.dim()
n_Vp = V_p.dim()
n_Vq = V_q.dim()

n = V.dim()
n_p = V_p.dim()

v = TestFunction(V)
v_p, v_q = split(v)

e_v = TrialFunction(V)
e_p, e_q = split(e_v)

al_p = rho * A * e_p
al_q = 1./EI * e_q

dx = Measure('dx')
ds = Measure('ds')
m_p = v_p * al_p * dx
m_q = v_q * al_q * dx
m = m_p + m_q

j_divDiv = -v_p * e_q.dx(0).dx(0) * dx
j_divDivIP = v_q.dx(0).dx(0) * e_p * dx

j_gradgrad = v_q * e_p.dx(0).dx(0) * dx
j_gradgradIP = -v_p.dx(0).dx(0) * e_q * dx

j = j_gradgrad + j_gradgradIP
# j = j_divDiv + j_divDivIP

# bc_w = DirichletBC(V.sub(0), Constant(0.0), 1)
# bc_M = DirichletBC(V.sub(0), Constant(0.0), 2)
# boundary_dofs = sorted(bc_w.nodes)

gCC_Hess = [- v_p * ds(1), + v_p.dx(0) * ds(1), - v_p * ds(2), v_p.dx(0) * ds(2)]
gSS_Hess = [- v_p * ds(1), - v_p * ds(2)]

gCF_Hess = [- v_p * ds(1), + v_p.dx(0) * ds(1)]

gFF_divDiv = [v_q * ds(1), - v_q.dx(0) * ds(1), + v_q * ds(2), - v_q.dx(0) * ds(2)]
gSS_divDiv = [v_q * ds(1), v_q * ds(2)]

gCF_divDiv = [+ v_q * ds(2), - v_q.dx(0) * ds(2)]

g_l = gCF_Hess
g_r = [] # gCF_divDiv

G_L = np.zeros((n, len(g_l)))
G_R = np.zeros((n, len(g_r)))


for counter, item in enumerate(g_l):
    G_L[:, counter] = assemble(item).vector().get_local()

for counter, item in enumerate(g_r):
    G_R[:, counter] = assemble(item).vector().get_local()

G = np.concatenate((G_L, G_R), axis=1)

# Assemble the stiffness matrix and the mass matrix.
J = assemble(j, mat_type='aij')
M = assemble(m, mat_type='aij')

petsc_j = J.M.handle
petsc_m = M.M.handle

JJ = np.array(petsc_j.convert("dense").getDenseArray())
MM = np.array(petsc_m.convert("dense").getDenseArray())

n_lmb = len(G.T)
# print(N_u)

Z_u = np.zeros((n_lmb, n_lmb))


J_aug = np.vstack([ np.hstack([JJ, G]),
                    np.hstack([-G.T, Z_u])
                ])

E_aug = la.block_diag(MM, Z_u)

B_f = assemble(v_p.dx(0) * ds(2)).vector().get_local()
B_aug = np.concatenate((B_f, np.zeros((n_lmb, ))), axis=0)
order = []

om_f = (1.8754760215949022/(L*(rho*A/(EI))**(0.25)))**2
invMM = la.inv(MM)


def dae_closed_phs(t, y, yd):

    u = sin(om_f*t) * (t > 0.01)

    res = E_aug @ yd - J_aug @ y - B_aug * u
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


y0 = np.zeros(n_e + n_lmb)  # Initial conditions
yd0 = np.zeros(n_e + n_lmb)  # Initial conditions

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
t_final = 1
n_ev = 1000
t_ev = np.linspace(0, t_final, n_ev)
t_sol, y_sol, yd_sol = imp_sim.simulate(t_final, 0, t_ev)
e_sol = y_sol[:, :n_e].T
lmb_sol = y_sol[:, n_e:].T

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


n_plot = 30
w_plot = np.zeros((n_plot, n_ev))
e_plot = np.zeros((n_plot, n_ev))

x_plot = np.linspace(0, L, n_plot)
t_plot = t_sol

for i in range(n_ev):
    e_plot[:, i] = draw_allbending(n_plot, [0,0,0], ep_sol[:, i], L)[2]
    if i > 1:
        w_plot[:, i] = w_plot[:, i-1] + 0.5 * (e_plot[:, i-1] + e_plot[:, i]) * dt_vec[i-1]

# Hpl_vec = np.zeros((n_ev,))
# for i in range(n_ev):
#     Hpl_vec[i] = 0.5 * (e_sol[:, i].T @ MM @ e_sol[:, i])
#
# fntsize = 16
# fig = plt.figure()
# plt.plot(t_ev, Hpl_vec, 'b-', label='Hamiltonian Plate (J)')
# plt.xlabel(r'{Time} (s)', fontsize = fntsize)
# plt.ylabel(r'{Hamiltonian} (J)', fontsize = fntsize)
# plt.title(r"Hamiltonian trend", fontsize=fntsize)
# # plt.legend(loc='upper left')

#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# X_plot, T_plot = np.meshgrid(x_plot, t_plot)
#
# # Customize the z axis.
#
#
# # Plot the surface.
# ax.set_xlabel('Space coordinate $[m]$', fontsize=fntsize)
# ax.set_ylabel('Time $[s]$', fontsize=fntsize)
# ax.set_zlabel('$w [m]$', fontsize=fntsize)
#
#
# W_plot = np.transpose(w_plot)
# ax.plot_surface(X_plot, T_plot, W_plot, cmap=cm.jet, linewidth=0, antialiased=False, label='Wave $w$')

from matplotlib import animation

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure(1)
ax = plt.axes(xlim=(0, L), ylim=(np.min(w_plot), np.max(w_plot)))
line, = ax.plot([], [], lw=2)

# initialization function: plot the background of each frame
def init():
    line.set_data(x_plot, w_plot[:, 0])
    return line,

# animation function.  This is called sequentially
def animate(i):
    line.set_data(x_plot, w_plot[:, i])
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(t_ev), interval=20, blit=False)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
# anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()
