from firedrake import *
import numpy as np
import scipy.linalg as la
from system_components.beams import FreeEB, ClampedEB, draw_allbending
from modules_phdae.classes_phsystem import SysPhdaeRig
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib import cm
from assimulo.solvers import IDA
from assimulo.implicit_ode import Implicit_Problem
from mpl_toolkits.mplot3d import Axes3D

plt.rc('text', usetex=True)

fntsize = 15
n_el = 6
L = 1
frac = 2
per = 1/frac

n_el1 = int(n_el/frac)


beamCC = ClampedEB(n_el, L, 1, 1, 1, 1)

mesh= IntervalMesh(n_el, L)

# Finite element defition
deg = 3
Vp = FunctionSpace(mesh, "Hermite", deg)

n_e = beamCC.n
n_p = beamCC.n_p
n_q = beamCC.n_q

x = SpatialCoordinate(mesh)[0]

omega_r = 4
sqrom_r = np.sqrt(omega_r)
exp_v = 0.5 * (cosh(sqrom_r * x) + cos(sqrom_r * x)) * omega_r

v_0 = Function(Vp)

v_0.assign(project(exp_v, Vp))

Mmat = beamCC.E
Jmat = beamCC.J
Bmat = beamCC.B[:, 2:]
Gmat = beamCC.B[:, :2]

n_lmb = Gmat.shape[1]

invM = la.inv(Mmat)

t0 = 0.0
t_fin = 1
n_t = 100
t_span = [t0, t_fin]


def dae_closed_phs(t, y, yd):
    print(t / t_fin * 100)

    e_var = y[:n_e]
    lmb_var = y[n_e:]
    ed_var = yd[:n_e]

    w1 = 0.5 * (np.cosh(sqrom_r) + np.cos(sqrom_r)) * omega_r * np.cos(omega_r * t)
    wx1 = sqrom_r / 2 * (np.sinh(sqrom_r) - np.sin(sqrom_r)) * omega_r * np.cos(omega_r * t)

    res_e = ed_var - invM @ (Jmat @ e_var + Bmat @ [w1, wx1] + Gmat @ lmb_var)
    res_lmb = - Gmat.T @ e_var

    return np.concatenate((res_e, res_lmb))


order = []


def handle_result(solver, t, y, yd):
    order.append(solver.get_last_order())

    solver.t_sol.extend([t])
    solver.y_sol.extend([y])
    solver.yd_sol.extend([yd])


y0 = np.zeros(n_e,)
y0[:n_p] = v_0.vector().get_local()

ep_0 = v_0.vector().get_local()
eq_0 = np.zeros(n_q,)

e_0 = np.concatenate((ep_0, eq_0))

w1_0 = 0.5 * (np.cosh(sqrom_r) + np.cos(sqrom_r)) * omega_r
wx1_0 = sqrom_r / 2 * (np.sinh(sqrom_r) - np.sin(sqrom_r)) * omega_r

lmb_0 = la.solve(- Gmat.T @ la.solve(Mmat, Gmat), Gmat.T @ la.solve(Mmat, Jmat @ e_0 + Bmat @ [w1_0, wx1_0]))

y0 = np.zeros(n_e + n_lmb)  # Initial conditions

de_0 = la.solve(Mmat, Jmat @ e_0 + Gmat @ lmb_0 + Bmat @ [w1_0, wx1_0])
dlmb_0 = la.solve(- Gmat.T @ la.solve(Mmat, Gmat), Gmat.T @ la.solve(Mmat, Jmat @ de_0))

y0[:n_p] = ep_0
y0[n_p:n_e] = eq_0
y0[n_e:] = lmb_0

yd0 = np.zeros(n_e + n_lmb)  # Initial conditions
yd0[:n_e] = de_0
yd0[n_e:] = dlmb_0

# Maug = la.block_diag(MM, np.zeros((n_lmb, n_lmb)))
# Jaug = la.block_diag(JJ, np.zeros((n_lmb, n_lmb)))
# Jaug[:n_e, n_e:] = G_D
# Jaug[n_e:, :n_e] = -G_D.T

# print_modes(Maug, Jaug, Vp, 10)

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
t_ev = np.linspace(0, t_fin, n_ev)

t_sol, y_sol, yd_sol = imp_sim.simulate(t_fin, 0, t_ev)
dt_vec = np.diff(t_sol)



t_sol = np.array(t_sol)
e_sol = y_sol[:, :n_e].T
lmb_sol = y_sol[:, n_e:].T

ep_sol = e_sol[:n_p, :]
eq_sol = e_sol[n_p:, :]


w0 = np.zeros((n_p,))
w_sol = np.zeros(ep_sol.shape)
w_sol[:, 0] = w0
w_old = w0


for i in range(1, n_ev):
    w_sol[:, i] = w_old + 0.5 * (ep_sol[:, i - 1] + ep_sol[:, i]) * dt_vec[i-1]
    w_old = w_sol[:, i]

fntsize = 16
# plt.figure()
# plt.plot(t_ev, ep_sol[0, :], 'bo', label='Simulated')
# plt.plot(t_ev, omega_r * np.cos(omega_r * t_ev), 'r--', label='Reference')
# plt.xlabel(r'{Time} (s)', fontsize=fntsize)
# plt.ylabel(r'w [m]', fontsize=fntsize)
# plt.title(r"Vertical velocity", fontsize=fntsize)
# plt.legend(loc='upper left')
#
# plt.figure()
# plt.plot(t_ev, w_sol[0, :], 'b*', label='Simulated')
# plt.plot(t_ev, np.sin(omega_r * t_ev), 'r-', label='Reference')
# plt.xlabel(r'{Time} (s)', fontsize=fntsize)
# plt.ylabel(r'w [m]', fontsize=fntsize)
# plt.title(r"Vertical displacement", fontsize=fntsize)
# plt.legend(loc='upper left')
#
# plt.show()

n_plot = 30

v_plot = np.zeros((n_plot, n_ev))
w_plot = np.zeros((n_plot, n_ev))

x_plot = np.linspace(0, L, n_plot)

t_plot = t_ev

w_old = w_plot[:, 0]

for i in range(n_ev):
    v_i = draw_allbending(n_plot, [0, 0, 0], ep_sol[:n_p, i], L)[2]

    v_plot[:n_plot, i] = v_i

    if i > 0:
        w_plot[:, i] = w_old + 0.5 * (v_plot[:, i - 1] + v_plot[:, i]) * dt_vec[i-1]

        w_old = w_plot[:n_plot, i]
# from matplotlib import animation
#
# # First set up the figure, the axis, and the plot element we want to animate
# fig = plt.figure(1)
# ax = plt.axes(xlim=(0, L), ylim=(np.min(w_plot), np.max(w_plot)))
# line, = ax.plot([], [], lw=2)
#
# # initialization function: plot the background of each frame
# def init():
#     line.set_data(x_plot, w_plot[:, 0])
#     return line,
#
# # animation function.  This is called sequentially
# def animate(i):
#     line.set_data(x_plot, w_plot[:, i])
#     return line,
#
# # call the animator.  blit=True means only re-draw the parts that have changed.
#
#
# anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(t_ev), interval=20, blit=False)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
# anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])


fig = plt.figure()
ax = fig.gca(projection='3d')
X_plot, T_plot = np.meshgrid(x_plot, t_plot)

# Customize the z axis.


# Plot the surface.
ax.set_xlabel('Space coordinate $[m]$', fontsize=fntsize)
ax.set_ylabel('Time $[s]$', fontsize=fntsize)
ax.set_zlabel('$w [m]$', fontsize=fntsize)


W_plot = np.transpose(w_plot)
surf = ax.plot_surface(X_plot, T_plot, W_plot, cmap=cm.jet, linewidth=0, antialiased=False, label='Beam $w$')

surf._facecolors2d = surf._facecolors3d
surf._edgecolors2d = surf._edgecolors3d

x0 = np.zeros((n_ev,))
w0_plot = ax.plot(x0, t_ev, np.sin(omega_r * t_ev), label='Reference $w$', color='black')
ax.legend(handles=[w0_plot[0]])

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
