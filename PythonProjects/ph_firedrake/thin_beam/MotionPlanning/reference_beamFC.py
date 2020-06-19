from firedrake import *
import numpy as np
import scipy.linalg as la
from system_components.beams import FreeEB, ClampedEB, draw_allbending
from modules_ph.classes_phsystem import SysPhdaeRig
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from firedrake.plot import calculate_one_dim_points

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})
rcParams['text.usetex'] = True

pathout = "/home/a.brugnoli/Plots/Python/Plots/EulerBernoulli/MotionPlanning/"


tev_DAE = np.load("t_evDAE.npy")
w0_DAE = np.load("w0_solDAE.npy")
ep0_DAE = np.load("ep0_solDAE.npy")

n_el = 6
L = 1
frac = 2
per = 1/frac
L1 = per * L
L2 = (1-per) * L

n_el1 = int(n_el/frac)
n_el2 = n_el - n_el1

beamFF = FreeEB(n_el1, L1, 1, 1, 1, 1)
beamCC = ClampedEB(n_el2, L2, 1, 1, 1, 1)

beamFC = SysPhdaeRig.gyrator_ordered(beamFF, beamCC, [2, 3], [0, 1], -np.eye(2))

mesh1 = IntervalMesh(n_el1, L1)
mesh2 = IntervalMesh(n_el2, L2)

# Finite element defition
deg = 3
Vp1 = FunctionSpace(mesh1, "Hermite", deg)
Vp2 = FunctionSpace(mesh2, "DG", 1)

n_e = beamFC.n
n_p = beamFC.n_p
n_p1 = beamFF.n_p
n_p2 = beamCC.n_p

x1 = SpatialCoordinate(mesh1)[0]
x2 = SpatialCoordinate(mesh2)[0]

omega_r = 4
sqrom_r = np.sqrt(omega_r)
exp_v1 = 0.5 * (cosh(sqrom_r * x1) + cos(sqrom_r * x1)) * omega_r
exp_v2 = 0.5 * (cosh(sqrom_r * (L1 + x2)) + cos(sqrom_r * (L1 + x2))) * omega_r

v1_0 = Function(Vp1)
v2_0 = Function(Vp2)

v1_0.assign(project(exp_v1, Vp1))
v2_0.assign(project(exp_v2, Vp2))

Mmat = beamFC.E
Jmat = beamFC.J
Bmat = beamFC.B[:, 2:]

invM = la.inv(Mmat)

t0 = 0.0
t_fin = 1
n_t = 100
t_span = [t0, t_fin]

def sys(t,y):

    w1 = 0.5 * (np.cosh(sqrom_r) + np.cos(sqrom_r)) * omega_r * np.cos(omega_r * t)
    wx1 = sqrom_r/2 * (np.sinh(sqrom_r) - np.sin(sqrom_r)) * omega_r * np.cos(omega_r * t)

    dydt = invM @ (Jmat @ y + Bmat @ [w1, wx1])
    return dydt


y0 = np.zeros(n_e,)
y0[:n_p1] = v1_0.vector().get_local()
y0[n_p1:n_p] = v2_0.vector().get_local()

t_ev = np.linspace(t0, t_fin, num=n_t)

sol = integrate.solve_ivp(sys, t_span, y0, method='BDF', t_eval=t_ev)
t_ev = sol.t
n_ev = len(t_ev)
dt_vec = np.diff(t_ev)

e_sol = sol.y
ep_sol = e_sol[:n_p, :]

w0 = np.zeros((n_p,))
w_sol = np.zeros(ep_sol.shape)
w_sol[:, 0] = w0
w_old = w0


for i in range(1, n_ev):
    w_sol[:, i] = w_old + 0.5 * (ep_sol[:, i - 1] + ep_sol[:, i]) * dt_vec[i-1]
    w_old = w_sol[:, i]


plt.figure()
plt.plot(t_ev, ep_sol[0, :], 'g1', label='Simulated VDD')
plt.plot(tev_DAE, ep0_DAE, 'bo', label='Simulated DAE', mfc='none')
plt.plot(t_ev, omega_r * np.cos(omega_r * t_ev), 'r--', label='Reference')
plt.xlabel(r'{Time} $\mathrm{[s]}$')
plt.ylabel(r'$e_w \; \mathrm{[m]}$')
plt.title(r"Vertical velocity")
plt.legend(loc='best')

plt.savefig(pathout + "ref_vel.eps", format="eps")

plt.figure()
plt.plot(t_ev, w_sol[0, :], 'g1', label='Simulated VDD')
plt.plot(tev_DAE, w0_DAE, 'bo', label='Simulated DAE', mfc='none')
plt.plot(t_ev, np.sin(omega_r * t_ev), 'r--', label='Reference')
plt.xlabel(r'{Time} $\mathrm{[s]}$')
plt.ylabel(r'$w \; \mathrm{[m]}$')
plt.title(r"Vertical displacement")
plt.legend(loc='best')

plt.savefig(pathout + "ref_w.eps", format="eps")


# n_plot = 30
n_plot1 = 15 # int(n_plot/frac)


x1_plot = np.linspace(0, L1, n_plot1)
# x2_plot = np.linspace(L1, L, n_plot2)


t_plot = t_ev

v2_f = Function(Vp2)

v1_i = draw_allbending(n_plot1, [0, 0, 0], ep_sol[:n_p1, 0], L1)[2]
# v2_i = draw_allbending(n_plot2, [0, 0, 0], ep_sol[n_p1:n_p, i], L2)[2]

v2_f.vector().set_local(ep_sol[n_p1:n_p, 0])
x2_plot, v2_i = calculate_one_dim_points(v2_f, 10)

x2_plot += L1
x_plot = np.concatenate((x1_plot, x2_plot))

n_plot2 = len(x2_plot)

n_plot = n_plot1 + n_plot2
v_plot = np.zeros((n_plot, n_ev))
w_plot = np.zeros((n_plot, n_ev))

v_plot[:n_plot1, i] = v1_i
v_plot[n_plot1:n_plot, i] = v2_i

w1_old = w_plot[:n_plot1, 0]
w2_old = w_plot[n_plot1:n_plot, 0]


for i in range(1, n_ev):
    v1_i = draw_allbending(n_plot1, [0, 0, 0], ep_sol[:n_p1, i], L1)[2]
    # v2_i = draw_allbending(n_plot2, [0, 0, 0], ep_sol[n_p1:n_p, i], L2)[2]

    v2_f.vector().set_local(ep_sol[n_p1:n_p, i])
    v2_i = calculate_one_dim_points(v2_f, 10)[1]

    v_plot[:n_plot1, i] = v1_i
    v_plot[n_plot1:n_plot, i] = v2_i

    w_plot[:n_plot1, i] = w1_old + 0.5 * (v_plot[:n_plot1, i - 1] + v_plot[:n_plot1, i]) * dt_vec[i-1]
    w_plot[n_plot1:n_plot, i] = w2_old + 0.5 * (v_plot[n_plot1:n_plot, i - 1] + v_plot[n_plot1:n_plot, i]) * dt_vec[i-1]

    w1_old = w_plot[:n_plot1, i]
    w2_old = w_plot[n_plot1:n_plot, i]

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
ax.set_xlabel('Space coordinate $\mathrm{[m]}$')
ax.set_ylabel('Time $[s]$')
ax.set_zlabel('Vertical deflection $\mathrm{[m]}$')

# ax.set_title(r'Vertical deflection', fontsize=fntsize, loc='left')


W_plot = np.transpose(w_plot)
surf = ax.plot_surface(X_plot, T_plot, W_plot, cmap=cm.jet, linewidth=0, antialiased=False, label='Beam $w$')
# ax.set_title('Vertical displacement', loc='left')

surf._facecolors2d = surf._facecolors3d
surf._edgecolors2d = surf._edgecolors3d

fig.colorbar(surf, shrink=0.5, aspect=5)
x0 = np.zeros((n_ev,))
w0_plot = ax.plot(x0, t_ev, np.sin(omega_r * t_ev), label='Output $w(0, t)$', color='purple', linewidth=5)

u_t = 0.5 * (np.cosh(sqrom_r) + np.cos(sqrom_r)) * np.sin(omega_r * t_ev)
x1 = np.ones((n_ev,))
w1_plot = ax.plot(x1, t_ev, u_t, label='Control $w(1, t)$', color='black', linewidth=5)

ax.legend(handles=[w0_plot[0], w1_plot[0]], loc="best")

ax.view_init(azim=140)

plt.savefig(pathout + "plotW.eps", format="eps")


plt.show()
