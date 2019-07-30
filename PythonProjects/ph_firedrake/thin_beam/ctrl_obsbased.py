from firedrake import *
import numpy as np
import scipy.linalg as la
from system_components.beams import FreeEB, ClampedEB, draw_allbending
from modules_phdae.classes_phsystem import SysPhdaeRig
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from control import lqr

plt.rc('text', usetex=True)

E = 2e11
rho = 7900  # kg/m^3
nu = 0.3

b = 0.05
h = 0.01
A = b * h

I = 1./12 * b * h**3

EI = E * I

fntsize = 15
n_el = 10
L = 1
frac = 2
per = 1/frac
L1 = per * L
L2 = (1-per) * L

n_el1 = int(n_el/frac)
n_el2 = n_el - n_el1

beamCC = ClampedEB(n_el2, L2, rho, A, E, I)
beamFF = FreeEB(n_el1, L1, rho, A, E, I)

beamCF = SysPhdaeRig.gyrator_ordered(beamCC, beamFF, [2, 3], [0, 1], np.eye(2))

mesh1 = IntervalMesh(n_el1, L1)
mesh2 = IntervalMesh(n_el2, L2)

# Finite element defition
deg = 3
Vp1 = FunctionSpace(mesh1, "Hermite", deg)
Vp2 = FunctionSpace(mesh2, "Hermite", deg)

n_e = beamCF.n
n_p = beamCF.n_p
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

Md = beamCF.E
Jd = beamCF.J
Bd = beamCF.B

Qd = la.inv(Md)

Ad = Jd @ Qd
Cd = Bd.T @ Qd

Q0 = 20 * np.eye(n_e)
R0 = np.eye(beamCF.m)

K, S, Eig = lqr(Ad.T, Cd.T, Q0, R0)

print(Eig)

Bc = K.T

alpha = 0
beta = 400
gamma = 0.006
delta = 0.5

# import cvxpy as cp
# G = cp.Variable(10)
# # define constraints
# constraints = [ G * (A1-b1*k1.T) + (A1-b1*k1.T).T * G << 0,
#                 G * (A1-b1*k2.T) + (A1-b1*k2.T).T * G << 0 ]
#
# # create optimization problem to decide feasibility
# prob = cp.Problem(cp.Minimize(0), constraints)
#
# # solve optimization problem and print results
# print(prob.solve())
# print(G.value)


# t0 = 0.0
# t_fin = 1
# n_t = 100
# t_span = [t0, t_fin]
#
#
# def sys(t,y):
#
#     w1 = 0.5 * (np.cosh(sqrom_r) + np.cos(sqrom_r)) * omega_r * np.cos(omega_r * t)
#     wx1 = sqrom_r/2 * (np.sinh(sqrom_r) - np.sin(sqrom_r)) * omega_r * np.cos(omega_r * t)
#
#     dydt = invM @ (Jmat @ y + Bmat @ [w1, wx1])
#     return dydt
#
#
# y0 = np.zeros(n_e,)
# y0[:n_p1] = v1_0.vector().get_local()
# y0[n_p1:n_p] = v2_0.vector().get_local()
#
# t_ev = np.linspace(t0, t_fin, num=n_t)
#
# sol = integrate.solve_ivp(sys, t_span, y0, method='BDF', t_eval=t_ev)
# t_ev = sol.t
# n_ev = len(t_ev)
# dt_vec = np.diff(t_ev)
#
# e_sol = sol.y
# ep_sol = e_sol[:n_p, :]
#
# w0 = np.zeros((n_p,))
# w_sol = np.zeros(ep_sol.shape)
# w_sol[:, 0] = w0
# w_old = w0
#
#
# for i in range(1, n_ev):
#     w_sol[:, i] = w_old + 0.5 * (ep_sol[:, i - 1] + ep_sol[:, i]) * dt_vec[i-1]
#     w_old = w_sol[:, i]
#
#
# fntsize = 16
# # plt.figure()
# # plt.plot(t_ev, ep_sol[0, :], 'bo', label='Simulated')
# # plt.plot(t_ev, omega_r * np.cos(omega_r * t_ev), 'r--', label='Reference')
# # plt.xlabel(r'{Time} (s)', fontsize=fntsize)
# # plt.ylabel(r'w [m]', fontsize=fntsize)
# # plt.title(r"Vertical velocity", fontsize=fntsize)
# # plt.legend(loc='upper left')
# #
# # plt.figure()
# # plt.plot(t_ev, w_sol[0, :], 'b*', label='Simulated')
# # plt.plot(t_ev, np.sin(omega_r * t_ev), 'r-', label='Reference')
# # plt.xlabel(r'{Time} (s)', fontsize=fntsize)
# # plt.ylabel(r'w [m]', fontsize=fntsize)
# # plt.title(r"Vertical displacement", fontsize=fntsize)
# # plt.legend(loc='upper left')
# #
# # plt.show()
#
# n_plot = 30
# n_plot1 = int(n_plot/frac)
# n_plot2 = n_plot - n_plot1
#
# v_plot = np.zeros((n_plot, n_ev))
# w_plot = np.zeros((n_plot, n_ev))
#
# x1_plot = np.linspace(0, L1, n_plot1)
# x2_plot = np.linspace(L1, L, n_plot2)
#
# x_plot = np.concatenate((x1_plot, x2_plot))
#
# t_plot = t_ev
#
# w1_old = w_plot[:n_plot1, 0]
# w2_old = w_plot[n_plot1:n_plot, 0]
#
# for i in range(n_ev):
#     v1_i = draw_allbending(n_plot1, [0, 0, 0], ep_sol[:n_p1, i], L1)[2]
#     v2_i = draw_allbending(n_plot2, [0, 0, 0], ep_sol[n_p1:n_p, i], L2)[2]
#
#     v_plot[:n_plot1, i] = v1_i
#     v_plot[n_plot1:n_plot, i] = v2_i
#
#     if i > 0:
#         w_plot[:n_plot1, i] = w1_old + 0.5 * (v_plot[:n_plot1, i - 1] + v_plot[:n_plot1, i]) * dt_vec[i-1]
#         w_plot[n_plot1:n_plot, i] = w2_old + 0.5 * (v_plot[n_plot1:n_plot, i - 1] + v_plot[n_plot1:n_plot, i]) * dt_vec[i-1]
#
#         w1_old = w_plot[:n_plot1, i]
#         w2_old = w_plot[n_plot1:n_plot, i]
# # from matplotlib import animation
# #
# # # First set up the figure, the axis, and the plot element we want to animate
# # fig = plt.figure(1)
# # ax = plt.axes(xlim=(0, L), ylim=(np.min(w_plot), np.max(w_plot)))
# # line, = ax.plot([], [], lw=2)
# #
# # # initialization function: plot the background of each frame
# # def init():
# #     line.set_data(x_plot, w_plot[:, 0])
# #     return line,
# #
# # # animation function.  This is called sequentially
# # def animate(i):
# #     line.set_data(x_plot, w_plot[:, i])
# #     return line,
# #
# # # call the animator.  blit=True means only re-draw the parts that have changed.
# #
# #
# # anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(t_ev), interval=20, blit=False)
#
# # save the animation as an mp4.  This requires ffmpeg or mencoder to be
# # installed.  The extra_args ensure that the x264 codec is used, so that
# # the video can be embedded in html5.  You may need to adjust this for
# # your system: for more information, see
# # http://matplotlib.sourceforge.net/api/animation_api.html
# # anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
#
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
# surf = ax.plot_surface(X_plot, T_plot, W_plot, cmap=cm.jet, linewidth=0, antialiased=False, label='Beam $w$')
#
# surf._facecolors2d = surf._facecolors3d
# surf._edgecolors2d = surf._edgecolors3d
#
# x0 = np.zeros((n_ev,))
# w0_plot = ax.plot(x0, t_ev, np.sin(omega_r * t_ev), label='Reference $w$', color='black')
# ax.legend(handles=[w0_plot[0]])
#
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.show()
