from scipy.integrate import solve_ivp
import numpy as np
from math import pi

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import animation

import scipy.io

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

rcParams.update({'figure.autolayout': True})
rcParams['text.usetex'] = True
#rcParams['text.latex.preamble']=r"\usepackage{bm}"

from EnergyShapingPDE.func_wave import matrices_wave

init = ('x[0]', '0')

M, J, R, B, e0, dofs_dict, x_dict = matrices_wave(n_el=10, deg=1, e0_string=init)

dofs_v = dofs_dict['v']
x_v = x_dict['v']

dofs_sig = dofs_dict['sig']
x_sig = x_dict['sig']

A_sys = np.linalg.solve(M, J)
B_sys = np.linalg.solve(M, B)
C_sys= B.T

np.save('Data/A_sys', A_sys)
np.save('Data/B_sys', B_sys)
np.save('Data/C_sys', C_sys)
np.save('Data/e0', e0)
np.save('Data/dofs_dict', dofs_dict)
np.save('Data/x_dict', x_dict)

# omega, eigvectors= np.linalg.eig(A_sys)
#
# print(omega)
#
# scipy.io.savemat('Data.mat', {"A": A_sys, "B": B_sys, "C": C_sys, "x0": e0})
#
# def fun(t,y):
#
#    dydt = A_sys @ y # + B_sys * np.sin(pi*t) * (t<=1 or t>=5)
#
#    return dydt
#
# t0 = 0.0
# t_fin = 10
# t_span = [t0, t_fin]
#
# n_ev = 500
# t_ev = np.linspace(t0, t_fin, num=n_ev)
#
# sol = solve_ivp(fun, t_span, e0, method='RK45', t_eval = t_ev, \
#                       atol = 1e-5, rtol = 1e-5)
#
# e_sol = sol.y
#
# v_sol= np.zeros((len(dofs_v), n_ev))
# sig_sol= np.zeros((len(dofs_sig), n_ev))
#
# for i in range(n_ev):
#
#    v_sol[:, i] = e_sol[dofs_v, i]
#
#    sig_sol[:, i] = e_sol[dofs_sig, i]
#
#
# H_vec = np.zeros((n_ev))
#
# for i in range(n_ev):
#    H_vec[i] = 0.5 *(e_sol[:, i] @ M @ e_sol[:, i])
#
# fig = plt.figure()
# plt.plot(t_ev, H_vec, 'g-')
# plt.xlabel(r'{Time} [t]')
# plt.ylabel(r'Total Energy [J]')
#
# # Plot of the different variables
# # Due to fenics ordering, a permutation is first needed
#
# perm_v = np.argsort(x_v)
# x_v_perm = x_v[perm_v]
# v_sol_perm = v_sol[perm_v, :]
#
# perm_sig = np.argsort(x_sig)
# x_sig_perm = x_sig[perm_sig]
# sig_sol_perm = sig_sol[perm_sig, :]
#
# # Plot variables
# fig, ax = plt.subplots()
# ax.set_xlabel('Space [m]')
# ax.set_ylabel('Coenergy variables')
# ax.set_xlim(0, 1)
# ax.set_ylim(np.min(np.min(e_sol)), np.max(np.max(e_sol)))
#
# line_v, = ax.plot([], [], lw=2, label = '${v}$ at $t$ =' \
#                   + '{0:.2f}'.format(t_ev[0]) + '[s]')
# line_sig, = ax.plot([], [], '*', lw=2, label = '${\sigma}$ at $t$ ='  \
#                   + '{0:.2f}'.format(t_ev[0]) + '[s]')
#
# # Functions for plot
# def animate(i):
#    line_v.set_data(np.pad(x_v_perm, (1, 0)), np.pad(v_sol_perm[:,i], (1, 0)))
#    line_sig.set_data(x_sig_perm, sig_sol_perm[:,i])
#
#    line_v.set_label('${v}$')
#    line_sig.set_label('${\sigma}$')
#
# #    line_vt.set_label('$\mathbf{v}_t$ at $t$ =' + '{0:.2f}'.format(t_ev[i]) + '[s]')
# #    line_vr.set_label('$\mathbf{v}_r$ at $t$ =' + '{0:.2f}'.format(t_ev[i]) + '[s]')
# #    line_sigr.set_label('$\bm{\sigma}_r$ at $t$ =' + '{0:.2f}'.format(t_ev[i]) + '[s]')
# #    line_sigt.set_label('$\bm{\sigma}_t$ at $t$ =' + '{0:.2f}'.format(t_ev[i]) + '[s]')
# #
#    ax.legend(bbox_to_anchor=(1.25, 1.25))
#
#    return [line_v, line_sig]
#
#
# anim = animation.FuncAnimation(fig, animate, frames=len(t_ev), interval=20, blit=False)
#
# ##path_out = "/home/andrea/Videos/"
# ##Writer = animation.writers['ffmpeg']
# ##writer = Writer(fps=25, metadata=dict(artist='Me'), bitrate=1800)
# ##anim.save(path_out + 'timo_bc.mp4', writer=writer)
#
# plt.show()
