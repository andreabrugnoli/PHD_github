from scipy.integrate import solve_ivp
import numpy as np
from math import pi

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import animation

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

from EnergyShapingPDE.func_timoshenko_all import matrices_timoshenko

init = ('exp(-x[0])-1', '0', '0', '0')

M_red, J_red, B_red, e0_red, dofs_dict, x_dict, T_r2t = matrices_timoshenko(n_el=10, deg=1,\
                                                                    e0_string=init)

dofs_vt = dofs_dict['v_t']
x_vt = x_dict['v_t']

dofs_vr = dofs_dict['v_r']
x_vr = x_dict['v_r']

dofs_sigr = dofs_dict['sig_r']
x_sigr = x_dict['sig_r']

dofs_sigt = dofs_dict['sig_t']
x_sigt = x_dict['sig_t']

u_in = np.array([1, 0])

A_sys = np.linalg.solve(M_red, J_red)
B_sys = np.linalg.solve(M_red, B_red)

def fun(t,y):

    dydt = A_sys @ y + B_sys @ u_in * np.sin(pi*t) * (t<=1 or t>=5) 

    return dydt

t0 = 0.0
t_fin = 10
t_span = [t0, t_fin]

n_ev = 500
t_ev = np.linspace(t0, t_fin, num=n_ev)

sol_red = solve_ivp(fun, t_span, e0_red, method='RK45', t_eval = t_ev, \
                       atol = 1e-5, rtol = 1e-5)

e_red = sol_red.y

n_all = len(T_r2t)
e_all = np.zeros((n_all, n_ev))

vt_all= np.zeros((len(dofs_vt), n_ev))
vr_all= np.zeros((len(dofs_vr), n_ev))

sigr_all= np.zeros((len(dofs_sigr), n_ev))
sigt_all= np.zeros((len(dofs_sigt), n_ev))

for i in range(n_ev):
    
    e_all[:, i] = T_r2t @ e_red[:,i]
    
    vt_all[:, i] = e_all[dofs_vt, i]
    vr_all[:, i] = e_all[dofs_vr, i]
    
    sigr_all[:, i] = e_all[dofs_sigr, i]
    sigt_all[:, i] = e_all[dofs_sigt, i]


H_vec = np.zeros((n_ev))

for i in range(n_ev):
    H_vec[i] = 0.5 *(e_red[:, i] @ M_red @ e_red[:, i])

fig = plt.figure()
plt.plot(t_ev, H_vec, 'g-')
plt.xlabel(r'{Time} [t]')
plt.ylabel(r'Total Energy [J]')


# Plot of the different variables
# Due to fenics ordering, a permutation is first needed

perm_vt = np.argsort(x_vt)
x_vt_perm = x_vt[perm_vt]
vt_all_perm = vt_all[perm_vt, :]

perm_vr = np.argsort(x_vr)
x_vr_perm = x_vr[perm_vr]
vr_all_perm = vr_all[perm_vr, :]

perm_sigr = np.argsort(x_sigr)
x_sigr_perm = x_sigr[perm_sigr]
sigr_all_perm = sigr_all[perm_sigr, :]

perm_sigt = np.argsort(x_sigt)
x_sigt_perm = x_sigt[perm_sigt]
sigt_all_perm = sigt_all[perm_sigt, :]

## Initial condition plot
#fig = plt.figure(1)
#ax = plt.axes(xlim=(0, 1), ylim=(np.min(np.min(v_all)), np.max(np.max(v_all))))
#line, = ax.plot(x_Vpw_perm, v_all_perm[:, 0], lw=2, label = 'Time =' + '{0:.2f}'.format(t_ev[0]) + '[s]')

# Plot variables
fig, ax = plt.subplots()
ax.set_xlabel('Space [m]')
ax.set_ylabel('Coenergy variables')
ax.set_xlim(0, 1)
ax.set_ylim(np.min(np.min(e_all)), np.max(np.max(e_all)))

line_vt, = ax.plot([], [], lw=2, label = '$\mathbf{v}_t$ at $t$ =' \
                   + '{0:.2f}'.format(t_ev[0]) + '[s]')
line_vr, = ax.plot([], [], lw=2, label = '$\mathbf{v}_r$ at $t$ =' \
                   + '{0:.2f}'.format(t_ev[0]) + '[s]')
line_sigr, = ax.plot([], [], lw=2, label = '$\mathbf{s}_r$ at $t$ ='  \
                   + '{0:.2f}'.format(t_ev[0]) + '[s]')
line_sigt, = ax.plot([], [], lw=2, label = '$\mathbf{s}_t$ at $t$ ='  \
                   + '{0:.2f}'.format(t_ev[0]) + '[s]')

# Functions for plot
def animate(i):
    line_vt.set_data(np.pad(x_vt_perm, (1, 0)), np.pad(vt_all_perm[:,i], (1, 0)))
    line_vr.set_data(np.pad(x_vr_perm, (1, 0)), np.pad(vr_all_perm[:,i], (1, 0)))
    line_sigr.set_data(x_sigr_perm, sigr_all_perm[:,i])
    line_sigt.set_data(x_sigt_perm, sigt_all_perm[:,i])
    
    line_vt.set_label('$\mathbf{v}_t$')
    line_vr.set_label('$\mathbf{v}_r$')
    line_sigr.set_label('$\mathbf{s}_r$')
    line_sigt.set_label('$\mathbf{s}_t$')
    
#    line_vt.set_label('$\mathbf{v}_t$ at $t$ =' + '{0:.2f}'.format(t_ev[i]) + '[s]')
#    line_vr.set_label('$\mathbf{v}_r$ at $t$ =' + '{0:.2f}'.format(t_ev[i]) + '[s]')
#    line_sigr.set_label('$\bm{\sigma}_r$ at $t$ =' + '{0:.2f}'.format(t_ev[i]) + '[s]')
#    line_sigt.set_label('$\bm{\sigma}_t$ at $t$ =' + '{0:.2f}'.format(t_ev[i]) + '[s]')
#    
    ax.legend(bbox_to_anchor=(1.25, 1.25))

    return [line_vt, line_vr, line_sigr, line_sigt]


anim = animation.FuncAnimation(fig, animate, frames=len(t_ev), interval=20, blit=False)

#path_out = "/home/andrea/Videos/"
#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=25, metadata=dict(artist='Me'), bitrate=1800)
#anim_vt.save(path_out + 'timo_all.mp4', writer=writer)
