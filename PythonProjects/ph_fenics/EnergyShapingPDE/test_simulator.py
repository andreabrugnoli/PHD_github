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

from EnergyShapingPDE.func_timoshenko import matrices_timoshenko

init = ('exp(-x[0])-1', '0', '0', '0')

M, J, B, e0, dofs_dict, x_dict = matrices_timoshenko(n_el=10, deg=1, e0_string=init)

dofs_vt = dofs_dict['v_t']
x_vt = x_dict['v_t']

dofs_vr = dofs_dict['v_r']
x_vr = x_dict['v_r']

dofs_sigr = dofs_dict['sig_r']
x_sigr = x_dict['sig_r']

dofs_sigt = dofs_dict['sig_t']
x_sigt = x_dict['sig_t']

u_in = np.array([1, 0])

A_sys = np.linalg.solve(M, J)
B_sys = np.linalg.solve(M, B)

def fun(t,y):

    dydt = A_sys @ y + B_sys @ u_in * np.sin(pi*t) * (t<=1 or t>=5) 

    return dydt

t0 = 0.0
t_fin = 10
t_span = [t0, t_fin]

n_ev = 500
t_ev = np.linspace(t0, t_fin, num=n_ev)

sol = solve_ivp(fun, t_span, e0, method='RK45', t_eval = t_ev, \
                       atol = 1e-5, rtol = 1e-5)

e_sol = sol.y

vt_sol= np.zeros((len(dofs_vt), n_ev))
vr_sol= np.zeros((len(dofs_vr), n_ev))

sigr_sol= np.zeros((len(dofs_sigr), n_ev))
sigt_sol= np.zeros((len(dofs_sigt), n_ev))

for i in range(n_ev):
    
    vt_sol[:, i] = e_sol[dofs_vt, i]
    vr_sol[:, i] = e_sol[dofs_vr, i]
    
    sigr_sol[:, i] = e_sol[dofs_sigr, i]
    sigt_sol[:, i] = e_sol[dofs_sigt, i]


H_vec = np.zeros((n_ev))

for i in range(n_ev):
    H_vec[i] = 0.5 *(e_sol[:, i] @ M @ e_sol[:, i])

fig = plt.figure()
plt.plot(t_ev, H_vec, 'g-', label = 'Total Energy [J]')
plt.xlabel(r'{Time} [t]',fontsize=16)
plt.legend(loc='upper left')


# Plot of the different variables
# Due to fenics ordering, a permutation is first needed

perm_vt = np.argsort(x_vt)
x_vt_perm = x_vt[perm_vt]
vt_sol_perm = vt_sol[perm_vt, :]

perm_vr = np.argsort(x_vr)
x_vr_perm = x_vr[perm_vr]
vr_sol_perm = vr_sol[perm_vr, :]

perm_sigr = np.argsort(x_sigr)
x_sigr_perm = x_sigr[perm_sigr]
sigr_sol_perm = sigr_sol[perm_sigr, :]

perm_sigt = np.argsort(x_sigt)
x_sigt_perm = x_sigt[perm_sigt]
sigt_sol_perm = sigt_sol[perm_sigt, :]

## Initial condition plot
#fig = plt.figure(1)
#ax = plt.axes(xlim=(0, 1), ylim=(np.min(np.min(vt_sol)), np.max(np.max(vt_sol))))
#line, = ax.plot(x_vt_perm, vt_sol_perm[:, 0], lw=2, label = 'Time =' + '{0:.2f}'.format(t_ev[0]) + '[s]')



# Vertical velocity
fig_vt = plt.figure()
ax_vt = plt.axes(xlim=(0, 1), ylim=(np.min(np.min(vt_sol)), np.max(np.max(vt_sol))))
ax_vt.set_xlabel('Space [m]')
ax_vt.set_ylabel('Vertical velocity')

line_vt, = ax_vt.plot([], [], lw=2, label = 'Time =' + '{0:.2f}'.format(t_ev[0]) + '[s]')

# Functions for plot
def animate_vt(i):
    line_vt.set_data(x_vt_perm, vt_sol_perm[:,i])
    
    line_vt.set_label('Time =' + '{0:.2f}'.format(t_ev[i]) + '[s]')
    ax_vt.legend(bbox_to_anchor=(1.2, 1.2))
    return line_vt,

anim_vt = animation.FuncAnimation(fig_vt, animate_vt, frames=len(t_ev), interval=20, blit=False)

#path_out = "/home/andrea/Videos/"
#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=25, metadata=dict(artist='Me'), bitrate=1800)
#anim_vt.save(path_out + 'timo_bc.mp4', writer=writer)

## Angular velocity
#fig_vr = plt.figure()
#ax_vr = plt.axes(xlim=(0, 1), ylim=(np.min(np.min(vr_sol)), np.max(np.max(vr_sol))))
#ax_vr.set_xlabel('Space [m]')
#ax_vr.set_ylabel('Angular velocity')
#
#line_vr, = ax_vr.plot([], [], lw=2, label = 'Time =' + '{0:.2f}'.format(t_ev[0]) + '[s]')
#
#def animate_vr(i):
#    line_vr.set_data(x_vr_perm, vr_sol_perm[:,i])
#    
#    line_vr.set_label('Time =' + '{0:.2f}'.format(t_ev[i]) + '[s]')
#    ax_vr.legend(bbox_to_anchor=(1.2, 1.2))
#    return line_vr,
#
#anim_vr = animation.FuncAnimation(fig_vr, animate_vr, \
#                                frames=len(t_ev), interval=20, blit=False)
#
#
#
## Bending stress
#fig_sigr = plt.figure()
#ax_sigr = plt.axes(xlim=(0, 1), ylim=(np.min(np.min(sigr_sol)), np.max(np.max(sigr_sol))))
#ax_sigr.set_xlabel('Space [m]')
#ax_sigr.set_ylabel('Bending stress')
#
#line_sigr, = ax_sigr.plot([], [], lw=2, label = 'Time =' + '{0:.2f}'.format(t_ev[0]) + '[s]')
#
#def animate_sigr(i):
#    line_sigr.set_data(x_sigr_perm, sigr_sol_perm[:,i])
#    
#    line_sigr.set_label('Time =' + '{0:.2f}'.format(t_ev[i]) + '[s]')
#    ax_sigr.legend(bbox_to_anchor=(1.2, 1.2))
#    return line_sigr,
#
#anim_sigr = animation.FuncAnimation(fig_sigr, animate_sigr, \
#                                frames=len(t_ev), interval=20, blit=False)
#
#
## Shear stress
#fig_sigt = plt.figure()
#ax_sigt = plt.axes(xlim=(0, 1), ylim=(np.min(np.min(sigt_sol)), np.max(np.max(sigt_sol))))
#ax_sigt.set_xlabel('Space [m]')
#ax_sigt.set_ylabel('Shear stress')
#
#line_sigt, = ax_sigt.plot([], [], lw=2, label = 'Time =' + '{0:.2f}'.format(t_ev[0]) + '[s]')
#
#
#def animate_sigt(i):
#    line_sigt.set_data(x_sigt_perm, sigt_sol_perm[:,i])
#    
#    line_sigt.set_label('Time =' + '{0:.2f}'.format(t_ev[i]) + '[s]')
#    ax_sigt.legend(bbox_to_anchor=(1.2, 1.2))
#    return line_sigt,
#
# # call the animator.  blit=True means only re-draw the parts that have changed.
#anim_sigt = animation.FuncAnimation(fig_sigt, animate_sigt, \
#                                frames=len(t_ev), interval=20, blit=False)
#
#
#plt.show()
#
#
