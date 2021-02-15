from scipy.integrate import solve_ivp
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from math import pi
from tools_plotting import setup


matplotlib.rcParams['text.usetex'] = True


from EnergyShapingPDE.matrices_timoshenko import matrices_constraints, \
matrices_timoshenko

init = ('exp(-x[0])-1', '0', '0', '0')
M_all, J_all, B_all, G, e0, dofs_dict, x_dict = matrices_timoshenko(n_el=20, deg=1,\
                                                                    e0_string=init)

dofs_Vpw = dofs_dict['Vpw']
x_Vpw = x_dict['Vpw']


M_red, J_red, B_red, T = matrices_constraints(M_all, J_all, B_all, G)

u_in = np.array([1, 0])

A_sys = np.linalg.solve(M_red, J_red)
B_sys = np.linalg.solve(M_red, B_red)

def fun(t,y):

    dydt = A_sys @ y + B_sys @ u_in * np.sin(pi*t) * (t<=1 or t>=5) 

    return dydt


n_red = len(M_red)

y0 = T.T @ e0

t0 = 0.0
t_fin = 10
n_t = 500
t_span = [t0, t_fin]

t_ev = np.linspace(t0,t_fin, num = n_t)

sol = solve_ivp(fun, t_span, y0, method='BDF', vectorized=False, t_eval = t_ev, \
                       atol = 1e-5, rtol = 1e-5)


e_red = sol.y

n_ev = len(t_ev)

n_all = len(T)

e_all = np.zeros((n_all, n_ev))

v_all= np.zeros((len(dofs_Vpw), n_ev))

for i in range(n_ev):
    
    e_all[:, i] = T @ e_red[:,i]
    v_all[:, i] = e_all[dofs_Vpw, i]

if matplotlib.is_interactive():
    plt.ioff()
plt.close('all')

H_vec = np.zeros((n_ev))

for i in range(n_ev):
    H_vec[i] = 0.5 *(e_red[:, i] @ M_red @ e_red[:, i])

fig0 = plt.figure(0)
plt.plot(t_ev, H_vec, 'g-', label = 'Total Energy (J)')
plt.xlabel(r'{Time} (s)',fontsize=16)
plt.legend(loc='upper left')

perm = np.argsort(x_Vpw)

x_Vpw_perm = x_Vpw[perm]
v_all_perm = v_all[perm, :]


fig = plt.figure(1)
ax = plt.axes(xlim=(0, 1), ylim=(np.min(np.min(v_all)), np.max(np.max(v_all))))
line, = ax.plot(x_Vpw_perm, v_all_perm[:, 0], lw=2, label = 'Time =' + '{0:.2f}'.format(t_ev[0]) + '[s]')


# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure(2)
ax = plt.axes(xlim=(0, 1), ylim=(np.min(np.min(v_all)), np.max(np.max(v_all))))
line, = ax.plot([], [], lw=2, label = 'Time =' + '{0:.2f}'.format(t_ev[0]) + '[s]')


# initialization function: plot the background of each frame
def init():
    line.set_data(x_Vpw_perm, v_all_perm[:, 0])
    
#    line.set_label('Time =' + '{0:.2f}'.format(t_ev[0]) + '[s]')
#    plt.legend(bbox_to_anchor=(1.1, 1.1))
    return line,

# animation function.  This is called sequentially
def animate(i):
    line.set_data(x_Vpw_perm, v_all_perm[:,i])
    
#    line.set_label('Time =' + '{0:.2f}'.format(t_ev[i]) + '[s]')
#    plt.legend(bbox_to_anchor=(1.1, 1.1))
    return line,

 # call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=len(t_ev), interval=20, blit=False)

plt.xlabel('x')
plt.ylabel('Vertical velocity')

plt.show()
