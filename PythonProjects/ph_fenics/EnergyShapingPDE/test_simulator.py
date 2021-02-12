from scipy.integrate import solve_ivp
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation


matplotlib.rcParams['text.usetex'] = True


from EnergyShapingPDE.matrices_timoshenko import matrices_constraints, \
matrices_timoshenko

M_all, J_all, B_all, G, dofs_Vpw, x_dofVpw = matrices_timoshenko(n_el=100)

M_red, J_red, B_red, T = matrices_constraints(M_all, J_all, B_all, G)

u_in = np.array([1, 0])

A_sys = np.linalg.solve(M_red, J_red)
B_sys = np.linalg.solve(M_red, B_red)

def fun(t,y):

    dydt = A_sys @ y + B_sys @ u_in * np.sin(t) * (t<3) # or t>7)

    return dydt


n_red = len(M_red)

y0 = np.zeros((n_red, ))

t0 = 0.0
t_fin = 10
n_t = 1000
t_span = [t0, t_fin]

t_ev = np.linspace(t0,t_fin, num = n_t)

sol = solve_ivp(fun, t_span, y0, method='RK45', vectorized=False, t_eval = t_ev, \
                       atol = 1e-5, rtol = 1e-5)


e_red = sol.y
t_ev = sol.t
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

plt.show()

perm = np.argsort(x_dofVpw)
x_dofVpw.sort()

v_all_perm = v_all[perm, :]

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure(1)
ax = plt.axes(xlim=(0, 1), ylim=(np.min(np.min(v_all)), np.max(np.max(v_all))))
line, = ax.plot([], [], lw=2)

 # initialization function: plot the background of each frame
def init():
    line.set_data(x_dofVpw, v_all_perm[:, 0])
    return line,

 # animation function.  This is called sequentially
def animate(i):
    line.set_data(x_dofVpw, v_all_perm[:,i])
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
