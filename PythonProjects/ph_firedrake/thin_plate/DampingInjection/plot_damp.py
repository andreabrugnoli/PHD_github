from firedrake import *
import numpy as np

np.set_printoptions(threshold=np.inf)

from scipy import linalg as la
import sys
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from tools_plotting.animate_surf import animate2D
import matplotlib.animation as animation
from matplotlib import cm
from firedrake.plot import _two_dimension_triangle_func_val

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

from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, csc_matrix

mesh = UnitSquareMesh(6, 6, quadrilateral=False)

# plot(mesh)
# plt.show()
name_FEp = 'Argyris'
name_FEq = 'DG'

Vp = FunctionSpace(mesh, name_FEp, 5)
Vq = VectorFunctionSpace(mesh, name_FEq, 3, dim=3)

V = Vp * Vq
n_Vp = Vp.dim()
n_Vq = Vq.dim()

path_res = "/home/a.brugnoli/LargeFiles/results_DampKirchh2/"

t_ev = np.load(path_res + "t_ev5s.npy")
ep_sol = np.load(path_res + "ep_sol5s.npy")
eq_sol = np.load(path_res + "eq_sol5s.npy")

n_ev = len(t_ev)

n_sol = len(t_ev)


v_CGvec = []
v_fun = Function(Vp)
Vp_CG = FunctionSpace(mesh, 'Lagrange', 3)
n_VpCG = Vp_CG.dim()
print(n_Vp, n_VpCG)

maxZvec = np.zeros(n_ev)
minZvec = np.zeros(n_ev)
for i in range(n_ev):
    v_fun.vector()[:] = ep_sol[:, i]
    v_CG = project(v_fun, Vp_CG)
    v_CGvec.append(v_CG)

    maxZvec[i] = max(v_CG.vector())
    minZvec[i] = min(v_CG.vector())

maxZ = max(maxZvec)
minZ = min(minZvec)

print(maxZ)
print(minZ)

if matplotlib.is_interactive():
    plt.ioff()
plt.close('all')

path_out = "/home/a.brugnoli/Plots/Python/Plots/Kirchhoff_plots/Simulations/Article_CDC/DampingInjection3_vel/"


# anim = animate2D(minZ, maxZ, v_CGvec, t_ev, xlabel='$x \;  \mathrm{[m]}$', ylabel='$y \;  \mathrm{[m]}$', \
#                  zlabel='$e_w \;  \mathrm{[m/s]}$', title='Vertical velocity')
#
# rallenty = 10
# Writer = animation.writers['ffmpeg']
# # writer = Writer(fps=int(n_ev/(t_f*rallenty)), metadata=dict(artist='Me'), bitrate=1800)
# writer = Writer(fps=25, metadata=dict(artist='Me'), bitrate=1800)
#
# anim.save(path_out + 'Kirchh_Damped.mp4', writer=writer)
# plt.show()

plot_solutions = True
if plot_solutions:

    n_fig = 100
    tol = 1e-6

    for i in range(n_fig):
        index = int(n_ev / n_fig * (i + 1) - 1)

        triangulation, Z = _two_dimension_triangle_func_val(v_CGvec[index], 10)
        fig = plt.figure()

        ax = fig.add_subplot(111, projection="3d")
        ax.collections.clear()

        surf_opts = {'cmap': cm.jet, 'linewidth': 0, 'antialiased': False}  # , 'vmin': minZ, 'vmax': maxZ}
        lab = 'Time =' + '{0:.2f}'.format(t_ev[index])
        surf = ax.plot_trisurf(triangulation, Z, **surf_opts)
        # fig.colorbar(surf)

        ax.set_xbound(-tol, 1 + tol)
        ax.set_xlabel('$x \;  \mathrm{[m]}$')

        ax.set_ybound(-tol, 1 + tol)
        ax.set_ylabel('$y \;  \mathrm{[m]}$')

        ax.w_zaxis.set_major_locator(LinearLocator(10))
        ax.w_zaxis.set_major_formatter(FormatStrFormatter('%1.2f'))

        ax.set_zlabel('$w \;  \mathrm{[mm]}$')
        ax.set_title('Vertical velocity ' + '$(t=$' + '{0:.2f}'.format(t_ev[index]) + '$\mathrm{[s]})$')
        # ax.set_title('Vertical displacement ' +'$(t=$' + str(t_ev[index]) + '$\mathrm{[s]})$')

        ax.set_zlim3d(minZ - 0.01 * abs(minZ), maxZ + 0.01 * abs(maxZ))

        plt.savefig(path_out + "Snapshot_t" + str(index + 1) + ".eps", format="eps")



