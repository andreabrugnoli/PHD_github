import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

from firedrake.plot import _two_dimension_triangle_func_val

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams["legend.loc"] = 'upper right'

def animate2D(minSol, maxSol, solFun_list, t, xlabel = None, ylabel = None,  zlabel = None, title = None):
    tol = 1e-4
    fntsize = 20

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update_plot(frame_number, solFun_list, plot):
        ax.collections.clear()

        lab = 'Time =' + '{0:.2e}'.format(t[frame_number])
        triangulation, Z = _two_dimension_triangle_func_val(solFun_list[frame_number], 10)
        plot = ax.plot_trisurf(triangulation, Z, label=lab, **surf_opts)
        plot._facecolors2d = plot._facecolors3d
        plot._edgecolors2d = plot._edgecolors3d

        ax.legend()


    if minSol == maxSol:
        raise ValueError('Constant function for drawnow')

    ax.set_xlabel(xlabel, fontsize=fntsize)
    ax.set_ylabel(ylabel, fontsize=fntsize)
    ax.set_zlabel(zlabel, fontsize=fntsize)

    ax.set_zlim(minSol - 1e-3 * abs(minSol), maxSol + 1e-3 * abs(maxSol))
    ax.w_zaxis.set_major_locator(LinearLocator(10))
    ax.w_zaxis.set_major_formatter(FormatStrFormatter('%1.2g'))

    ax.set_title(title, fontsize=fntsize, loc='left')

    lab = 'Time =' + '{0:.2e}'.format(t[0])
    triangulation, Z = _two_dimension_triangle_func_val(solFun_list[0], 10)

    surf_opts = {'cmap': cm.jet, 'linewidth': 0, 'antialiased': False, 'vmin':minSol, 'vmax':maxSol}
    plot = ax.plot_trisurf( triangulation, Z, label=lab, **surf_opts)
    fig.colorbar(plot)

    anim =  animation.FuncAnimation(fig, update_plot, frames=len(t), interval = 10, fargs=(solFun_list, plot))

    return anim


