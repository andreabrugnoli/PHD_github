import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

from firedrake.plot import _two_dimension_triangle_func_val


matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams["legend.loc"] = 'upper right'

def animateInt2D(minSol, maxSol, solPl_list, x2, y2, solRod, t,\
              xlabel = None, ylabel = None,  zlabel = None, z2label = None, title = None):
    tol = 1e-4
    fntsize = 20

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update_plot(frame_number, solPl_list, solRod):
        ax.collections.clear()
        ax.lines.pop(0)
        lab = 'Time =' + '{0:.2e}'.format(t[frame_number])
        triangulation, Z = _two_dimension_triangle_func_val(solPl_list[frame_number], 10)
        plot_pl = ax.plot_trisurf( \
            triangulation, Z, cmap=cm.jet, label=lab, linewidth=0, antialiased=False)

        ax.plot(x2, y2, solRod[:, frame_number],\
                            linewidth=5, label = z2label, color='black')

        plot_pl._facecolors2d = plot_pl._facecolors3d
        plot_pl._edgecolors2d = plot_pl._edgecolors3d

        ax.legend()


    if minSol == maxSol:
        raise ValueError('Constant function for drawnow')

    ax.set_xlabel(xlabel, fontsize=fntsize)
    ax.set_ylabel(ylabel, fontsize=fntsize)
    ax.set_zlabel(zlabel, fontsize=fntsize)

    ax.set_zlim(minSol - 1e-3 * abs(minSol), maxSol + 1e-3 * abs(maxSol))
    ax.w_zaxis.set_major_locator(LinearLocator(10))
    ax.w_zaxis.set_major_formatter(FormatStrFormatter('%1.2g'))

    ax.set_title(title, fontsize=fntsize, loc ='left')

    lab = 'Time =' + '{0:.2e}'.format(t[0])
    triangulation, Z = _two_dimension_triangle_func_val(solPl_list[0], 10)
    ax.plot_trisurf( triangulation, Z, cmap=cm.jet, label=lab, linewidth=0, antialiased=False)
    ax.plot(x2, y2, solRod[:, 0], label=z2label, color='black')

    anim =  animation.FuncAnimation(fig, update_plot, frames=len(t),\
                                    interval=10, fargs=(solPl_list, solRod))




    return anim

