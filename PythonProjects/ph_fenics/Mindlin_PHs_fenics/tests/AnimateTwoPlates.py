import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams["legend.loc"] = 'upper right'

def animate2D(x, y, sol1, sol2, t, xlabel = None, ylabel = None,  zlabel = None, title = None):
    tol = 1e-4
    fntsize = 20

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update_plot(frame_number, sol1, sol2, plot):
        ax.collections.clear()
        lab = 'Time =' + '{0:.2e}'.format(t[frame_number])
        plot = ax.plot_trisurf(x, y, sol1[:, frame_number], label=lab, **surf_opts1)
        plot._facecolors2d = plot._facecolors3d
        plot._edgecolors2d = plot._edgecolors3d

        ax.plot_trisurf(x + x.max(), y, sol2[:, frame_number], **surf_opts2)
        ax.legend()

    ax.set_xlim(min(x) - tol, max(2 * x) + tol)
    ax.set_xlabel(xlabel, fontsize=fntsize)

    ax.set_ylim(min(x) - tol, max(y) + tol)
    ax.set_ylabel(ylabel, fontsize=fntsize)

    ax.set_zlabel(zlabel, fontsize=fntsize)

    minSol = sol1.min()
    maxSol = sol1.max()

    if minSol == maxSol:
        raise ValueError('Constant function for drawnow')

    ax.set_zlim(minSol - 1e-3 * abs(minSol), maxSol + 1e-3 * abs(maxSol))
    ax.w_zaxis.set_major_locator(LinearLocator(10))
    ax.w_zaxis.set_major_formatter(FormatStrFormatter('%1.2g'))

    ax.set_title(title, fontsize=fntsize, loc='left')

    lab = 'Time =' + '{0:.2e}'.format(t[0])

    surf_opts1 = {'cmap': cm.winter, 'linewidth': 0, 'antialiased': False, 'vmin': minSol, 'vmax': maxSol}
    plot = ax.plot_trisurf(x, y, sol1[:, 0], label = lab, **surf_opts1)
    surf_opts2 = {'cmap': cm.inferno, 'linewidth': 0, 'antialiased': False, 'vmin': minSol, 'vmax': maxSol}

    ax.plot_trisurf(x + x.max(), y, sol2[:, 0], **surf_opts2)

    anim = animation.FuncAnimation(fig, update_plot, frames=len(t),\
                                    interval = 10, fargs=(sol1, sol2, plot))
    return anim
