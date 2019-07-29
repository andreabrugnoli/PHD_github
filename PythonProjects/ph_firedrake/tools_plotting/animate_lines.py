import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams["legend.loc"] = 'upper right'


def animate_line3d(data, t, title=None):
    tol = 1e-4
    fntsize = 20


    fig = plt.figure()
    ax = p3.Axes3D(fig)

    lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]

    def update_lines(num, dataLines, lines):
        for line, data in zip(lines, dataLines):
            # NOTE: there is no .set_data() for 3 dim data...
            line.set_data(data[0:2, :num])
            line.set_3d_properties(data[2, :num])
            line.set_marker("o")
        return lines



    x_min = np.min(data[:, 0, :])
    x_max = np.max(data[:, 0, :])

    y_min = np.min(data[:, 1, :])
    y_max = np.max(data[:, 1, :])

    z_min = np.min(data[:, 2, :])
    z_max = np.max(data[:, 2, :])
    #
    # ax.set_xlabel(xlabel, fontsize=fntsize)
    # ax.set_ylabel(ylabel, fontsize=fntsize)
    # ax.set_zlabel(zlabel, fontsize=fntsize)

    ax.set_xlim3d(x_min, x_max)
    ax.set_ylim3d(y_min, y_max)
    ax.set_zlim3d(z_min, z_max)
    ax.w_zaxis.set_major_locator(LinearLocator(10))
    ax.w_zaxis.set_major_formatter(FormatStrFormatter('%1.2g'))
    ax.set_title(title, fontsize=fntsize, loc='left')
    ax.set_xlabel('$x [m]$', fontsize=fntsize)
    ax.set_ylabel('$y [m]$', fontsize=fntsize)
    ax.set_zlabel('$z [m]$', fontsize=fntsize)

    lab = 'Time =' + '{0:.2e}'.format(t[0])

    anim = animation.FuncAnimation(fig, update_lines, len(t), fargs=(data, lines),
                                   interval=10)

    return anim


