import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams["legend.loc"] = 'upper right'
import numpy as np


def animate_plot(t_vec, x_vec, y_vec, xlabel=None, ylabel=None, title=None):

    fntsize = 20

    fig = plt.figure()
    ax = fig.add_subplot(111)

    def update_plot(frame_number):
        ax.collections.clear()
        ax.lines.pop(0)
        lab = 'Time =' + '{0:.2e}'.format(t_vec[frame_number])
        ax.plot(x_vec[:, frame_number], y_vec[:, frame_number], label=lab, color='black')

        ax.legend()

    ax.set_xlabel(xlabel, fontsize=fntsize)
    ax.set_ylabel(ylabel, fontsize=fntsize)

    ax.set_title(title, fontsize=fntsize, loc ='left')

    lab = 'Time =' + '{0:.2e}'.format(t_vec[0])
    ax.plot(x_vec[:, 0], y_vec[:, 0], label=lab, color='black')

    anim = animation.FuncAnimation(fig, update_plot, frames=len(t_vec), interval=5)

    return anim

