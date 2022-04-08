import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import LineCollection
from matplotlib import animation
import tutorials.utilities.plot_setup

path_fig = "/home/andrea/PHD_github/LaTeXProjects/CandidatureISAE/imagesEqTr/"
path_video = "/home/andrea/Videos/PresEqTransport/"


def u_0(x):
    # x : array of spatial coordinates
    return np.exp(-x**2/4)


def ex_solution(x_vec, t_vec, c, save_fig=False, plot_fig=False):

    x_mat, t_mat = np.meshgrid(x_vec, t_vec)

    x_plot, t_plot = x_mat.flatten(), t_mat.flatten()

    # Triangulate parameter space to determine the triangles
    tri = mtri.Triangulation(x_plot, t_plot)

    # Solution
    u_ex = u_0(x_plot - c * t_plot)

    # Plot the first surface.
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x_plot, t_plot, u_ex, triangles=tri.triangles,\
                    vmin=0, vmax=1, cmap="winter")

    ax.set_zlim(0, 1)
    ax.set_xlabel(r'Space')
    ax.set_ylabel(r'Time')
    ax.set_title(r'Solution $u(x, t), \; c=$' + str(c))
    ax.view_init(azim=-110)
    ax.plot(c*t_vec, t_vec, np.zeros((len(t_vec), )), '-.r', linewidth=4, label=r'$\gamma$')
    ax.legend()
    if save_fig:
        plt.savefig(path_fig + "u_sol_nobcs.eps", format="eps", bbox_inches='tight')
    if plot_fig:
        plt.show()

    return u_ex


def mesh(x_vec, t_vec, save_fig=False, plot_fig=False):
    x_grid, t_grid = np.meshgrid(x_vec, t_vec)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.scatter(x_grid, t_grid)

    segs1 = np.stack((x_grid, t_grid), axis=2)
    segs2 = segs1.transpose(1, 0, 2)
    plt.gca().add_collection(LineCollection(segs1))
    plt.gca().add_collection(LineCollection(segs2))
    ax.set_xlabel(r'Space')
    ax.set_ylabel(r'Time')

    if save_fig:
        plt.savefig(path_fig + "mesh.eps", format="eps", bbox_inches='tight')
    if plot_fig:
        plt.show()
    else:
        plt.close(fig)

    return x_grid, t_grid


def animate_sol(x_vec, t_vec, u_num, c, sig, save_anim = False):
    fig = plt.figure()



    min_sol = 0
    max_sol = 1

    min_plot_sol = min_sol-np.abs(min_sol)/10
    max_plot_sol = max_sol+np.abs(max_sol)/10

    x_min = min(x_vec)
    x_max = max(x_vec)

    if c > 0:
        loc_str = "upper left"
        cord_ann = (x_min, max_sol)
    else:
        loc_str = "upper right"
        cord_ann = (x_max, max_sol)

    ax = plt.axes(xlim=(x_min, x_max), ylim=(min_plot_sol, max_plot_sol))
    ax.set_xlabel(r'Space')
    ax.set_title(r'$c=' + str(c) + ",\quad \sigma=" + str(sig) + "$")
    # line, = ax.plot([], [], lw=2)

    # initialization function: plot the background of each frame
    def init():
        # line.set_data(x_vec, u_num[:, 0])
        # line.set_data(x_vec, u_0(x_vec))
        ax.plot(x_vec, u_num[:, 0], 'b')
        ax.plot(x_vec, u_0(x_vec), 'r')

        return 0

    # animation function.  This is called sequentially
    def animate(i):
        ax.lines.clear()

        # line.set_data(x_vec, u_num[:, i])
        # line.set_data(x_vec, u_0(x_vec-c*t_vec[i]))
        ax.plot(x_vec, u_num[:, i], 'b', label="numerical")
        ax.plot(x_vec, u_0(x_vec-c*t_vec[i]), 'r', label="exact")
        ax.set_title(r'$c=' + str(c) + ",\quad \sigma=" + str(sig) + ", \quad t=" + '{0:.2}'.format(t_vec[i]) + ".$")

        ax.legend(loc=loc_str, frameon=False)

        return 0

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(t_vec), interval=0, blit=False)

    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    if save_anim:
        name_vid = input("Name video : ")
        anim.save(path_video + name_vid + '.mp4', fps=20, extra_args=['-vcodec', 'libx264'])

    plt.show()

    return 1


def exp_time_aval_space(fun_u0, sigma, x_vec, t_vec):
    n_x = len(x_vec)
    n_t = len(t_vec)
    u_sol = np.zeros((n_x, n_t))

    u_sol[:, 0] = fun_u0(x_vec)

    for n in range(1, n_t):
        u_sol[:-1, n] = -sigma*u_sol[1:, n-1] + (sigma+1) * u_sol[:-1, n-1]

    return u_sol


def exp_time_amont_space(fun_u0, sigma, x_vec, t_vec):
    n_x = len(x_vec)
    n_t = len(t_vec)
    u_sol = np.zeros((n_x, n_t))

    u_sol[:, 0] = fun_u0(x_vec)

    for n in range(1, n_t):
        u_sol[1:, n] = sigma*u_sol[:-1, n-1] + (1-sigma) * u_sol[1:, n-1]

    return u_sol


def exp_time_centre_space(fun_u0, sigma, x_vec, t_vec):
    n_x = len(x_vec)
    n_t = len(t_vec)
    u_sol = np.zeros((n_x, n_t))

    u_sol[:, 0] = fun_u0(x_vec)

    for n in range(1, n_t):
        u_sol[1:-1, n] = u_sol[1:-1, n-1] + sigma/2 * (u_sol[2:, n-1]-u_sol[:-2, n-1])

    return u_sol


def imp_time_amont_space(fun_u0, sigma, x_vec, t_vec):
    n_x = len(x_vec)
    n_t = len(t_vec)
    u_sol = np.zeros((n_x, n_t))

    u_sol[:, 0] = fun_u0(x_vec)

    A = np.zeros((n_x, n_x))

    np.fill_diagonal(A, 1+sigma)

    A[0, 0] = 1

    for ii in range(1, n_x):
        A[ii, ii-1] = -sigma

    invA = np.linalg.inv(A)

    for n in range(1, n_t):
        u_sol[:, n] = invA.dot(u_sol[:, n-1])

    return u_sol
