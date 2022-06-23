import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import LineCollection

import utilities.plot_setup

path_fig = "/home/andrea/PHD_github/LaTeXProjects/CandidatureISAE/imagesEqTr/"

def g(x):
    # x : array of spatial coordinates
    return np.exp(-x**2/4)

def f1(t):
    # t : array of temporal coordinates
    return np.ones(t.shape)

def f2(t):
    # t : array of temporal coordinates
    return np.exp(-t**2)

def f3(t):
    # t : array of temporal coordinates
    return np.cos(t)**2

L = 20
T = 10

x_vec = np.linspace(0, L, 200)
t_vec = np.linspace(0, T, 100)
c = 2

u_0 = g(x_vec)

x_mat, t_mat = np.meshgrid(x_vec, t_vec)

x_plot, t_plot = x_mat.flatten(), t_mat.flatten()
index_below_ct = x_plot >= c*t_plot
index_above_ct = x_plot <= c*t_plot

x_below_ct = x_plot[index_below_ct]
t_below_ct = t_plot[index_below_ct]

x_above_ct = x_plot[index_above_ct]
t_above_ct = t_plot[index_above_ct]

# Triangulate parameter space to determine the triangles
tri_below = mtri.Triangulation(x_below_ct, t_below_ct)
tri_above = mtri.Triangulation(x_above_ct, t_above_ct)

# Solution
u_below_ct = g(x_below_ct-c*t_below_ct)

u1_above_ct = f1(t_above_ct-x_above_ct/c)
u2_above_ct = f2(t_above_ct-x_above_ct/c)
u3_above_ct = f3(t_above_ct-x_above_ct/c)

plot_surface = False

if plot_surface:

    # Plot the first surface.
    resultant1 = np.concatenate([u_below_ct, u1_above_ct])
    min_val1, max_val1 = np.amin(resultant1), np.amax(resultant1)
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x_below_ct, t_below_ct, u_below_ct, triangles=tri_below.triangles,\
                    vmin=min_val1, vmax=max_val1, cmap="winter")
    ax.plot_trisurf(x_above_ct, t_above_ct, u1_above_ct, triangles=tri_above.triangles,\
                    vmin=min_val1, vmax=max_val1, cmap="winter")

    ax.set_zlim(min_val1, max_val1)
    ax.set_xlabel(r'Space')
    ax.set_ylabel(r'Time')
    ax.set_title(r'Solution $u(x, t)$')
    ax.view_init(azim=-120)
    ax.plot(c*t_vec, t_vec, np.zeros((len(t_vec), )), '-.r', linewidth=4, label=r'$\gamma$')
    ax.plot(np.zeros((len(t_vec), )), t_vec, f1(t_vec), '-.k', linewidth=4, label=r'$f(t)$')
    ax.legend()

    plt.savefig(path_fig + "u_sol_f1.eps", format="eps", bbox_inches='tight')

    # Plot the second surface.
    resultant2 = np.concatenate([u_below_ct, u2_above_ct])
    min_val2, max_val2 = np.amin(resultant2), np.amax(resultant2)
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x_below_ct, t_below_ct, u_below_ct, triangles=tri_below.triangles,\
                    vmin=min_val2, vmax=max_val2, cmap="winter")
    ax.plot_trisurf(x_above_ct, t_above_ct, u2_above_ct, triangles=tri_above.triangles,\
                    vmin=min_val2, vmax=max_val2, cmap="winter")

    ax.set_zlim(min_val2, max_val2)
    ax.set_xlabel(r'Space')
    ax.set_ylabel(r'Time')
    ax.set_title(r'Solution $u(x, t)$')
    ax.view_init(azim=-120)
    ax.plot(c*t_vec, t_vec, np.zeros((len(t_vec), )), '-.r', linewidth=4, label=r'$\gamma$')
    ax.plot(np.zeros((len(t_vec), )), t_vec, f2(t_vec), '-.k', linewidth=4, label=r'$f(t)$')

    ax.legend()

    plt.savefig(path_fig + "u_sol_f2.eps", format="eps", bbox_inches='tight')

    # Plot the third surface.
    resultant3 = np.concatenate([u_below_ct, u3_above_ct])
    min_val3, max_val3 = np.amin(resultant3), np.amax(resultant3)
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x_below_ct, t_below_ct, u_below_ct, triangles=tri_below.triangles,\
                    vmin=min_val3, vmax=max_val3, cmap="winter")
    ax.plot_trisurf(x_above_ct, t_above_ct, u3_above_ct, triangles=tri_above.triangles,\
                    vmin=min_val3, vmax=max_val3, cmap="winter")

    ax.set_zlim(min_val3, max_val3)
    ax.set_xlabel(r'Space')
    ax.set_ylabel(r'Time')
    ax.set_title(r'Solution $u(x, t)$')
    ax.view_init(azim=-120)
    ax.plot(c*t_vec, t_vec, np.zeros((len(t_vec), )), '-.r', linewidth=4, label=r'$\gamma$')
    ax.plot(np.zeros((len(t_vec), )), t_vec, f3(t_vec), '-.k', linewidth=4, label=r'$f(t)$')
    ax.legend()

    plt.savefig(path_fig + "u_sol_f3.eps", format="eps", bbox_inches='tight')

    plt.show()
