import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import LineCollection

import tutorials.utilities.plot_setup

path_fig = "/home/andrea/PHD_github/LaTeXProjects/CandidatureISAE/imagesEqTr/"

def g(x):
    # x : array of spatial coordinates
    return np.exp(-x**2/4)

L = 20
T = 10

x_vec = np.linspace(-10, L, 300)
t_vec = np.linspace(0, T, 100)
c = 2

u_0 = g(x_vec)

x_mat, t_mat = np.meshgrid(x_vec, t_vec)

x_plot, t_plot = x_mat.flatten(), t_mat.flatten()

# Triangulate parameter space to determine the triangles
tri = mtri.Triangulation(x_plot, t_plot)

# Solution
u_plot = g(x_plot-c*t_plot)
plot_surface = True

if plot_surface:

    # Plot the first surface.
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x_plot, t_plot, u_plot, triangles=tri.triangles,\
                    vmin=0, vmax=1, cmap="winter")

    ax.set_zlim(0, 1)
    ax.set_xlabel(r'Space')
    ax.set_ylabel(r'Time')
    ax.set_title(r'Solution $u(x, t)$')
    ax.view_init(azim=-100)
    ax.plot(c*t_vec, t_vec, np.zeros((len(t_vec), )), '-.r', linewidth=4, label=r'$\gamma$')
    ax.legend()

    plt.savefig(path_fig + "u_sol_nobcs.eps", format="eps", bbox_inches='tight')

    plt.show()


Dx = float(input("Enter Delta x : "))
n_x = int(np.floor(L/Dx) + 1)
x_mesh = np.linspace(0, L, n_x)

Dt = float(input("Enter Delta t : "))
n_t = int(np.floor(T/Dt) + 1)
t_mesh = np.linspace(0, T, n_x)


x_grid, t_grid = np.meshgrid(x_mesh, t_mesh)

fig = plt.figure()
ax = fig.add_subplot(111)

plt.scatter(x_grid, t_grid)

segs1 = np.stack((x_grid,t_grid), axis=2)
segs2 = segs1.transpose(1,0,2)
plt.gca().add_collection(LineCollection(segs1))
plt.gca().add_collection(LineCollection(segs2))
ax.set_xlabel(r'Space')
ax.set_ylabel(r'Time')

plt.savefig(path_fig + "mesh.eps", format="eps", bbox_inches='tight')

plt.show()



