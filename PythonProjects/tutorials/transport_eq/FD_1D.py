import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import tutorials.utilities.plot_setup

def init_cond(x):
    # x : array of spatial coordinates
    return np.exp(-x**2/4)


x_vec = np.linspace(0, 20, 200)
t_vec = np.linspace(0, 10, 100)
u_0 = init_cond(x_vec)

x_mat, t_mat = np.meshgrid(x_vec, t_vec)
x_plot, t_plot = x_mat.flatten(), t_mat.flatten()

c = 2
u_plot = init_cond(x_plot-c*t_plot)

# Triangulate parameter space to determine the triangles
tri = mtri.Triangulation(x_plot, t_plot)
# Plot the surface.
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(x_plot, t_plot, u_plot, triangles=tri.triangles, cmap=plt.cm.Spectral)
ax.set_zlim(0, 1)
ax.set_xlabel(r'Space')
ax.set_ylabel(r'Time')
ax.set_title(r'Solution $u(x, t)$')
ax.view_init(azim=-100)
ax.plot(c*t_vec, t_vec, 0, '-.', linewidth=4, label=r'$\gamma$')
ax.legend()

path_fig = "/home/andrea/GitProjects/PHD_github/LaTeXProjects/CandidatureISAE/imagesEqTr/"
plt.savefig(path_fig + "u_sol.eps", format="eps", bbox_inches='tight')

plt.show()

