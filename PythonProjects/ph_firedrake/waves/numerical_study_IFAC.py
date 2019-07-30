import numpy as np
import matplotlib.pyplot as plt
from waves.wavedae_func import computeH_dae
from waves.waveode_func import computeH_ode
from math import pi
plt.rc('text', usetex=True)

n_mesh = 10
erral_vec = np.zeros(n_mesh)
errH_vec = np.zeros(n_mesh)

h_vec = np.zeros(n_mesh)

for i in range(n_mesh):

    mesh_ind = i + 6
    H_dae, t_dae = computeH_dae(mesh_ind)
    H_ode, t_ode = computeH_ode(mesh_ind)

    assert (t_dae-t_ode).all() == 0
    dt = np.diff(t_dae)[0]

    norm_al_dae = np.sqrt(2*dt*(np.sum(H_dae) - 0.5 * (H_dae[0] + H_dae[-1])))
    norm_al_ode = np.sqrt(2*dt*(np.sum(H_ode) - 0.5 * (H_ode[0] + H_ode[-1])))

    erral_vec[i] = (norm_al_dae - norm_al_ode)/(norm_al_dae + norm_al_ode)

    diffH = np.abs(H_dae - H_ode)
    norm_diffH = np.sqrt(np.sum(diffH) - 0.5 * (diffH[0] + diffH[-1]))

    errH_vec[i] = norm_diffH

    h_vec[i] = pi/mesh_ind


np.save("al_err.npy", erral_vec)
np.save("H_err.npy", errH_vec)

fntsize = 16
fig = plt.figure()
plt.plot(h_vec, erral_vec, 'b-')
plt.xlabel(r'Mesh size', fontsize=fntsize)
plt.ylabel(r'$\frac{||\alpha_1||_{L^2} - ||\alpha_2||_{L^2}}{||\alpha_1||_{L^2} + ||\alpha_2||_{L^2}}$', fontsize=fntsize)
plt.title(r"Relative difference energy variable",
          fontsize=fntsize)

path_figs = "./"
plt.savefig(path_figs + "alpha_diff.eps", format="eps")

fig = plt.figure()
plt.plot(h_vec, errH_vec, 'b-')
plt.xlabel(r'Mesh size', fontsize=fntsize)
plt.ylabel(r'$||H_1 - H_2||_{L^1}$', fontsize=fntsize)
plt.title(r"$L^1$ norm Hamiltonian difference", fontsize=fntsize)
# plt.legend(loc='upper left')

path_figs = "./"
plt.savefig(path_figs + "H_diff.eps", format="eps")

plt.show()