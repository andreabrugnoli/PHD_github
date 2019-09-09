import numpy as np
import matplotlib.pyplot as plt
from waves.wavedae_wc_func import computeH_dae
from waves.waveode_wc_func import computeH_ode

plt.rc('text', usetex=True)

R_ext = 1
ref_file = 'H_dae_5.npy'
H_ref = np.load('/home/a.brugnoli/GitProjects/PythonProjects/ph_firedrake/waves/results_ifacwc/' + ref_file)

n_mesh = 2

errHdae = np.zeros(n_mesh)
errHode = np.zeros(n_mesh)

h_vec = np.zeros(n_mesh)

for i in range(n_mesh):

    mesh_ind = 2 + i
    H_dae, t_dae = computeH_dae(mesh_ind)
    H_ode, t_ode = computeH_ode(mesh_ind)

    assert (t_dae-t_ode).all() == 0
    # dt = np.diff(t_dae)[0]

    # norm_al_dae = np.sqrt(2*dt*(np.sum(H_dae) - 0.5 * (H_dae[0] + H_dae[-1])))
    # norm_al_ode = np.sqrt(2*dt*(np.sum(H_ode) - 0.5 * (H_ode[0] + H_ode[-1])))

    # erral_vec[i] = (norm_al_dae - norm_al_ode)/(norm_al_dae + norm_al_ode)

    # diffH = np.abs(H_dae - H_ode)
    # norm_diffH = np.sqrt(np.sum(diffH) - 0.5 * (diffH[0] + diffH[-1]))

    # errH_vec[i] = norm_diffH

    errHdae[i] = np.sqrt(np.linalg.norm(H_ref - H_dae))
    errHode[i] = np.sqrt(np.linalg.norm(H_ref - H_ode))

    h_vec[i] = R_ext/mesh_ind


np.save("Hdae_err.npy", errHdae)
np.save("Hode_err.npy", errHode)

fntsize = 16

fig = plt.figure()
plt.plot(h_vec, errHdae, 'b-')
plt.xlabel(r'Mesh size', fontsize=fntsize)
plt.ylabel(r'$||H_{ref} - H_{dae}||_{L^2}$', fontsize=fntsize)
plt.title(r"$L^2$ norm Hamiltonian difference", fontsize=fntsize)
# plt.legend(loc='upper left')

path_figs = "/home/a.brugnoli/Plots_Videos/Python/Plots/Waves/IFAC_WC2020/"
plt.savefig(path_figs + "Hdae_diff.eps", format="eps")

fig = plt.figure()
plt.plot(h_vec, errHode, 'b-')
plt.xlabel(r'Mesh size', fontsize=fntsize)
plt.ylabel(r'$||H_{ref} - H_{ode}||_{L^2}$', fontsize=fntsize)
plt.title(r"$L^2$ norm Hamiltonian difference", fontsize=fntsize)
# plt.legend(loc='upper left')

plt.savefig(path_figs + "Hode_diff.eps", format="eps")

fig = plt.figure()
plt.plot(h_vec, errHode, 'b-', label="ODE")
plt.plot(h_vec, errHdae, 'b-', label="DAE")
plt.xlabel(r'Mesh size', fontsize=fntsize)
plt.ylabel(r'$||H_{ref} - H_{ode}||_{L^2}$', fontsize=fntsize)
plt.title(r"$L^2$ norm Hamiltonian difference", fontsize=fntsize)
plt.legend(loc='upper left')

plt.savefig(path_figs + "Hall_diff.eps", format="eps")

plt.show()