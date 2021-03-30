import numpy as np
import matplotlib.pyplot as plt
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

import matplotlib
matplotlib.rcParams["legend.loc"] = 'best'
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble']=r"\usepackage{amsmath} \usepackage{bm}"

path_res = "./errors_data_plate10/"
# path_res = "./errors_data_beam2/"

save_fig = True

coeff = 0.8

h_vec = np.load(path_res + "h_vec.npy")
n_h = len(h_vec)

e_u_err_deg1 = np.load(path_res + "e_u_err_deg1.npy")
e_u_err_deg2 = np.load(path_res + "e_u_err_deg2.npy")
e_u_err_deg3 = np.load(path_res + "e_u_err_deg3.npy")

e_eps_err_deg1 = np.load(path_res + "e_eps_err_deg1.npy")
e_eps_err_deg2 = np.load(path_res + "e_eps_err_deg2.npy")
e_eps_err_deg3 = np.load(path_res + "e_eps_err_deg3.npy")

e_w_err_deg1 = np.load(path_res + "e_w_err_deg1.npy")
e_w_err_deg2 = np.load(path_res + "e_w_err_deg2.npy")
e_w_err_deg3 = np.load(path_res + "e_w_err_deg3.npy")

e_kap_err_deg1 = np.load(path_res + "e_kap_err_deg1.npy")
e_kap_err_deg2 = np.load(path_res + "e_kap_err_deg2.npy")
e_kap_err_deg3 = np.load(path_res + "e_kap_err_deg3.npy")

e_disp_err_deg1 = np.load(path_res + "e_disp_err_deg1.npy")
e_disp_err_deg2 = np.load(path_res + "e_disp_err_deg2.npy")
e_disp_err_deg3 = np.load(path_res + "e_disp_err_deg3.npy")

r_e_u_err_deg1 = np.zeros((n_h-1,))
r_e_u_err_deg2 = np.zeros((n_h-1,))
r_e_u_err_deg3 = np.zeros((n_h-1,))

r_e_eps_err_deg1 = np.zeros((n_h-1,))
r_e_eps_err_deg2 = np.zeros((n_h-1,))
r_e_eps_err_deg3 = np.zeros((n_h-1,))

r_e_w_err_deg1 = np.zeros((n_h-1,))
r_e_w_err_deg2 = np.zeros((n_h-1,))
r_e_w_err_deg3 = np.zeros((n_h-1,))

r_e_kap_err_deg1 = np.zeros((n_h-1,))
r_e_kap_err_deg2 = np.zeros((n_h-1,))
r_e_kap_err_deg3 = np.zeros((n_h-1,))

r_e_disp_err_deg1 = np.zeros((n_h-1,))
r_e_disp_err_deg2 = np.zeros((n_h-1,))
r_e_disp_err_deg3 = np.zeros((n_h-1,))

for i in range(1, n_h):

    r_e_u_err_deg1[i-1] = np.log(e_u_err_deg1[i]/e_u_err_deg1[i-1])/np.log(h_vec[i]/h_vec[i-1])
    r_e_u_err_deg2[i-1] = np.log(e_u_err_deg2[i]/e_u_err_deg2[i-1])/np.log(h_vec[i]/h_vec[i-1])
    r_e_u_err_deg3[i-1] = np.log(e_u_err_deg3[i]/e_u_err_deg3[i-1])/np.log(h_vec[i]/h_vec[i-1])
    
    r_e_eps_err_deg1[i-1] = np.log(e_eps_err_deg1[i]/e_eps_err_deg1[i-1])/np.log(h_vec[i]/h_vec[i-1])
    r_e_eps_err_deg2[i-1] = np.log(e_eps_err_deg2[i]/e_eps_err_deg2[i-1])/np.log(h_vec[i]/h_vec[i-1])
    r_e_eps_err_deg3[i-1] = np.log(e_eps_err_deg3[i]/e_eps_err_deg3[i-1])/np.log(h_vec[i]/h_vec[i-1])
    
    r_e_w_err_deg1[i-1] = np.log(e_w_err_deg1[i]/e_w_err_deg1[i-1])/np.log(h_vec[i]/h_vec[i-1])
    r_e_w_err_deg2[i-1] = np.log(e_w_err_deg2[i]/e_w_err_deg2[i-1])/np.log(h_vec[i]/h_vec[i-1])
    r_e_w_err_deg3[i-1] = np.log(e_w_err_deg3[i]/e_w_err_deg3[i-1])/np.log(h_vec[i]/h_vec[i-1])
    
    r_e_kap_err_deg1[i-1] = np.log(e_kap_err_deg1[i]/e_kap_err_deg1[i-1])/np.log(h_vec[i]/h_vec[i-1])
    r_e_kap_err_deg2[i-1] = np.log(e_kap_err_deg2[i]/e_kap_err_deg2[i-1])/np.log(h_vec[i]/h_vec[i-1])
    r_e_kap_err_deg3[i-1] = np.log(e_kap_err_deg3[i]/e_kap_err_deg3[i-1])/np.log(h_vec[i]/h_vec[i-1])
    
    r_e_disp_err_deg1[i-1] = np.log(e_disp_err_deg1[i]/e_disp_err_deg1[i-1])/np.log(h_vec[i]/h_vec[i-1])
    r_e_disp_err_deg2[i-1] = np.log(e_disp_err_deg2[i]/e_disp_err_deg2[i-1])/np.log(h_vec[i]/h_vec[i-1])
    r_e_disp_err_deg3[i-1] = np.log(e_disp_err_deg3[i]/e_disp_err_deg3[i-1])/np.log(h_vec[i]/h_vec[i-1])


r_int_e_u_deg1 = np.polyfit(np.log(h_vec), np.log(e_u_err_deg1), 1)[0]

print("Error deg=1 for e_u: " + str(e_u_err_deg1))
print("Estimated rate of convergence deg=1 for e_u: " + str(r_e_u_err_deg1))
print("Interpolated rate of convergence deg=1 for e_u: " + str(r_int_e_u_deg1))
print("")

r_int_e_u_deg2 = np.polyfit(np.log(h_vec), np.log(e_u_err_deg2), 1)[0]

print("Error deg=2 for e_u: " + str(e_u_err_deg2))
print("Estimated rate of convergence deg=2 for e_u: " + str(r_e_u_err_deg2))
print("Interpolated rate of convergence deg=2 for e_u: " + str(r_int_e_u_deg2))
print("")

r_int_e_u_deg3 = np.polyfit(np.log(h_vec), np.log(e_u_err_deg3), 1)[0]

print("Error deg=3 for e_u: " + str(e_u_err_deg3))
print("Estimated rate of convergence deg=3 for e_u: " + str(r_e_u_err_deg3))
print("Interpolated rate of convergence deg=3 for e_u: " + str(r_int_e_u_deg3))
print("")

plt.figure()

plt.plot(np.log(h_vec), np.log(e_u_err_deg1), '-.+', label='$k=1$')
plt.plot(np.log(h_vec), np.log(h_vec) + coeff*(np.log(e_u_err_deg1)[-1] \
                                               - np.log(h_vec)[-1]), '-v', label=r'$h$')

plt.plot(np.log(h_vec), np.log(e_u_err_deg2), '-.+', label='$k=2$')
plt.plot(np.log(h_vec), np.log(h_vec**2) + coeff*(np.log(e_u_err_deg2)[-1] \
                                                    - np.log(h_vec**2)[-1]), '-v', label=r'$h^2$')

# plt.plot(np.log(h_vec), np.log(e_u_err_deg3), '-.+', label='$k=3$')
# plt.plot(np.log(h_vec), np.log(h_vec**3) + coeff*(np.log(e_u_err_deg3)[-1] \
#                                                   - np.log(h_vec**3)[-1]), '-v', label=r'$h^3$')

plt.xlabel(r'log(Mesh size $h$)')
plt.ylabel(r'log($||e_u - e_u^h||_{L^\infty H^1}$)')
plt.title(r'Error for $e_u$')
plt.legend()
path_fig = "/home/andrea/Plots/Python/VonKarman/"
if save_fig:
    plt.savefig(path_fig + "u_dot.eps", format="eps")


r_int_e_eps_deg1 = np.polyfit(np.log(h_vec), np.log(e_eps_err_deg1), 1)[0]

print("Error deg=1 for e_eps: " + str(e_eps_err_deg1))
print("Estimated rate of convergence deg=1 for e_eps: " + str(r_e_eps_err_deg1))
print("Interpolated rate of convergence deg=1 for e_eps: " + str(r_int_e_eps_deg1))
print("")

r_int_e_eps_deg2 = np.polyfit(np.log(h_vec), np.log(e_eps_err_deg2), 1)[0]

print("Error deg=2 for e_eps: " + str(e_eps_err_deg2))
print("Estimated rate of convergence deg=2 for e_eps: " + str(r_e_eps_err_deg2))
print("Interpolated rate of convergence deg=2 for e_eps: " + str(r_int_e_eps_deg2))
print("")

r_int_e_eps_deg3 = np.polyfit(np.log(h_vec), np.log(e_eps_err_deg3), 1)[0]

print("Error deg=3 for e_eps: " + str(e_eps_err_deg3))
print("Estimated rate of convergence deg=3 for e_eps: " + str(r_e_eps_err_deg3))
print("Interpolated rate of convergence deg=3 for e_eps: " + str(r_int_e_eps_deg3))
print("")

plt.figure()

plt.plot(np.log(h_vec), np.log(e_eps_err_deg1), '-.+', label='$k=1$')
plt.plot(np.log(h_vec), np.log(h_vec) +0.95*(np.log(e_eps_err_deg1)[-1] \
                                               - np.log(h_vec)[-1]), '-v', label=r'$h$')

plt.plot(np.log(h_vec), np.log(e_eps_err_deg2), '-.+', label='$k=2$')
plt.plot(np.log(h_vec), np.log(h_vec**2) + 0.95*(np.log(e_eps_err_deg2)[-1] \
                                                    - np.log(h_vec**2)[-1]), '-v', label=r'$h^2$')

# plt.plot(np.log(h_vec), np.log(e_eps_err_deg3), '-.+', label='$k=3$')
# plt.plot(np.log(h_vec), np.log(h_vec**3) + coeff*(np.log(e_eps_err_deg3)[-1] \
#                                                   - np.log(h_vec**3)[-1]), '-v', label=r'$h^3$')

plt.xlabel(r'log(Mesh size $h$)')
plt.ylabel(r'log($||e_\varepsilon - e_\varepsilon^h||_{L^\infty L^2}$)')
plt.title(r'Error for $e_\varepsilon$')
plt.legend()
path_fig = "/home/andrea/Plots/Python/VonKarman/"
if save_fig:
    plt.savefig(path_fig + "n_xx.eps", format="eps")


r_int_e_w_deg1 = np.polyfit(np.log(h_vec), np.log(e_w_err_deg1), 1)[0]

print("Error deg=1 for e_w: " + str(e_w_err_deg1))
print("Estimated rate of convergence deg=1 for e_w: " + str(r_e_w_err_deg1))
print("Interpolated rate of convergence deg=1 for e_w: " + str(r_int_e_w_deg1))
print("")

r_int_e_w_deg2 = np.polyfit(np.log(h_vec), np.log(e_w_err_deg2), 1)[0]

print("Error deg=2 for e_w: " + str(e_w_err_deg2))
print("Estimated rate of convergence deg=2 for e_w: " + str(r_e_w_err_deg2))
print("Interpolated rate of convergence deg=2 for e_w: " + str(r_int_e_w_deg2))
print("")

r_int_e_w_deg3 = np.polyfit(np.log(h_vec), np.log(e_w_err_deg3), 1)[0]

print("Error deg=3 for e_w: " + str(e_w_err_deg3))
print("Estimated rate of convergence deg=3 for e_w: " + str(r_e_w_err_deg3))
print("Interpolated rate of convergence deg=3 for e_w: " + str(r_int_e_w_deg3))
print("")

plt.figure()

plt.plot(np.log(h_vec), np.log(e_w_err_deg1), '-.+', label='$k=1$')
plt.plot(np.log(h_vec), np.log(h_vec) + coeff*(np.log(e_w_err_deg1)[-1] \
                                               - np.log(h_vec)[-1]), '-v', label=r'$h$')

plt.plot(np.log(h_vec), np.log(e_w_err_deg2), '-.+', label='$k=2$')
plt.plot(np.log(h_vec), np.log(h_vec**2) + coeff*(np.log(e_w_err_deg2)[-1] \
                                                    - np.log(h_vec**2)[-1]), '-v', label=r'$h^2$')

# plt.plot(np.log(h_vec), np.log(e_w_err_deg3), '-.+', label='$k=3$')
# plt.plot(np.log(h_vec), np.log(h_vec**3) + coeff*(np.log(e_w_err_deg3)[-1] \
#                                                   - np.log(h_vec**3)[-1]), '-v', label=r'$h^3$')

plt.xlabel(r'log(Mesh size $h$)')
plt.ylabel(r'log($||e_w - e_w^h||_{L^\infty H^1}$)')
plt.title(r'Error for $e_w$')
plt.legend()
path_fig = "/home/andrea/Plots/Python/VonKarman/"
if save_fig:
    plt.savefig(path_fig + "w_dot.eps", format="eps")
    
    
r_int_e_kap_deg1 = np.polyfit(np.log(h_vec), np.log(e_kap_err_deg1), 1)[0]

print("Error deg=1 for e_kap: " + str(e_kap_err_deg1))
print("Estimated rate of convergence deg=1 for e_kap: " + str(r_e_kap_err_deg1))
print("Interpolated rate of convergence deg=1 for e_kap: " + str(r_int_e_kap_deg1))
print("")

r_int_e_kap_deg2 = np.polyfit(np.log(h_vec), np.log(e_kap_err_deg2), 1)[0]

print("Error deg=2 for e_kap: " + str(e_kap_err_deg2))
print("Estimated rate of convergence deg=2 for e_kap: " + str(r_e_kap_err_deg2))
print("Interpolated rate of convergence deg=2 for e_kap: " + str(r_int_e_kap_deg2))
print("")

r_int_e_kap_deg3 = np.polyfit(np.log(h_vec), np.log(e_kap_err_deg3), 1)[0]

print("Error deg=3 for e_kap: " + str(e_kap_err_deg3))
print("Estimated rate of convergence deg=3 for e_kap: " + str(r_e_kap_err_deg3))
print("Interpolated rate of convergence deg=3 for e_kap: " + str(r_int_e_kap_deg3))
print("")

plt.figure()

plt.plot(np.log(h_vec), np.log(e_kap_err_deg1), '-.+', label='$k=1$')
plt.plot(np.log(h_vec), np.log(h_vec) + coeff*(np.log(e_kap_err_deg1)[-1] \
                                               - np.log(h_vec)[-1]), '-v', label=r'$h$')

plt.plot(np.log(h_vec), np.log(e_kap_err_deg2), '-.+', label='$k=2$')
plt.plot(np.log(h_vec), np.log(h_vec**2) + coeff*(np.log(e_kap_err_deg2)[-1] \
                                                    - np.log(h_vec**2)[-1]), '-v', label=r'$h^2$')

# plt.plot(np.log(h_vec), np.log(e_kap_err_deg3), '-.+', label='$k=3$')
# plt.plot(np.log(h_vec), np.log(h_vec**3) + coeff*(np.log(e_kap_err_deg3)[-1] \
#                                                   - np.log(h_vec**3)[-1]), '-v', label=r'$h^3$')

plt.xlabel(r'log(Mesh size $h$)')
plt.ylabel(r'log($||e_\kappa - e_\kappa^h||_{L^\infty H^1}$)')
plt.title(r'Error for $e_\kappa$')
plt.legend()
path_fig = "/home/andrea/Plots/Python/VonKarman/"
if save_fig:
    plt.savefig(path_fig + "m_xx.eps", format="eps")
    
    
r_int_e_disp_deg1 = np.polyfit(np.log(h_vec), np.log(e_disp_err_deg1), 1)[0]

print("Error deg=1 for e_disp: " + str(e_disp_err_deg1))
print("Estimated rate of convergence deg=1 for e_disp: " + str(r_e_disp_err_deg1))
print("Interpolated rate of convergence deg=1 for e_disp: " + str(r_int_e_disp_deg1))
print("")

r_int_e_disp_deg2 = np.polyfit(np.log(h_vec), np.log(e_disp_err_deg2), 1)[0]

print("Error deg=2 for e_disp: " + str(e_disp_err_deg2))
print("Estimated rate of convergence deg=2 for e_disp: " + str(r_e_disp_err_deg2))
print("Interpolated rate of convergence deg=2 for e_disp: " + str(r_int_e_disp_deg2))
print("")

r_int_e_disp_deg3 = np.polyfit(np.log(h_vec), np.log(e_disp_err_deg3), 1)[0]

print("Error deg=3 for e_disp: " + str(e_disp_err_deg3))
print("Estimated rate of convergence deg=3 for e_disp: " + str(r_e_disp_err_deg3))
print("Interpolated rate of convergence deg=3 for e_disp: " + str(r_int_e_disp_deg3))
print("")

plt.figure()

plt.plot(np.log(h_vec), np.log(e_disp_err_deg1), '-.+', label='$k=1$')
plt.plot(np.log(h_vec), np.log(h_vec) + coeff*(np.log(e_disp_err_deg1)[-1] \
                                               - 0.9*np.log(h_vec)[-1]), '-v', label=r'$h$')

plt.plot(np.log(h_vec), np.log(e_disp_err_deg2), '-.+', label='$k=2$')
plt.plot(np.log(h_vec), np.log(h_vec**2) + coeff*(np.log(e_disp_err_deg2)[-1] \
                                                    - 0.9*np.log(h_vec**2)[-1]), '-v', label=r'$h^2$')

# plt.plot(np.log(h_vec), np.log(e_disp_err_deg3), '-.+', label='$k=3$')
# plt.plot(np.log(h_vec), np.log(h_vec**3) + coeff*(np.log(e_disp_err_deg3)[-1] \
#                                                   - np.log(h_vec**3)[-1]), '-v', label=r'$h^3$')

plt.xlabel(r'log(Mesh size $h$)')
plt.ylabel(r'log($||w - w^h||_{L^\infty H^1}$)')
plt.title(r'Error for $w$')
plt.legend()
path_fig = "/home/andrea/Plots/Python/VonKarman/"
if save_fig:
    plt.savefig(path_fig + "w.eps", format="eps")
    
    
    