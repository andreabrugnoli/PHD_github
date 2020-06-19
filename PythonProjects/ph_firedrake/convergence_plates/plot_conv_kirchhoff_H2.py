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
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{bm}"]

save_fig = True
path_res = "./convergence_results_kirchhoff/"
bc = "SSSS_BellDG3_"

h1_vec = np.load(path_res + bc + "h1.npy")
print(h1_vec)
coeff = 0.95

n_h = len(h1_vec)
v_err_r1 = np.load(path_res + bc + "v_errF_r1.npy")
v_errInf_r1 = np.load(path_res + bc +  "v_errInf_r1.npy")
v_errQuad_r1 = np.load(path_res + bc + "v_errQuad_r1.npy")

sig_err_r1 = np.load(path_res + bc + "sig_errF_r1.npy")
sig_errInf_r1 = np.load(path_res + bc + "sig_errInf_r1.npy")
sig_errQuad_r1 = np.load(path_res + bc + "sig_errQuad_r1.npy")



v_r1_atF = np.zeros((n_h-1,))
v_r1_max = np.zeros((n_h-1,))
v_r1_L2 = np.zeros((n_h-1,))

sig_r1_atF = np.zeros((n_h-1,))
sig_r1_max = np.zeros((n_h-1,))
sig_r1_L2 = np.zeros((n_h-1,))


for i in range(1, n_h):

    v_r1_atF[i-1] = np.log(v_err_r1[i]/v_err_r1[i-1])/np.log(h1_vec[i]/h1_vec[i-1])
    v_r1_max[i-1] = np.log(v_errInf_r1[i]/v_errInf_r1[i-1])/np.log(h1_vec[i]/h1_vec[i-1])
    v_r1_L2[i-1] = np.log(v_errQuad_r1[i]/v_errQuad_r1[i-1])/np.log(h1_vec[i]/h1_vec[i-1])

    sig_r1_atF[i - 1] = np.log(sig_err_r1[i] / sig_err_r1[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])
    sig_r1_max[i - 1] = np.log(sig_errInf_r1[i] / sig_errInf_r1[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])
    sig_r1_L2[i - 1] = np.log(sig_errQuad_r1[i] / sig_errQuad_r1[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])

v_r1int_atF = np.polyfit(np.log(h1_vec), np.log(v_err_r1), 1)[0]
v_r1int_max = np.polyfit(np.log(h1_vec), np.log(v_errInf_r1), 1)[0]
v_r1int_L2 = np.polyfit(np.log(h1_vec), np.log(v_errQuad_r1), 1)[0]

print("Error for v Linf: " + str(v_errInf_r1))
print("Error for v L2: " + str(v_errQuad_r1))
print("Estimated order of convergence r=1 for v at T fin: " + str(v_r1_atF))
print("Interpolated order of convergence r=1 for v at T fin: " + str(v_r1int_atF))
print("Estimated order of convergence r=1 for v Linf: " + str(v_r1_max))
print("Interpolated order of convergence r=1 for v Linf: " + str(v_r1int_max))
print("Estimated order of convergence r=1 for v L2: " + str(v_r1_L2))
print("Interpolated order of convergence r=1 for v L2: " + str(v_r1int_L2))
print("")


plt.figure()

# plt.plot(np.log(h1_vec), np.log(v_r1_atF), ':o', label='H2')
plt.plot(np.log(h1_vec), np.log(v_errInf_r1), '-.+', label='Bell')
# plt.plot(np.log(h1_vec), np.log(v_errQuad_r1), '--*', label='H2 $L^2$')
plt.plot(np.log(h1_vec), np.log(h1_vec) + coeff*(np.log(v_errInf_r1)[-1] - np.log(h1_vec)[-1]), '-v', label=r'$h$')
plt.plot(np.log(h1_vec), np.log(h1_vec**2) + 1.02*(np.log(v_errInf_r1)[-1] - np.log(h1_vec**2)[-1]), '-v', label=r'$h^2$')

plt.xlabel(r'log(Mesh size $h$)')
plt.ylabel(r'log($||e_w - e_w^h||_{L^\infty H^2}$)')
plt.title(r'$e_w$ error (BellDG3 element)')
plt.legend()
path_fig = "/home/a.brugnoli/Plots/Python/Plots/Kirchhoff_plots/Convergence/firedrake/"

if save_fig:
    plt.savefig(path_fig + bc + "_vel.eps", format="eps")

sig_r1int_atF = np.polyfit(np.log(h1_vec), np.log(sig_err_r1), 1)[0]
sig_r1int_max = np.polyfit(np.log(h1_vec), np.log(sig_errInf_r1), 1)[0]
sig_r1int_L2 = np.polyfit(np.log(h1_vec), np.log(sig_errQuad_r1), 1)[0]

print("Error for sig Linf: " + str(sig_errInf_r1))
print("Error for sig L2: " + str(sig_errQuad_r1))
print("Estimated order of convergence r=1 for sigma at T fin: " + str(sig_r1_atF))
print("Interpolated order of convergence r=1 for sigma at T fin: " + str(sig_r1int_atF))
print("Estimated order of convergence r=1 for sigma Linf: " + str(sig_r1_max))
print("Interpolated order of convergence r=1 for sigma Linf: " + str(sig_r1int_max))
print("Estimated order of convergence r=1 for sigma L2: " + str(sig_r1_L2))
print("Interpolated order of convergence r=1 for sigma L2: " + str(sig_r1int_L2))
print("")

plt.figure()

# plt.plot(np.log(h1_vec), np.log(sig_r1_atF), ':o', label='H2')
plt.plot(np.log(h1_vec), np.log(sig_errInf_r1), '-.+', label='DG $k =3$')
# plt.plot(np.log(h1_vec), np.log(sig_errQuad_r1), '--*', label='H2 $L^2$')
plt.plot(np.log(h1_vec), np.log(h1_vec)+ coeff*(np.log(sig_errInf_r1)[-1] - np.log(h1_vec)[-1]), '-v', label=r'$h$')
plt.plot(np.log(h1_vec), np.log(h1_vec**2) + 1.02*(np.log(sig_errInf_r1)[-1] - np.log(h1_vec**2)[-1]), '-v', label=r'$h^2$')

plt.xlabel(r'log(Mesh size $h$)')
plt.ylabel(r'log($||\bm{E}_{\kappa} - \bm{E}_{\kappa}^h||_{L^\infty L^2}$)')
plt.title(r'$\bm{E}_\kappa$ error (BellDG3 element)')
plt.legend()
if save_fig:
    plt.savefig(path_fig + bc + "_sig.eps", format="eps")
plt.show()