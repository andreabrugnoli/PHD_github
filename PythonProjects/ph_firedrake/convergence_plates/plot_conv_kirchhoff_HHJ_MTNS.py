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
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{bm}"]

path_res = "./convergence_results_kirchhoff/"
bc_input = "SSSS_HHJ_"
save_res = True

coeff = 0.90
h1_vec = np.load(path_res + bc_input + "h1.npy")
h2_vec = np.load(path_res + bc_input + "h3.npy")

n_h = len(h1_vec)
v_err_r1 = np.load(path_res + bc_input + "v_errF_r1.npy")
v_errInf_r1 = np.load(path_res + bc_input +  "v_errInf_r1.npy")
v_errQuad_r1 = np.load(path_res + bc_input + "v_errQuad_r1.npy")

v_err_r2 = np.load(path_res + bc_input + "v_errF_r2.npy")
v_errInf_r2= np.load(path_res + bc_input + "v_errInf_r2.npy")
v_errQuad_r2 = np.load(path_res + bc_input + "v_errQuad_r2.npy")

v_err_r3 = np.load(path_res + bc_input + "v_errF_r3.npy")
v_errInf_r3 = np.load(path_res + bc_input + "v_errInf_r3.npy")
v_errQuad_r3 = np.load(path_res + bc_input + "v_errQuad_r3.npy")

sig_err_r1 = np.load(path_res + bc_input + "sig_errF_r1.npy")
sig_errInf_r1 = np.load(path_res + bc_input + "sig_errInf_r1.npy")
sig_errQuad_r1 = np.load(path_res + bc_input + "sig_errQuad_r1.npy")

sig_err_r2 = np.load(path_res + bc_input + "sig_errF_r2.npy")
sig_errInf_r2 = np.load(path_res + bc_input + "sig_errInf_r2.npy")
sig_errQuad_r2 = np.load(path_res + bc_input + "sig_errQuad_r2.npy")

sig_err_r3 = np.load(path_res + bc_input + "sig_errF_r3.npy")
sig_errInf_r3 = np.load(path_res + bc_input + "sig_errInf_r3.npy")
sig_errQuad_r3 = np.load(path_res + bc_input + "sig_errQuad_r3.npy")

v_r1_atF = np.zeros((n_h-1,))
v_r1_max = np.zeros((n_h-1,))
v_r1_L2 = np.zeros((n_h-1,))

v_r2_atF = np.zeros((n_h-1,))
v_r2_max = np.zeros((n_h-1,))
v_r2_L2 = np.zeros((n_h-1,))

v_r3_atF = np.zeros((n_h-1,))
v_r3_max = np.zeros((n_h-1,))
v_r3_L2 = np.zeros((n_h-1,))

sig_r1_atF = np.zeros((n_h-1,))
sig_r1_max = np.zeros((n_h-1,))
sig_r1_L2 = np.zeros((n_h-1,))

sig_r2_atF = np.zeros((n_h-1,))
sig_r2_max = np.zeros((n_h-1,))
sig_r2_L2 = np.zeros((n_h-1,))

sig_r3_atF = np.zeros((n_h-1,))
sig_r3_max = np.zeros((n_h-1,))
sig_r3_L2 = np.zeros((n_h-1,))


for i in range(1, n_h):

    v_r1_atF[i-1] = np.log(v_err_r1[i]/v_err_r1[i-1])/np.log(h1_vec[i]/h1_vec[i-1])
    v_r1_max[i-1] = np.log(v_errInf_r1[i]/v_errInf_r1[i-1])/np.log(h1_vec[i]/h1_vec[i-1])
    v_r1_L2[i-1] = np.log(v_errQuad_r1[i]/v_errQuad_r1[i-1])/np.log(h1_vec[i]/h1_vec[i-1])

    v_r2_atF[i-1] = np.log(v_err_r2[i]/v_err_r2[i-1])/np.log(h1_vec[i]/h1_vec[i-1])
    v_r2_max[i-1] = np.log(v_errInf_r2[i]/v_errInf_r2[i-1])/np.log(h1_vec[i]/h1_vec[i-1])
    v_r2_L2[i-1] = np.log(v_errQuad_r2[i]/v_errQuad_r2[i-1])/np.log(h1_vec[i]/h1_vec[i-1])

    v_r3_atF[i-1] = np.log(v_err_r3[i]/v_err_r3[i-1])/np.log(h2_vec[i]/h2_vec[i-1])
    v_r3_max[i-1] = np.log(v_errInf_r3[i]/v_errInf_r3[i-1])/np.log(h2_vec[i]/h2_vec[i-1])
    v_r3_L2[i-1] = np.log(v_errQuad_r3[i]/v_errQuad_r3[i-1])/np.log(h2_vec[i]/h2_vec[i-1])

    sig_r1_atF[i - 1] = np.log(sig_err_r1[i] / sig_err_r1[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])
    sig_r1_max[i - 1] = np.log(sig_errInf_r1[i] / sig_errInf_r1[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])
    sig_r1_L2[i - 1] = np.log(sig_errQuad_r1[i] / sig_errQuad_r1[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])

    sig_r2_atF[i - 1] = np.log(sig_err_r2[i] / sig_err_r2[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])
    sig_r2_max[i - 1] = np.log(sig_errInf_r2[i] / sig_errInf_r2[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])
    sig_r2_L2[i - 1] = np.log(sig_errQuad_r2[i] / sig_errQuad_r2[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])

    sig_r3_atF[i - 1] = np.log(sig_err_r3[i] / sig_err_r3[i - 1]) / np.log(h2_vec[i] / h2_vec[i - 1])
    sig_r3_max[i - 1] = np.log(sig_errInf_r3[i] / sig_errInf_r3[i - 1]) / np.log(h2_vec[i] / h2_vec[i - 1])
    sig_r3_L2[i - 1] = np.log(sig_errQuad_r3[i] / sig_errQuad_r3[i - 1]) / np.log(h2_vec[i] / h2_vec[i - 1])


v_r1int_atF = np.polyfit(np.log(h1_vec), np.log(v_err_r1), 1)[0]
v_r1int_max = np.polyfit(np.log(h1_vec), np.log(v_errInf_r1), 1)[0]
v_r1int_L2 = np.polyfit(np.log(h1_vec), np.log(v_errQuad_r1), 1)[0]

print("Estimated order of convergence r=1 for v at T fin: " + str(v_r1_atF))
print("Interpolated order of convergence r=1 for v at T fin: " + str(v_r1int_atF))
print("Estimated order of convergence r=1 for v Linf: " + str(v_r1_max))
print("Interpolated order of convergence r=1 for v Linf: " + str(v_r1int_max))
print("Estimated order of convergence r=1 for v L2: " + str(v_r1_L2))
print("Interpolated order of convergence r=1 for v L2: " + str(v_r1int_L2))
print("")

v_r2int = np.polyfit(np.log(h1_vec), np.log(v_err_r2), 1)[0]
v_r2int_max = np.polyfit(np.log(h1_vec), np.log(v_errInf_r2), 1)[0]
v_r2int_L2 = np.polyfit(np.log(h1_vec), np.log(v_errQuad_r2), 1)[0]

print("Estimated order of convergence r=2 for v at T fin: " + str(v_r2_atF))
print("Interpolated order of convergence r=2 for v at T fin: " + str(v_r2int))
print("Estimated order of convergence r=2 for v Linf: " + str(v_r2_max))
print("Interpolated order of convergence r=2 for v Linf: " + str(v_r2int_max))
print("Estimated order of convergence r=2 for v L2: " + str(v_r2_L2))
print("Interpolated order of convergence r=2 for v L2: " + str(v_r2int_L2))
print("")

v_r3int = np.polyfit(np.log(h2_vec), np.log(v_err_r3), 1)[0]
v_r3int_max = np.polyfit(np.log(h2_vec), np.log(v_errInf_r3), 1)[0]
v_r3int_L2 = np.polyfit(np.log(h2_vec), np.log(v_errQuad_r3), 1)[0]

print("Estimated order of convergence r=3 for v at T fin: " + str(v_r3_atF))
print("Interpolated order of convergence r=3 for v at T fin: " + str(v_r3int))
print("Estimated order of convergence r=3 for v Linf: " + str(v_r3_max))
print("Interpolated order of convergence r=3 for v Linf: " + str(v_r3int_max))
print("Estimated order of convergence r=3 for v L2: " + str(v_r3_L2))
print("Interpolated order of convergence r=3 for v L2: " + str(v_r3int_L2))
print("")

plt.figure()
# plt.plot(np.log(h1_vec), np.log(v_r1_atF), ':o', label='BEC 1')
plt.plot(np.log(h1_vec), np.log(v_errInf_r1), '-.+', label='$k=1$')
plt.plot(np.log(h1_vec), np.log(h1_vec) + coeff*(np.log(v_errInf_r1)[-1] - np.log(h1_vec)[-1]) + np.log(2), '-v', label=r'$h$')

# plt.plot(np.log(h1_vec), np.log(v_r2_atF), ':o', label='BEC 2')
plt.plot(np.log(h1_vec), np.log(v_errInf_r2), '-.+', label='$k=2$')
plt.plot(np.log(h1_vec), np.log(h1_vec**2) + coeff*(np.log(v_errInf_r2)[-1] - np.log(h1_vec**2)[-1]) + np.log(2), '-v', label=r'$h^2$')

# plt.plot(np.log(h2_vec), np.log(v_r3_atF), ':o', label='BEC 3')
plt.plot(np.log(h2_vec), np.log(v_errInf_r3), '-.+', label='$k=3$')
plt.plot(np.log(h2_vec), np.log(h2_vec**3) + coeff*(np.log(v_errInf_r3)[-1] - np.log(h2_vec**3)[-1]) + np.log(2), '-v', label=r'$h^3$')

plt.xlabel(r'log(Mesh size)')
plt.ylabel(r'log($||e_w - e_w^h||_{L^\infty H^1}$)')
plt.title(r'$e_w$ error (HHJ element)')
plt.legend()
path_fig = "/home/a.brugnoli/Plots/Python/Plots/Kirchhoff_plots/Convergence/firedrake/"
if save_res:
    plt.savefig(path_fig + bc_input + "_vel.eps", format="eps")

sig_r1int_atF = np.polyfit(np.log(h1_vec), np.log(sig_err_r1), 1)[0]
sig_r1int_max = np.polyfit(np.log(h1_vec), np.log(sig_errInf_r1), 1)[0]
sig_r1int_L2 = np.polyfit(np.log(h1_vec), np.log(sig_errQuad_r1), 1)[0]

print("Estimated order of convergence r=1 for sigma at T fin: " + str(sig_r1_atF))
print("Interpolated order of convergence r=1 for sigma at T fin: " + str(sig_r1int_atF))
print("Estimated order of convergence r=1 for sigma Linf: " + str(sig_r1_max))
print("Interpolated order of convergence r=1 for sigma Linf: " + str(sig_r1int_max))
print("Estimated order of convergence r=1 for sigma L2: " + str(sig_r1_L2))
print("Interpolated order of convergence r=1 for sigma L2: " + str(sig_r1int_L2))
print("")

sig_r2int = np.polyfit(np.log(h1_vec), np.log(sig_err_r2), 1)[0]
sig_r2int_max = np.polyfit(np.log(h1_vec), np.log(sig_errInf_r2), 1)[0]
sig_r2int_L2 = np.polyfit(np.log(h1_vec), np.log(sig_errQuad_r2), 1)[0]

print("Estimated order of convergence r=2 for sigma at T fin: " + str(sig_r2_atF))
print("Interpolated order of convergence r=2 for sigma at T fin: " + str(sig_r2int))
print("Estimated order of convergence r=2 for sigma Linf: " + str(sig_r2_max))
print("Interpolated order of convergence r=2 for sigma Linf: " + str(sig_r2int_max))
print("Estimated order of convergence r=2 for sigma L2: " + str(sig_r2_L2))
print("Interpolated order of convergence r=2 for sigma L2: " + str(sig_r2int_L2))
print("")

sig_r3int = np.polyfit(np.log(h2_vec), np.log(sig_err_r3), 1)[0]
sig_r3int_max = np.polyfit(np.log(h2_vec), np.log(sig_errInf_r3), 1)[0]
sig_r3int_L2 = np.polyfit(np.log(h2_vec), np.log(sig_errQuad_r3), 1)[0]

print("Estimated order of convergence r=3 for sigma at T fin: " + str(sig_r3_atF))
print("Interpolated order of convergence r=3 for sigma at T fin: " + str(sig_r3int))
print("Estimated order of convergence r=3 for sigma Linf: " + str(sig_r3_max))
print("Interpolated order of convergence r=3 for sigma Linf: " + str(sig_r3int_max))
print("Estimated order of convergence r=3 for sigma L2: " + str(sig_r3_L2))
print("Interpolated order of convergence r=3 for sigma L2: " + str(sig_r3int_L2))
print("")

plt.figure()


plt.plot(np.log(h1_vec), np.log(sig_errInf_r1), '-.+', label='$k=1$')
plt.plot(np.log(h1_vec), np.log(h1_vec) + coeff*(np.log(sig_errInf_r1)[-1] - np.log(h1_vec)[-1]), '-v', label=r'$h$')

plt.plot(np.log(h1_vec), np.log(sig_errInf_r2), '-.+', label='$k=2$')
plt.plot(np.log(h1_vec), np.log(h1_vec**2) + coeff*(np.log(sig_errInf_r2)[-1] - np.log(h1_vec**2)[-1]), '-v', label=r'$h^2$')

plt.plot(np.log(h2_vec), np.log(sig_errInf_r3), '-.+', label='$k=3$')
plt.plot(np.log(h2_vec), np.log(h2_vec**3) + coeff*(np.log(sig_errInf_r3)[-1] - np.log(h2_vec**3)[-1]), '-v', label=r'$h^3$')

plt.xlabel(r'log(Mesh size $h$)')
plt.ylabel(r'log($||\bm{E}_{\kappa} - \bm{E}_{\kappa}^h||_{L^\infty L^2}$)')
plt.title(r'$\bm{E}_\kappa$ error (HHJ element)')
plt.legend()
if save_res:
    plt.savefig(path_fig + bc_input + "_sig.eps", format="eps")

plt.show()
