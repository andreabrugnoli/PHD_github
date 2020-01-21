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

path_res = "./convergence_results_mindlin/"
bc_input= "CCCC_AFW_"
save_res = False

coeff = 0.95

# h1_vec = np.load(path_res + bc_input + "h1.npy")
# h2_vec = np.load(path_res + bc_input + "h3.npy")
#
# n_h = len(h1_vec)
# v_errInf_r1 = np.load(path_res + bc_input + "v_errInf_r1.npy")
# v_errQuad_r1 = np.load(path_res + bc_input + "v_errQuad_r1.npy")
#
# v_errInf_r2 = np.load(path_res + bc_input + "v_errInf_r2.npy")
# v_errQuad_r2 = np.load(path_res + bc_input + "v_errQuad_r2.npy")
#
# v_errInf_r3 = np.load(path_res + bc_input + "v_errInf_r3.npy")
# v_errQuad_r3 = np.load(path_res + bc_input + "v_errQuad_r3.npy")
#
# om_errInf_r1 = np.load(path_res + bc_input + "om_errInf_r1.npy")
# om_errQuad_r1 = np.load(path_res + bc_input + "om_errQuad_r1.npy")
#
# om_errInf_r2 = np.load(path_res + bc_input + "om_errInf_r2.npy")
# om_errQuad_r2 = np.load(path_res + bc_input + "om_errQuad_r2.npy")
#
# om_errInf_r3 = np.load(path_res + bc_input + "om_errInf_r3.npy")
# om_errQuad_r3 = np.load(path_res + bc_input + "om_errQuad_r3.npy")
#
# sig_errInf_r1 = np.load(path_res + bc_input + "sig_errInf_r1.npy")
# sig_errQuad_r1 = np.load(path_res + bc_input + "sig_errQuad_r1.npy")
#
# sig_errInf_r2 = np.load(path_res + bc_input + "sig_errInf_r2.npy")
# sig_errQuad_r2 = np.load(path_res + bc_input + "sig_errQuad_r2.npy")
#
# sig_errInf_r3 = np.load(path_res + bc_input + "sig_errInf_r3.npy")
# sig_errQuad_r3 = np.load(path_res + bc_input + "sig_errQuad_r3.npy")
#
# q_errInf_r1 = np.load(path_res + bc_input + "q_errInf_r1.npy")
# q_errQuad_r1 = np.load(path_res + bc_input + "q_errQuad_r1.npy")
#
# q_errInf_r2 = np.load(path_res + bc_input + "q_errInf_r2.npy")
# q_errQuad_r2 = np.load(path_res + bc_input + "q_errQuad_r2.npy")
#
# q_errInf_r3 = np.load(path_res + bc_input + "q_errInf_r3.npy")
# q_errQuad_r3 = np.load(path_res + bc_input + "q_errQuad_r3.npy")
#
# r_errInf_r1 = np.load(path_res + bc_input + "r_errInf_r1.npy")
# r_errQuad_r1 = np.load(path_res + bc_input + "r_errQuad_r1.npy")
#
# r_errInf_r2 = np.load(path_res + bc_input + "r_errInf_r2.npy")
# r_errQuad_r2 = np.load(path_res + bc_input + "r_errQuad_r2.npy")
#
# r_errInf_r3 = np.load(path_res + bc_input + "r_errInf_r3.npy")
# r_errQuad_r3 = np.load(path_res + bc_input + "r_errQuad_r3.npy")

h1_vec = np.load(path_res + bc_input + "h1.npy")[:-1]
h2_vec = np.load(path_res + bc_input + "h3.npy")[:-1]

n_h = len(h1_vec)
v_errInf_r1 = np.load(path_res + bc_input + "v_errInf_r1.npy")[:-1]
v_errQuad_r1 = np.load(path_res + bc_input + "v_errQuad_r1.npy")[:-1]

v_errInf_r2 = np.load(path_res + bc_input + "v_errInf_r2.npy")[:-1]
v_errQuad_r2 = np.load(path_res + bc_input + "v_errQuad_r2.npy")[:-1]

v_errInf_r3 = np.load(path_res + bc_input + "v_errInf_r3.npy")[:-1]
v_errQuad_r3 = np.load(path_res + bc_input + "v_errQuad_r3.npy")[:-1]

om_errInf_r1 = np.load(path_res + bc_input + "om_errInf_r1.npy")[:-1]
om_errQuad_r1 = np.load(path_res + bc_input + "om_errQuad_r1.npy")[:-1]

om_errInf_r2 = np.load(path_res + bc_input + "om_errInf_r2.npy")[:-1]
om_errQuad_r2 = np.load(path_res + bc_input + "om_errQuad_r2.npy")[:-1]

om_errInf_r3 = np.load(path_res + bc_input + "om_errInf_r3.npy")[:-1]
om_errQuad_r3 = np.load(path_res + bc_input + "om_errQuad_r3.npy")[:-1]

sig_errInf_r1 = np.load(path_res + bc_input + "sig_errInf_r1.npy")[:-1]
sig_errQuad_r1 = np.load(path_res + bc_input + "sig_errQuad_r1.npy")[:-1]

sig_errInf_r2 = np.load(path_res + bc_input + "sig_errInf_r2.npy")[:-1]
sig_errQuad_r2 = np.load(path_res + bc_input + "sig_errQuad_r2.npy")[:-1]

sig_errInf_r3 = np.load(path_res + bc_input + "sig_errInf_r3.npy")[:-1]
sig_errQuad_r3 = np.load(path_res + bc_input + "sig_errQuad_r3.npy")[:-1]

q_errInf_r1 = np.load(path_res + bc_input + "q_errInf_r1.npy")[:-1]
q_errQuad_r1 = np.load(path_res + bc_input + "q_errQuad_r1.npy")[:-1]

q_errInf_r2 = np.load(path_res + bc_input + "q_errInf_r2.npy")[:-1]
q_errQuad_r2 = np.load(path_res + bc_input + "q_errQuad_r2.npy")[:-1]

q_errInf_r3 = np.load(path_res + bc_input + "q_errInf_r3.npy")[:-1]
q_errQuad_r3 = np.load(path_res + bc_input + "q_errQuad_r3.npy")[:-1]

r_errInf_r1 = np.load(path_res + bc_input + "r_errInf_r1.npy")[:-1]
r_errQuad_r1 = np.load(path_res + bc_input + "r_errQuad_r1.npy")[:-1]

r_errInf_r2 = np.load(path_res + bc_input + "r_errInf_r2.npy")[:-1]
r_errQuad_r2 = np.load(path_res + bc_input + "r_errQuad_r2.npy")[:-1]

r_errInf_r3 = np.load(path_res + bc_input + "r_errInf_r3.npy")[:-1]
r_errQuad_r3 = np.load(path_res + bc_input + "r_errQuad_r3.npy")[:-1]
#
# v_errInf_r1 = np.load(path_res + bc_input + "v_errF_r1.npy")
# v_errQuad_r1 = np.load(path_res + bc_input + "v_errQuad_r1.npy")[:-1]
#
# v_errInf_r2 = np.load(path_res + bc_input + "v_errF_r2.npy")
# v_errQuad_r2 = np.load(path_res + bc_input + "v_errQuad_r2.npy")[:-1]
#
# v_errInf_r3 = np.load(path_res + bc_input + "v_errF_r3.npy")
# v_errQuad_r3 = np.load(path_res + bc_input + "v_errQuad_r3.npy")[:-1]
#
# om_errInf_r1 = np.load(path_res + bc_input + "om_errF_r1.npy")
# om_errQuad_r1 = np.load(path_res + bc_input + "om_errQuad_r1.npy")[:-1]
#
# om_errInf_r2 = np.load(path_res + bc_input + "om_errF_r2.npy")
# om_errQuad_r2 = np.load(path_res + bc_input + "om_errQuad_r2.npy")[:-1]
#
# om_errInf_r3 = np.load(path_res + bc_input + "om_errF_r3.npy")
# om_errQuad_r3 = np.load(path_res + bc_input + "om_errQuad_r3.npy")[:-1]


v_r1_max = np.zeros((n_h-1,))
v_r1_L2 = np.zeros((n_h-1,))

v_r2_max = np.zeros((n_h-1,))
v_r2_L2 = np.zeros((n_h-1,))

v_r3_max = np.zeros((n_h-1,))
v_r3_L2 = np.zeros((n_h-1,))

om_r1_max = np.zeros((n_h-1,))
om_r1_L2 = np.zeros((n_h-1,))

om_r2_max = np.zeros((n_h-1,))
om_r2_L2 = np.zeros((n_h-1,))

om_r3_max = np.zeros((n_h-1,))
om_r3_L2 = np.zeros((n_h-1,))

sig_r1_max = np.zeros((n_h-1,))
sig_r1_L2 = np.zeros((n_h-1,))

sig_r2_max = np.zeros((n_h-1,))
sig_r2_L2 = np.zeros((n_h-1,))

sig_r3_max = np.zeros((n_h-1,))
sig_r3_L2 = np.zeros((n_h-1,))

q_r1_max = np.zeros((n_h-1,))
q_r1_L2 = np.zeros((n_h-1,))

q_r2_max = np.zeros((n_h-1,))
q_r2_L2 = np.zeros((n_h-1,))

q_r3_max = np.zeros((n_h-1,))
q_r3_L2 = np.zeros((n_h-1,))

r_r1_max = np.zeros((n_h-1,))
r_r1_L2 = np.zeros((n_h-1,))

r_r2_max = np.zeros((n_h-1,))
r_r2_L2 = np.zeros((n_h-1,))

r_r3_max = np.zeros((n_h-1,))
r_r3_L2 = np.zeros((n_h-1,))

for i in range(1, n_h):

    v_r1_max[i-1] = np.log(v_errInf_r1[i]/v_errInf_r1[i-1])/np.log(h1_vec[i]/h1_vec[i-1])
    v_r1_L2[i-1] = np.log(v_errQuad_r1[i]/v_errQuad_r1[i-1])/np.log(h1_vec[i]/h1_vec[i-1])

    v_r2_max[i-1] = np.log(v_errInf_r2[i]/v_errInf_r2[i-1])/np.log(h1_vec[i]/h1_vec[i-1])
    v_r2_L2[i-1] = np.log(v_errQuad_r2[i]/v_errQuad_r2[i-1])/np.log(h1_vec[i]/h1_vec[i-1])

    v_r3_max[i-1] = np.log(v_errInf_r3[i]/v_errInf_r3[i-1])/np.log(h2_vec[i]/h2_vec[i-1])
    v_r3_L2[i-1] = np.log(v_errQuad_r3[i]/v_errQuad_r3[i-1])/np.log(h2_vec[i]/h2_vec[i-1])

    om_r1_max[i - 1] = np.log(om_errInf_r1[i] / om_errInf_r1[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])
    om_r1_L2[i - 1] = np.log(om_errQuad_r1[i] / om_errQuad_r1[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])

    om_r2_max[i - 1] = np.log(om_errInf_r2[i] / om_errInf_r2[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])
    om_r2_L2[i - 1] = np.log(om_errQuad_r2[i] / om_errQuad_r2[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])

    om_r3_max[i - 1] = np.log(om_errInf_r3[i] / om_errInf_r3[i - 1]) / np.log(h2_vec[i] / h2_vec[i - 1])
    om_r3_L2[i - 1] = np.log(om_errQuad_r3[i] / om_errQuad_r3[i - 1]) / np.log(h2_vec[i] / h2_vec[i - 1])

    sig_r1_max[i - 1] = np.log(sig_errInf_r1[i] / sig_errInf_r1[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])
    sig_r1_L2[i - 1] = np.log(sig_errQuad_r1[i] / sig_errQuad_r1[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])

    sig_r2_max[i - 1] = np.log(sig_errInf_r2[i] / sig_errInf_r2[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])
    sig_r2_L2[i - 1] = np.log(sig_errQuad_r2[i] / sig_errQuad_r2[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])

    sig_r3_max[i - 1] = np.log(sig_errInf_r3[i] / sig_errInf_r3[i - 1]) / np.log(h2_vec[i] / h2_vec[i - 1])
    sig_r3_L2[i - 1] = np.log(sig_errQuad_r3[i] / sig_errQuad_r3[i - 1]) / np.log(h2_vec[i] / h2_vec[i - 1])

    q_r1_max[i - 1] = np.log(q_errInf_r1[i] / q_errInf_r1[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])
    q_r1_L2[i - 1] = np.log(q_errQuad_r1[i] / q_errQuad_r1[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])

    q_r2_max[i - 1] = np.log(q_errInf_r2[i] / q_errInf_r2[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])
    q_r2_L2[i - 1] = np.log(q_errQuad_r2[i] / q_errQuad_r2[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])

    q_r3_max[i - 1] = np.log(q_errInf_r3[i] / q_errInf_r3[i - 1]) / np.log(h2_vec[i] / h2_vec[i - 1])
    q_r3_L2[i - 1] = np.log(q_errQuad_r3[i] / q_errQuad_r3[i - 1]) / np.log(h2_vec[i] / h2_vec[i - 1])

    r_r1_max[i - 1] = np.log(r_errInf_r1[i] / r_errInf_r1[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])
    r_r1_L2[i - 1] = np.log(r_errQuad_r1[i] / r_errQuad_r1[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])

    r_r2_max[i - 1] = np.log(r_errInf_r2[i] / r_errInf_r2[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])
    r_r2_L2[i - 1] = np.log(r_errQuad_r2[i] / r_errQuad_r2[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])

    r_r3_max[i - 1] = np.log(r_errInf_r3[i] / r_errInf_r3[i - 1]) / np.log(h2_vec[i] / h2_vec[i - 1])
    r_r3_L2[i - 1] = np.log(r_errQuad_r3[i] / r_errQuad_r3[i - 1]) / np.log(h2_vec[i] / h2_vec[i - 1])

v_r1int_max = np.polyfit(np.log(h1_vec), np.log(v_errInf_r1), 1)[0]
v_r1int_L2 = np.polyfit(np.log(h1_vec), np.log(v_errQuad_r1), 1)[0]

print("Estimated order of convergence r=1 for v Linf: " + str(v_r1_max))
print("Interpolated order of convergence r=1 for v Linf: " + str(v_r1int_max))
print("Estimated order of convergence r=1 for v L2: " + str(v_r1_L2))
print("Interpolated order of convergence r=1 for v L2: " + str(v_r1int_L2))
print("")

v_r2int_max = np.polyfit(np.log(h1_vec), np.log(v_errInf_r2), 1)[0]
v_r2int_L2 = np.polyfit(np.log(h1_vec), np.log(v_errQuad_r2), 1)[0]

print("Estimated order of convergence r=2 for v Linf: " + str(v_r2_max))
print("Interpolated order of convergence r=2 for v Linf: " + str(v_r2int_max))
print("Estimated order of convergence r=2 for v L2: " + str(v_r2_L2))
print("Interpolated order of convergence r=2 for v L2: " + str(v_r2int_L2))
print("")

v_r3int_max = np.polyfit(np.log(h2_vec), np.log(v_errInf_r3), 1)[0]
v_r3int_L2 = np.polyfit(np.log(h2_vec), np.log(v_errQuad_r3), 1)[0]

print("Estimated order of convergence r=3 for v Linf: " + str(v_r3_max))
print("Interpolated order of convergence r=3 for v Linf: " + str(v_r3int_max))
print("Estimated order of convergence r=3 for v L2: " + str(v_r3_L2))
print("Interpolated order of convergence r=3 for v L2: " + str(v_r3int_L2))
print("")

plt.figure()

# plt.plot(np.log(h1_vec), np.log(v_r1_atF), ':o', label='BEC 1')
plt.plot(np.log(h1_vec), np.log(v_errInf_r1), '-.+', label='$k=1$')
plt.plot(np.log(h1_vec), np.log(h1_vec) + coeff*(np.log(v_errInf_r1)[-1] - np.log(h1_vec)[-1]), '-v', label=r'$h$')

# plt.plot(np.log(h1_vec), np.log(v_r2_atF), ':o', label='BEC 2')
plt.plot(np.log(h1_vec), np.log(v_errInf_r2), '-.+', label='$k=2$')
plt.plot(np.log(h1_vec), np.log(h1_vec**2) + coeff*(np.log(v_errInf_r2)[-1] - np.log(h1_vec**2)[-1]), '-v', label=r'$h^2$')

# plt.plot(np.log(h2_vec), np.log(v_r3_atF), ':o', label='BEC 3')
plt.plot(np.log(h2_vec), np.log(v_errInf_r3), '-.+', label='$k=3$')
plt.plot(np.log(h2_vec), np.log(h2_vec**3) + coeff*(np.log(v_errInf_r3)[-1] - np.log(h2_vec**3)[-1]), '-v', label=r'$h^3$')

plt.xlabel(r'log(Mesh size $h$)')
plt.ylabel(r'log($||e_w - e_w^h||_{L^\infty L^2}$)')
plt.title(r'$e_w$ error (AWF element)')
plt.legend()
path_fig = "/home/a.brugnoli/Plots/Python/Plots/Mindlin_plots/Convergence/firedrake/"
if save_res:
    plt.savefig(path_fig + bc_input + "_vel.eps", format="eps")

om_r1int_max = np.polyfit(np.log(h1_vec), np.log(om_errInf_r1), 1)[0]
om_r1int_L2 = np.polyfit(np.log(h1_vec), np.log(om_errQuad_r1), 1)[0]

print("Estimated order of convergence r=1 for om Linf: " + str(om_r1_max))
print("Interpolated order of convergence r=1 for om Linf: " + str(om_r1int_max))
print("Estimated order of convergence r=1 for om L2: " + str(om_r1_L2))
print("Interpolated order of convergence r=1 for om L2: " + str(om_r1int_L2))
print("")

om_r2int_max = np.polyfit(np.log(h1_vec), np.log(om_errInf_r2), 1)[0]
om_r2int_L2 = np.polyfit(np.log(h1_vec), np.log(om_errQuad_r2), 1)[0]

print("Estimated order of convergence r=2 for om Linf: " + str(om_r2_max))
print("Interpolated order of convergence r=2 for om Linf: " + str(om_r2int_max))
print("Estimated order of convergence r=2 for om L2: " + str(om_r2_L2))
print("Interpolated order of convergence r=2 for om L2: " + str(om_r2int_L2))
print("")

om_r3int_max = np.polyfit(np.log(h2_vec), np.log(om_errInf_r3), 1)[0]
om_r3int_L2 = np.polyfit(np.log(h2_vec), np.log(om_errQuad_r3), 1)[0]

print("Estimated order of convergence r=3 for om Linf: " + str(om_r3_max))
print("Interpolated order of convergence r=3 for om Linf: " + str(om_r3int_max))
print("Estimated order of convergence r=3 for om L2: " + str(om_r3_L2))
print("Interpolated order of convergence r=3 for om L2: " + str(om_r3int_L2))
print("")

plt.figure()

# plt.plot(np.log(h1_vec), np.log(om_r1_atF), ':o', label='BEC 1')
plt.plot(np.log(h1_vec), np.log(om_errInf_r1), '-.+', label='$k=1$')
plt.plot(np.log(h1_vec), np.log(h1_vec) + coeff*(np.log(om_errInf_r1)[-1] - np.log(h1_vec)[-1]), '-v', label=r'$h$')

# plt.plot(np.log(h1_vec), np.log(om_r2_atF), ':o', label='BEC 2')
plt.plot(np.log(h1_vec), np.log(om_errInf_r2), '-.+', label='$k=2$')
plt.plot(np.log(h1_vec), np.log(h1_vec**2) + coeff*(np.log(om_errInf_r2)[-1] - np.log(h1_vec**2)[-1]), '-v', label=r'$h^2$')

# plt.plot(np.log(h2_vec), np.log(om_r3_atF), ':o', label='BEC 3')
plt.plot(np.log(h2_vec), np.log(om_errInf_r3), '-.+', label='$k=3$')
plt.plot(np.log(h2_vec), np.log(h2_vec**3) + coeff*(np.log(om_errInf_r3)[-1] - np.log(h2_vec**3)[-1]), '-v', label=r'$h^3$')

plt.xlabel(r'log(Mesh size $h$)')
plt.ylabel(r'log($||\bm{e}_\theta - \bm{e}_\theta^h||_{L^\infty L^2}$)')
plt.title(r'$\bm{e}_\theta$ error (AWF element)')
plt.legend()
if save_res:
    plt.savefig(path_fig + bc_input + "_om.eps", format="eps")

sig_r1int_max = np.polyfit(np.log(h1_vec), np.log(sig_errInf_r1), 1)[0]
sig_r1int_L2 = np.polyfit(np.log(h1_vec), np.log(sig_errQuad_r1), 1)[0]

print("Estimated order of convergence r=1 for sig Linf: " + str(sig_r1_max))
print("Interpolated order of convergence r=1 for sig Linf: " + str(sig_r1int_max))
print("Estimated order of convergence r=1 for sig L2: " + str(sig_r1_L2))
print("Interpolated order of convergence r=1 for sig L2: " + str(sig_r1int_L2))
print("")

sig_r2int_max = np.polyfit(np.log(h1_vec), np.log(sig_errInf_r2), 1)[0]
sig_r2int_L2 = np.polyfit(np.log(h1_vec), np.log(sig_errQuad_r2), 1)[0]

print("Estimated order of convergence r=2 for sig Linf: " + str(sig_r2_max))
print("Interpolated order of convergence r=2 for sig Linf: " + str(sig_r2int_max))
print("Estimated order of convergence r=2 for sig L2: " + str(sig_r2_L2))
print("Interpolated order of convergence r=2 for sig L2: " + str(sig_r2int_L2))
print("")

sig_r3int_max = np.polyfit(np.log(h2_vec), np.log(sig_errInf_r3), 1)[0]
sig_r3int_L2 = np.polyfit(np.log(h2_vec), np.log(sig_errQuad_r3), 1)[0]

print("Estimated order of convergence r=3 for sig Linf: " + str(sig_r3_max))
print("Interpolated order of convergence r=3 for sig Linf: " + str(sig_r3int_max))
print("Estimated order of convergence r=3 for sig L2: " + str(sig_r3_L2))
print("Interpolated order of convergence r=3 for sig L2: " + str(sig_r3int_L2))
print("")

plt.figure()

# plt.plot(np.log(h1_vec), np.log(sig_r1_atF), ':o', label='BEC 1')
plt.plot(np.log(h1_vec), np.log(sig_errInf_r1), '-.+', label='$k=1$')
plt.plot(np.log(h1_vec), np.log(h1_vec) + coeff*(np.log(sig_errInf_r1)[-1] - np.log(h1_vec)[-1]), '-v', label=r'$h$')

# plt.plot(np.log(h1_vec), np.log(sig_r2_atF), ':o', label='BEC 2')
plt.plot(np.log(h1_vec), np.log(sig_errInf_r2), '-.+', label='$k=2$')
plt.plot(np.log(h1_vec), np.log(h1_vec**2) + coeff*(np.log(sig_errInf_r2)[-1] - np.log(h1_vec**2)[-1]), '-v', label=r'$h^2$')

# plt.plot(np.log(h2_vec), np.log(sig_r3_atF), ':o', label='BEC 3')
plt.plot(np.log(h2_vec), np.log(sig_errInf_r3), '-.+', label='$k=3$')
plt.plot(np.log(h2_vec), np.log(h2_vec**3) + coeff*(np.log(sig_errInf_r3)[-1] - np.log(h2_vec**3)[-1]), '-v', label=r'$h^3$')

plt.xlabel(r'log(Mesh size $h$)')
plt.ylabel(r'log($||\bm{E}_{\kappa} - \bm{E}_{\kappa}^h||_{L^\infty L^2}$)')
plt.title(r'$\bm{E}_\kappa$ error (AFW element)')
plt.legend()
if save_res:
    plt.savefig(path_fig + bc_input + "_sig.eps", format="eps")


q_r1int_max = np.polyfit(np.log(h1_vec), np.log(q_errInf_r1), 1)[0]
q_r1int_L2 = np.polyfit(np.log(h1_vec), np.log(q_errQuad_r1), 1)[0]

print("Estimated order of convergence r=1 for q Linf: " + str(q_r1_max))
print("Interpolated order of convergence r=1 for q Linf: " + str(q_r1int_max))
print("Estimated order of convergence r=1 for q L2: " + str(q_r1_L2))
print("Interpolated order of convergence r=1 for q L2: " + str(q_r1int_L2))
print("")

q_r2int_max = np.polyfit(np.log(h1_vec), np.log(q_errInf_r2), 1)[0]
q_r2int_L2 = np.polyfit(np.log(h1_vec), np.log(q_errQuad_r2), 1)[0]

print("Estimated order of convergence r=2 for q Linf: " + str(q_r2_max))
print("Interpolated order of convergence r=2 for q Linf: " + str(q_r2int_max))
print("Estimated order of convergence r=2 for q L2: " + str(q_r2_L2))
print("Interpolated order of convergence r=2 for q L2: " + str(q_r2int_L2))
print("")

q_r3int_max = np.polyfit(np.log(h2_vec), np.log(q_errInf_r3), 1)[0]
q_r3int_L2 = np.polyfit(np.log(h2_vec), np.log(q_errQuad_r3), 1)[0]

print("Estimated order of convergence r=3 for q Linf: " + str(q_r3_max))
print("Interpolated order of convergence r=3 for q Linf: " + str(q_r3int_max))
print("Estimated order of convergence r=3 for q L2: " + str(q_r3_L2))
print("Interpolated order of convergence r=3 for q L2: " + str(q_r3int_L2))
print("")

plt.figure()

# plt.plot(np.log(h1_vec), np.log(q_r1_atF), ':o', label='BEC 1')
plt.plot(np.log(h1_vec), np.log(q_errInf_r1), '-.+', label='$k=1$')
plt.plot(np.log(h1_vec), np.log(h1_vec) + coeff*(np.log(q_errInf_r1)[-1] - np.log(h1_vec)[-1]), '-v', label=r'$h$')

# plt.plot(np.log(h1_vec), np.log(q_r2_atF), ':o', label='BEC 2')
plt.plot(np.log(h1_vec), np.log(q_errInf_r2), '-.+', label='$k=2$')
plt.plot(np.log(h1_vec), np.log(h1_vec**2) + coeff*(np.log(q_errInf_r2)[-1] - np.log(h1_vec**2)[-1]), '-v', label=r'$h^2$')

# plt.plot(np.log(h2_vec), np.log(q_r3_atF), ':o', label='BEC 3')
plt.plot(np.log(h2_vec), np.log(q_errInf_r3), '-.+', label='$k=3$')
plt.plot(np.log(h2_vec), np.log(h2_vec**3) + coeff*(np.log(q_errInf_r3)[-1] - np.log(h2_vec**3)[-1]), '-v', label=r'$h^3$')

plt.xlabel(r'log(Mesh size $h$)')
plt.ylabel(r'log($||\bm{e}_{\gamma} - \bm{e}_{\gamma}^h||_{L^\infty L^2}$)')
plt.title(r'$\bm{e}_\gamma$ error (AFW element)')
plt.legend()
if save_res:
    plt.savefig(path_fig + bc_input + "_q.eps", format="eps")

r_r1int_max = np.polyfit(np.log(h1_vec), np.log(r_errInf_r1), 1)[0]
r_r1int_L2 = np.polyfit(np.log(h1_vec), np.log(r_errQuad_r1), 1)[0]

print("Estimated order of convergence r=1 for r Linf: " + str(r_r1_max))
print("Interpolated order of convergence r=1 for r Linf: " + str(r_r1int_max))
print("Estimated order of convergence r=1 for r L2: " + str(r_r1_L2))
print("Interpolated order of convergence r=1 for r L2: " + str(r_r1int_L2))
print("")

r_r2int_max = np.polyfit(np.log(h1_vec), np.log(r_errInf_r2), 1)[0]
r_r2int_L2 = np.polyfit(np.log(h1_vec), np.log(r_errQuad_r2), 1)[0]

print("Estimated order of convergence r=2 for r Linf: " + str(r_r2_max))
print("Interpolated order of convergence r=2 for r Linf: " + str(r_r2int_max))
print("Estimated order of convergence r=2 for r L2: " + str(r_r2_L2))
print("Interpolated order of convergence r=2 for r L2: " + str(r_r2int_L2))
print("")

r_r3int_max = np.polyfit(np.log(h2_vec), np.log(r_errInf_r3), 1)[0]
r_r3int_L2 = np.polyfit(np.log(h2_vec), np.log(r_errQuad_r3), 1)[0]

print("Estimated order of convergence r=3 for r Linf: " + str(r_r3_max))
print("Interpolated order of convergence r=3 for r Linf: " + str(r_r3int_max))
print("Estimated order of convergence r=3 for r L2: " + str(r_r3_L2))
print("Interpolated order of convergence r=3 for r L2: " + str(r_r3int_L2))
print("")

plt.figure()

# plt.plot(np.log(h1_vec), np.log(q_r1_atF), ':o', label='BEC 1')
plt.plot(np.log(h1_vec), np.log(r_errInf_r1), '-.+', label='$k=1$')
plt.plot(np.log(h1_vec), np.log(h1_vec) + coeff*(np.log(r_errInf_r1)[-1] - np.log(h1_vec)[-1]), '-v', label=r'$h$')

# plt.plot(np.log(h1_vec), np.log(q_r2_atF), ':o', label='BEC 2')
plt.plot(np.log(h1_vec), np.log(r_errInf_r2), '-.+', label='$k=2$')
plt.plot(np.log(h1_vec), np.log(h1_vec**2) + coeff*(np.log(r_errInf_r2)[-1] - np.log(h1_vec**2)[-1]), '-v', label=r'$h^2$')

# plt.plot(np.log(h2_vec), np.log(q_r3_atF), ':o', label='BEC 3')
plt.plot(np.log(h2_vec), np.log(r_errInf_r3), '-.+', label='$k=3$')
plt.plot(np.log(h2_vec), np.log(h2_vec**3) + coeff*(np.log(r_errInf_r3)[-1] - np.log(h2_vec**3)[-1]), '-v', label=r'$h^3$')

plt.xlabel(r'log(Mesh size $h$)')
plt.ylabel(r'log($||\bm{E}_{r} - \bm{E}_{r}^h||_{L^\infty L^2}$)')
plt.title(r'$\bm{E}_r$ error (AFW element)')
plt.legend()
if save_res:
    plt.savefig(path_fig + bc_input + "_r.eps", format="eps")

plt.show()