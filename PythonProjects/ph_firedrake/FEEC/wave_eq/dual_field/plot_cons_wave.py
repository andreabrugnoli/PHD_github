import numpy as np
import matplotlib.pyplot as plt
from tools_plotting import setup
from matplotlib.ticker import FormatStrFormatter
import pickle

save_plots = input("Save plots? ")

path_fig = "/home/andrea/Pictures/PythonPlots/DualField_wave3D/"
bc_case = "_DN"
geo_case = "_3D"

res_file = open("results_wave.pkl", "rb")
results = pickle.load(res_file)

t_vec = results["t_span"]

Hdot_vec = results["power"]

bdflow_vec = results["flow"]
bdflow_mid = results["flow_mid"]

bdflow10_mid = results["flow10_mid"]
bdflow32_mid = results["flow32_mid"]
int_bdflow = results["int_flow"]

H_df = results["energy_df"]
H_3210 = results["energy_3210"]

H_32 = results["energy_32"]
H_01 = results["energy_01"]

H_31 = results["energy_31"]
H_02 = results["energy_02"]

H_ex = results["energy_ex"]
bdflow_ex_vec = results["flow_ex"]

errL2_p3 = results["err_p3"]
errL2_u1, errHcurl_u1 = results["err_u1"]
errL2_p0, errH1_p0 = results["err_p0"]
errL2_u2, errHdiv_u2 = results["err_u2"]

err_Hs, err_H10, err_H32 = results["err_H"]

dt = t_vec[-1] / (len(t_vec)-1)


plt.figure()
plt.plot(t_vec[1:]-dt/2, Hdot_vec - bdflow_mid, 'r-.')
plt.xlabel(r'Time $[\mathrm{s}]$')
plt.ylabel(r'$P_h -<e^\partial_{h}, f^\partial_{h}>_{\partial M}$')
plt.title(r'Power balance conservation')

if save_plots:
    plt.savefig(path_fig + "pow_bal" + geo_case + bc_case + ".pdf", format="pdf")

plt.figure()
plt.plot(t_vec[1:]-dt/2, np.diff(H_01)/dt - bdflow10_mid, 'r-.')
plt.xlabel(r'Time $[\mathrm{s}]$')
plt.ylabel(r'$\dot{H}^{3\widehat{1}}_h-P^{3\widehat{1}}_h$')
# plt.title(r'Conservation law $\dot{H}^{3\widehat{1}}_h-P^{3\widehat{1}}_h$')

if save_plots:
    plt.savefig(path_fig + "pow_bal10" + geo_case + bc_case + ".pdf", format="pdf")

plt.figure()
plt.plot(t_vec[1:]-dt/2, np.diff(H_32)/dt - bdflow32_mid, 'r-.')
plt.xlabel(r'Time $[\mathrm{s}]$')
plt.ylabel(r'$\dot{H}^{\widehat{3}1}_h-P^{\widehat{3}1}_h$')
# plt.title(r'Conservation law $\dot{H}^{\widehat{3}1}_h-P^{\widehat{3}1}_h$')

if save_plots:
    plt.savefig(path_fig + "pow_bal32" + geo_case + bc_case + ".pdf", format="pdf")

plt.figure()
ax = plt.gca()
plt.plot(t_vec, bdflow_vec - bdflow_ex_vec, 'r-.')
plt.xlabel(r'Time $[\mathrm{s}]$')
plt.ylabel(r'$<e^\partial_{h}, f^\partial_{h}>_{\partial M} - <e^\partial_{\mathrm{ex}}, f^\partial_{\mathrm{ex}}>_{\partial M}$')
plt.title(r'Discrete and exact boundary flow')

if save_plots:
    plt.savefig(path_fig + "bd_flow" + geo_case + bc_case + ".pdf", format="pdf")


plt.figure()
plt.plot(t_vec[1:]-dt/2, np.abs(np.diff(H_01)/dt - bdflow_mid), '-v', label=r"$\dot{H}_{h}^{3\widehat{1}}$")
plt.plot(t_vec[1:]-dt/2, np.abs(np.diff(H_32)/dt - bdflow_mid), '--', label=r"$\dot{H}_{h}^{\widehat{3}1}$")
# plt.plot(t_vec[1:]-dt/2, np.abs(np.diff(H_02)/dt - bdflow_mid), '-.+', label=r"$\dot{H}^{02}$")
# plt.plot(t_vec[1:]-dt/2, np.abs(np.diff(H_31)/dt - bdflow_mid), '--*', label=r"$\dot{H}^{31}$")
plt.plot(t_vec[1:]-dt/2, np.abs(np.diff(H_3210)/dt - bdflow_mid), '-.', label=r'$\frac{\dot{H}_{T, h}}{2}$')
plt.xlabel(r'Time $[\mathrm{s}]$')
plt.title(r'$|\dot{H}_h - <e^\partial_{h}, f^\partial_{h}>_{\partial M}|$')
plt.legend()

if save_plots:
    plt.savefig(path_fig + "dHdt" + geo_case + bc_case + ".pdf", format="pdf")

plt.figure()
plt.plot(t_vec, np.abs((H_01 - H_01[0]) - (H_ex-H_ex[0])), '-v', label=r'$\Delta H_{h}^{3\widehat{1}}$')
plt.plot(t_vec, np.abs((H_32 - H_32[0]) - (H_ex-H_ex[0])), '--', label=r'$\Delta H_{h}^{\widehat{3}1}$')
# plt.plot(t_vec, np.abs((H_02 - H_02[0]) - (H_ex-H_ex[0])), '--+', label=r'$\Delta H^{02}$')
# plt.plot(t_vec, np.abs((H_31 - H_31[0]) - (H_ex-H_ex[0])), '--*', label=r'$\Delta H^{31}$')
plt.plot(t_vec, np.abs((H_3210 - H_3210[0]) - (H_ex-H_ex[0])), '-.', label=r'$\frac{\Delta H_{T, h}}{2}$')
plt.plot(t_vec, np.abs(int_bdflow - (H_ex-H_ex[0])), '-.+', label=r'$\int_0^t P_h(\tau) d\tau$')
plt.xlabel(r'Time $[\mathrm{s}]$')
plt.title(r'$|\Delta H_h - \Delta H_{\mathrm{ex}}|$')
plt.legend()

if save_plots:
    plt.savefig(path_fig + "deltaH" + geo_case + bc_case + ".pdf", format="pdf")

plt.figure()
plt.plot(t_vec, np.abs(H_01 - H_ex), '-v', label=r'$H_h^{3\widehat{1}}$')
plt.plot(t_vec, np.abs(H_32 - H_ex), '--', label=r'$H_h^{\widehat{3}1}$')
# plt.plot(t_vec, np.abs(H_02 - H_ex), '--+', label=r'$H^{02}$')
# plt.plot(t_vec, np.abs(H_31 - H_ex), '--*', label=r'$H^{31}$')
plt.plot(t_vec, np.abs(H_3210 - H_ex), '-.', label=r'$\frac{H_{T, h}}{2}$')
# plt.plot(t_vec, np.abs(H_df - H_ex), '-.+', label=r'$H_{\mathrm{df}}$')
plt.xlabel(r'Time $[\mathrm{s}]$')
plt.title(r'$|H_h - H_{\mathrm{ex}}|$')
plt.legend()

if save_plots:
    plt.savefig(path_fig + "H" + geo_case + bc_case + ".pdf", format="pdf")

plt.show()


# print("Error L2 p3: " + str(errL2_p3))
#
# print("Error L2 u1: " + str(errL2_u1))
# print("Error Hcurl u1: " + str(errHcurl_u1))
#
# print("Error L2 p0: " + str(errL2_p0))
# print("Error H1 p0: " + str(errH1_p0))
#
# print("Error L2 u2: " + str(errL2_u2))
# print("Error Hdiv u2: " + str(errHdiv_u2))
#
# print("Error Hs: " + str(err_Hs))
# print("Error H_10: " + str(err_H10))
# print("Error H_32: " + str(err_H32))
