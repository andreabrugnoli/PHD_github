import numpy as np
import matplotlib.pyplot as plt
from tools_plotting import setup
from matplotlib.ticker import FormatStrFormatter
import pickle

save_plots = input("Save plots? ")

path_fig = "/home/andrea/Pictures/PythonPlots/DualField_Maxwell3D/"
bc_case = "_EH"
geo_case = "_3D"

res_file = open("results_maxwell.pkl", "rb")
results = pickle.load(res_file)


t_vec = results["t_span"]
Hdot_vec = results["power"]

bdflow_vec = results["flow"]
bdflow_mid = results["flow_mid"]

bdflowE2H1_mid = results["flowE2H1_mid"]
bdflowH2E1_mid = results["flowH2E1_mid"]
int_bdflow = results["int_flow"]

H_df = results["energy_df"]
H_dual = results["energy_dual"]

H_E2H1 = results["energy_E2H1"]
H_H2E1 = results["energy_H2E1"]

H_E2H2 = results["energy_E2H2"]
H_E1H1 = results["energy_E1H1"]

H_ex = results["energy_ex"]
bdflow_ex_vec = results["flow_ex"]

errL2_E2, errHdiv_E2 = results["err_E2"]
errL2_E1, errHcurl_E1 = results["err_E1"]

errL2_H2, errHdiv_H2 = results["err_H2"]
errL2_H1, errHcurl_H1 = results["err_H1"]

err_Hs, err_H_E2H1, err_H_H2E1 = results["err_H"]

divE2 = results["divE2"]
divH2 = results["divH2"]

dt = t_vec[-1] / (len(t_vec)-1)


plt.figure()
plt.plot(t_vec, divE2, 'r-.') # , label=r"\mathrm{d}^2(E^2_h)")
plt.xlabel(r'Time $[\mathrm{s}]$')
plt.ylabel(r'$||d^2 E^2_h||_{L^2}$')

if save_plots:
    plt.savefig(path_fig + "div_E2" + geo_case + bc_case + ".pdf", format="pdf")

plt.figure()
plt.plot(t_vec, divH2, 'b-.') #, label=r"\mathrm{d}^2(H^2_h)")
plt.xlabel(r'Time $[\mathrm{s}]$')
plt.ylabel(r'$||d^2 H^2_h||_{L^2}$')

if save_plots:
    plt.savefig(path_fig + "div_H2" + geo_case + bc_case + ".pdf", format="pdf")

plt.figure()
plt.plot(t_vec[1:]-dt/2, Hdot_vec-bdflow_mid, 'r-.')
plt.xlabel(r'Time $[\mathrm{s}]$')
plt.ylabel(r'$P_h -<e^\partial_{h}, f^\partial_{h}>_{\partial M}$')
plt.title(r'Power balance conservation')

if save_plots:
    plt.savefig(path_fig + "pow_bal" + geo_case + bc_case + ".pdf", format="pdf")

plt.figure()
plt.plot(t_vec[1:]-dt/2, np.diff(H_E2H1)/dt - bdflowE2H1_mid, 'r-.')
plt.xlabel(r'Time $[\mathrm{s}]$')
plt.ylabel(r'$\dot{H}^{\widehat{2}2}_h-P^{\widehat{2}2}_h$')
# plt.title(r'Conservation law $\dot{H}^{\widehat{2}2}_h-P^{\widehat{2}2}_h$')

if save_plots:
    plt.savefig(path_fig + "pow_balE2H1" + geo_case + bc_case + ".pdf", format="pdf")

plt.figure()
plt.plot(t_vec[1:]-dt/2, np.diff(H_H2E1)/dt - bdflowH2E1_mid, 'r-.')
plt.xlabel(r'Time $[\mathrm{s}]$')
plt.ylabel(r'$\dot{H}^{2\widehat{2}}_h-P^{2\widehat{2}}_h$')
# plt.title(r'Conservation law $\dot{H}^{2\widehat{2}}_h-P^{2\widehat{2}}_h$')

if save_plots:
    plt.savefig(path_fig + "pow_balH2E1" + geo_case + bc_case + ".pdf", format="pdf")

plt.figure()
plt.plot(t_vec, bdflow_vec-bdflow_ex_vec, 'r-.')
plt.xlabel(r'Time $[\mathrm{s}]$')
plt.ylabel(r'$<e^\partial_{h}, f^\partial_{h}>_{\partial M} - <e^\partial_{\mathrm{ex}}, f^\partial_{\mathrm{ex}}>_{\partial M}$')
plt.title(r'Discrete and exact boundary flow')

if save_plots:
    plt.savefig(path_fig + "bd_flow" + geo_case + bc_case + ".pdf", format="pdf")


plt.figure()
plt.plot(t_vec[1:]-dt/2, np.abs(np.diff(H_E2H1)/dt - bdflow_mid), '-v', label=r"$\dot{H}^{\widehat{2}2}_h$")
plt.plot(t_vec[1:]-dt/2, np.abs(np.diff(H_H2E1)/dt - bdflow_mid), '--', label=r"$\dot{H}^{2\widehat{2}}_h$")
# plt.plot(t_vec[1:]-dt/2, np.abs(np.diff(H_E1H1)/dt - bdflow_mid), '-.+', label=r"$\dot{H}^{E^1 H^1}_h$")
# plt.plot(t_vec[1:]-dt/2, np.abs(np.diff(H_E2H2)/dt - bdflow_mid), '--*', label=r"$\dot{H}^{E^2 H^2}_h$")
plt.plot(t_vec[1:]-dt/2, np.abs(np.diff(H_dual)/dt - bdflow_mid), '-.', label=r'$\frac{\dot{H}_{T, h}}{2}$')
plt.xlabel(r'Time $[\mathrm{s}]$')
plt.title(r'$|\dot{H}_h - <e^\partial_{h}, f^\partial_{h}>_{\partial M}|$')
plt.legend()

if save_plots:
    plt.savefig(path_fig + "dHdt" + geo_case + bc_case + ".pdf", format="pdf")

plt.figure()
plt.plot(t_vec, np.abs((H_E2H1 - H_E2H1[0]) - (H_ex-H_ex[0])), '-v', label=r'$\Delta H^{\widehat{2}2}_h$')
plt.plot(t_vec, np.abs((H_H2E1 - H_H2E1[0]) - (H_ex-H_ex[0])), '--', label=r'$\Delta H^{2\widehat{2}}_h$')
# plt.plot(t_vec, np.abs((H_E1H1 - H_E1H1[0]) - (H_ex-H_ex[0])), '--+', label=r'$\Delta H^{E^1 H^1}_h$')
# plt.plot(t_vec, np.abs((H_E2H2 - H_E2H2[0]) - (H_ex-H_ex[0])), '--*', label=r'$\Delta H^{E^2 H^2}_h$')
plt.plot(t_vec, np.abs((H_dual - H_dual[0]) - (H_ex-H_ex[0])), '-.', label=r'$\frac{\Delta H_{T, h}}{2}$')
plt.plot(t_vec, np.abs(int_bdflow - (H_ex-H_ex[0])), '-.+', label=r'$\int_0^t P_h(\tau) d\tau$')
plt.xlabel(r'Time $[\mathrm{s}]$')
plt.title(r'$|\Delta H_h - \Delta H_{\mathrm{ex}}|$')
plt.legend()

if save_plots:
    plt.savefig(path_fig + "deltaH" + geo_case + bc_case + ".pdf", format="pdf")

plt.figure()
plt.plot(t_vec, np.abs(H_E2H1 - H_ex), '-v', label=r'$H^{\widehat{2}2}_h$')
plt.plot(t_vec, np.abs(H_H2E1 - H_ex), '--', label=r'$H^{2\widehat{2}}_h$')
# plt.plot(t_vec, np.abs(H_E1H1 - H_ex), '--+', label=r'$H^{E^1 H^1}_h$')
# plt.plot(t_vec, np.abs(H_E2H2 - H_ex), '--*', label=r'$H^{E^2 H^2}_h$')
plt.plot(t_vec, np.abs(H_dual - H_ex), '-.', label=r'$\frac{H_{T, h}}{2}$')
# plt.plot(t_vec, np.abs(H_df - H_ex), '-.+', label=r'$H_{\mathrm{df}}$')
plt.xlabel(r'Time $[\mathrm{s}]$')
plt.title(r'$|H_h - H_{\mathrm{ex}}|$')
plt.legend()

if save_plots:
    plt.savefig(path_fig + "H" + geo_case + bc_case + ".pdf", format="pdf")

plt.show()

print("Error L2 E2: " + str(errL2_E2))
print("Error Hdiv E2: " + str(errHdiv_E2))

print("Error L2 H2: " + str(errL2_H2))
print("Error Hdiv H2: " + str(errHdiv_H2))

print("Error L2 E1: " + str(errL2_E1))
print("Error Hcurl E1: " + str(errHcurl_E1))

print("Error L2 H1: " + str(errL2_H1))
print("Error Hcurl H1: " + str(errHcurl_H1))

print("Error Hs: " + str(err_Hs))
print("Error H_E2H1: " + str(err_H_E2H1))
print("Error H_H2E1: " + str(err_H_H2E1))

