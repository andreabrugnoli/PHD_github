import numpy as np
import matplotlib.pyplot as plt
from tools_plotting import setup

save_plots = input("Save plots? ")
path_fig = "/home/andrea/Pictures/PythonPlots/Hybridization_maxwell/"
path_res = "results_hybrid/"
bc_case = "_EH"
geo_case = "_3D"

deg_vec = np.arange(1, 4)

h_dict = {}

# E1H2
err_E1_dict = {}
err_H2_dict = {}
err_H1nor_dict = {}
err_E1tan_dict = {}

# E2H1
err_E2_dict = {}
err_H1_dict = {}
err_H1tan_dict = {}
err_E1nor_dict = {}

# Dual Field
err_E12_dict = {}
err_H12_dict = {}

# Orders
# E1H2
ord_E1_dict = {}
ord_H2_dict = {}
ord_H1nor_dict = {}
ord_E1tan_dict = {}
# E2H1
ord_E2_dict = {}
ord_H1_dict = {}
ord_H1tan_dict = {}
ord_E1nor_dict = {}

# Dual field
ord_E12_dict = {}
ord_H12_dict = {}


for ii in deg_vec:
    h_deg_ii = np.load(path_res + "h_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    h_dict[ii] = h_deg_ii

    # E1H2
    err_E1_deg_ii = np.load(path_res + "E1_err_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    err_E1_dict[ii] = err_E1_deg_ii

    err_H2_deg_ii = np.load(path_res + "H2_err_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    err_H2_dict[ii] = err_H2_deg_ii

    err_H1nor_deg_ii = np.load(path_res + "H1nor_err_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    err_H1nor_dict[ii] = err_H1nor_deg_ii

    err_E1tan_deg_ii = np.load(path_res + "E1tan_err_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    err_E1tan_dict[ii] = err_E1tan_deg_ii

    # E2H1
    err_E2_deg_ii = np.load(path_res + "E2_err_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    err_E2_dict[ii] = err_E2_deg_ii

    err_H1_deg_ii = np.load(path_res + "H1_err_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    err_H1_dict[ii] = err_H1_deg_ii

    err_E1nor_deg_ii = np.load(path_res + "E1nor_err_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    err_E1nor_dict[ii] = err_E1nor_deg_ii

    err_H1tan_deg_ii = np.load(path_res + "H1tan_err_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    err_H1tan_dict[ii] = err_H1tan_deg_ii


    # Dual field
    err_E12_deg_ii = np.load(path_res + "E12_err_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    err_E12_dict[ii] = err_E12_deg_ii

    err_H12_deg_ii = np.load(path_res + "H12_err_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    err_H12_dict[ii] = err_H12_deg_ii

    # Orders
    # E1H2
    ord_E1_deg_ii = np.load(path_res + "order_E1_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    ord_E1_dict[ii] = ord_E1_deg_ii

    ord_H2_deg_ii = np.load(path_res + "order_H1_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    ord_H2_dict[ii] = ord_H2_deg_ii

    ord_H1nor_deg_ii = np.load(path_res + "order_H1nor_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    ord_H1nor_dict[ii] = ord_H1nor_deg_ii

    ord_E1tan_deg_ii = np.load(path_res + "order_E1tan_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    ord_E1tan_dict[ii] = ord_E1tan_deg_ii

    # E2H1
    ord_E2_deg_ii = np.load(path_res + "order_E2_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    ord_E2_dict[ii] = ord_E2_deg_ii

    ord_H1_deg_ii = np.load(path_res + "order_H1_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    ord_H1_dict[ii] = ord_H1_deg_ii

    ord_E1nor_deg_ii = np.load(path_res + "order_E1nor_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    ord_E1nor_dict[ii] = ord_E1nor_deg_ii

    ord_H1tan_deg_ii = np.load(path_res + "order_H1tan_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    ord_H1tan_dict[ii] = ord_H1tan_deg_ii

    # Dual Field
    ord_E12_deg_ii = np.load(path_res + "order_E12_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    ord_E12_dict[ii] = ord_E12_deg_ii

    ord_H12_deg_ii = np.load(path_res + "order_H12_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    ord_H12_dict[ii] = ord_H12_deg_ii


# E1H2 system

plt.figure()
for ii in deg_vec:
    h = h_dict[ii]
    errL2_E1 = err_E1_dict[ii][:, 0]
    plt.plot(np.log(h), np.log(errL2_E1), '-.+', label=r'NED$^1_' + str(ii)+ '$')
    plt.plot(np.log(h), np.log(h**(ii)) + \
             + 1.1*(np.log(errL2_E1)[0] - np.log(h**(ii))[0]), '-v', label=r'$h^' + str(ii) + '$')

plt.xlabel(r'$\log(h)$')
plt.ylabel(r'$\log||E^1_h - E^1_{\mathrm{ex}}||_{L^2}$')
plt.title(r'Error $E^1_h$')

plt.legend()

if save_plots:
    plt.savefig(path_fig + "E_1" + geo_case + bc_case + ".pdf", format="pdf")


plt.figure()
for ii in deg_vec:
    h = h_dict[ii]
    errL2_H2 = err_H2_dict[ii][:, 0]
    plt.plot(np.log(h), np.log(errL2_H2), '-.+', label=r'RT$^1_' + str(ii) + '$')
    plt.plot(np.log(h), np.log(h**ii) + \
             + 1.3*(np.log(errL2_H2)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')

plt.xlabel(r'$\log(h)$')
plt.ylabel(r'$\log||H^2_h - H^2_{\mathrm{ex}}||_{L^2}$')
plt.title(r'Error $H^2_h$')

plt.legend()

if save_plots:
    plt.savefig(path_fig + "H_2" + geo_case + bc_case + ".pdf", format="pdf")

plt.figure()
for ii in deg_vec:
    h = h_dict[ii]
    errL2_H1nor = err_H1nor_dict[ii][:]
    plt.plot(np.log(h), np.log(errL2_H1nor), '-.+', label=r'NED$^1_' + str(ii) + '$')

    plt.plot(np.log(h), np.log(h**ii) + \
        + 1.5*(np.log(errL2_H1nor)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')

plt.xlabel(r'$\log(h)$')
plt.ylabel(r'$\log|||H^{1, \bm{n}}_h - P_h H^{1,\bm{n}}_{\mathrm{ex}}|||_{\mathcal{T}_h}$')
plt.title(r'Error $H^{1, \bm{n}}_h$')

plt.legend()

if save_plots:
    plt.savefig(path_fig + "H_1nor" + geo_case + bc_case + ".pdf", format="pdf")


plt.figure()
for ii in deg_vec:
    h = h_dict[ii]
    errL2_E1tan = err_E1tan_dict[ii][:]
    plt.plot(np.log(h), np.log(errL2_E1tan), '-.+', label=r'NED$^1_' + str(ii) + '$')
    plt.plot(np.log(h), np.log(h**ii) + \
             + 1.3*(np.log(errL2_E1tan)[0] - np.log(h**ii)[0]), '-v', label=r'$h^' + str(ii) + '$')

plt.xlabel(r'$\log(h)$')
plt.ylabel(r'$\log|||E^{1, \bm{t}}_h - E^{1, \bm{t}}_{\mathrm{ex}}|||_{\mathcal{T}_h}$')
plt.title(r'Error $E^{1, \bm{t}}_h$')

plt.legend()

if save_plots:
    plt.savefig(path_fig + "E_1tan" + geo_case + bc_case + ".pdf", format="pdf")

# E2H1 system
plt.figure()
for ii in deg_vec:
    h = h_dict[ii]
    errL2_E2 = err_E2_dict[ii][:, 0]
    plt.plot(np.log(h), np.log(errL2_E2), '-.+', label=r'RT$_' + str(ii) + '$')
    plt.plot(np.log(h), np.log(h ** ii) + \
             + 1.1 * (np.log(errL2_E2)[0] - np.log(h ** ii)[0]), '-v', label=r'$h^' + str(ii) + '$')
    # if ii==1:
    #     plt.plot(np.log(h), np.log(h ** ii) + \
    #              + 1.1 * (np.log(errL2_E2)[-1] - np.log(h ** ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')
    # else:
    #     plt.plot(np.log(h), np.log(h**ii) + \
    #          + 0.9*(np.log(errL2_E2)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')

plt.xlabel(r'$\log(h)$')
plt.ylabel(r'$\log||\widehat{E}^2_h - \widehat{E}^2_{\mathrm{ex}}||_{L^2}$')
plt.title(r'Error $\widehat{E}^2_h$')

plt.legend()

if save_plots:
    plt.savefig(path_fig + "E_2" + geo_case + bc_case + ".pdf", format="pdf")

plt.figure()
for ii in deg_vec:
    h = h_dict[ii]
    errL2_H1 = err_H1_dict[ii][:, 0]
    plt.plot(np.log(h), np.log(errL2_H1), '-.+', label=r'NED$^1_' + str(ii) + '$')
    plt.plot(np.log(h), np.log(h**ii) + \
             + 1.1*(np.log(errL2_H1)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')

plt.xlabel(r'$\log(h)$')
plt.ylabel(r'$\log||\widehat{H}^1_h - \widehat{H}^1_{\mathrm{ex}}||_{L^2}$')
plt.title(r'Error $\widehat{H}^1_h$')

plt.legend()

if save_plots:
    plt.savefig(path_fig + "H_1" + geo_case + bc_case + ".pdf", format="pdf")

plt.figure()
for ii in deg_vec:
    h = h_dict[ii]
    errL2_E1nor = err_E1nor_dict[ii][:]
    plt.plot(np.log(h), np.log(errL2_E1nor), '-.+', label=r'NED$^1_' + str(ii) + '$')
    plt.plot(np.log(h), np.log(h**(ii)) + \
             + 1.3*(np.log(errL2_E1nor)[0] - np.log(h**(ii))[0]), '-v', label=r'$h^' + str(ii) + '$')

plt.xlabel(r'$\log(h)$')
plt.ylabel(r'$\log|||\widehat{E}^{1, \bm{n}}_h - P_h \widehat{E}^{1, \bm{n}}_{\mathrm{ex}}|||_{\mathcal{T}_h}$')
plt.title(r'Error $\widehat{E}^{1, \bm{n}}_h$')

plt.legend()

if save_plots:
    plt.savefig(path_fig + "E_1nor" + geo_case + bc_case + ".pdf", format="pdf")


plt.figure()
for ii in deg_vec:
    h = h_dict[ii]
    errL2_H1tan = err_H1tan_dict[ii][:]
    plt.plot(np.log(h), np.log(errL2_H1tan), '-.+', label=r'NED$^1_' + str(ii) + '$')
    plt.plot(np.log(h), np.log(h**ii) + \
             + 1.3*(np.log(errL2_H1tan)[-1] - 0.95*np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')

plt.xlabel(r'$\log(h)$')
plt.ylabel(r'$\log|||\widehat{H}^{1, \bm{t}}_h - \widehat{H}^{1, \bm{t}}_{\mathrm{ex}}|||_{\mathcal{T}_h}$')
plt.title(r'Error $\widehat{H}^{1, \bm{t}}_h$')

plt.legend()

if save_plots:
    plt.savefig(path_fig + "H_1tan" + geo_case + bc_case + ".pdf", format="pdf")


# Dual Field
#

plt.figure()
for ii in deg_vec:
    h = h_dict[ii]
    errL2_E12 = err_E12_dict[ii]
    plt.plot(np.log(h), np.log(errL2_E12), '-.+', label=r'$s=' + str(ii) + '$')
    plt.plot(np.log(h), np.log(h**ii) + \
             + 1.1*(np.log(errL2_E12)[0] - np.log(h**ii)[0]), '-v', label=r'$h^' + str(ii) + '$')

plt.xlabel(r'$\log(h)$')
plt.ylabel(r'$\log||\widehat{E}^2_h - E^1_h||_{L^2}$')
plt.title(r'Error between $\widehat{E}^2$ and $E^1$')

plt.legend()

if save_plots:
    plt.savefig(path_fig + "E_12" + geo_case + bc_case + ".pdf", format="pdf")

plt.figure()
for ii in deg_vec:
    h = h_dict[ii]
    errL2_H12 = err_H12_dict[ii]
    plt.plot(np.log(h), np.log(errL2_H12), '-.+', label=r'$s=' + str(ii) + '$')
    plt.plot(np.log(h), np.log(h**ii) + \
             + 1.25*(np.log(errL2_H12)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')

plt.xlabel(r'$\log(h)$')
plt.ylabel(r'$\log||H^2_h - \widehat{H}^1_h||_{L^2}$')
plt.title(r'Error between $H^2_h$ and $\widehat{H}^1_h$')

plt.legend()

if save_plots:
    plt.savefig(path_fig + "H_12" + geo_case + bc_case + ".pdf", format="pdf")


plt.show()
