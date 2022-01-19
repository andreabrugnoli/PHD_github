import numpy as np

import matplotlib.pyplot as plt
from tools_plotting import setup
save_plots = input("Save plots? ")
path_fig = "/home/andrea/Pictures/PythonPlots/DualField_Maxwell3D/"
path_res = "results_maxwell/"
bc_case = "_EH"
geo_case = "_3D"

deg_vec = np.arange(1, 4)

h_dict = {}
err_E2_dict = {}
err_H2_dict = {}
err_E1_dict = {}
err_H1_dict = {}
err_E21_dict = {}
err_H21_dict = {}


ord_E2_dict = {}
ord_H2_dict = {}
ord_E1_dict = {}
ord_H1_dict = {}
ord_E21_dict = {}
ord_H21_dict = {}

for ii in deg_vec:
    h_deg_ii = np.load(path_res + "h_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    h_dict[ii] = h_deg_ii

    err_E2_deg_ii = np.load(path_res + "E2_err_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    err_E2_dict[ii] = err_E2_deg_ii

    err_H2_deg_ii = np.load(path_res + "H2_err_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    err_H2_dict[ii] = err_H2_deg_ii

    err_E1_deg_ii = np.load(path_res + "E1_err_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    err_E1_dict[ii] = err_E1_deg_ii

    err_H1_deg_ii = np.load(path_res + "H1_err_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    err_H1_dict[ii] = err_H1_deg_ii

    err_E21_deg_ii = np.load(path_res + "E21_err_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    err_E21_dict[ii] = err_E21_deg_ii

    err_H21_deg_ii = np.load(path_res + "H21_err_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    err_H21_dict[ii] = err_H21_deg_ii

    ord_E2_deg_ii = np.load(path_res + "order_E2_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    ord_E2_dict[ii] = ord_E2_deg_ii

    ord_H2_deg_ii = np.load(path_res + "order_H2_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    ord_H2_dict[ii] = ord_H2_deg_ii

    ord_E1_deg_ii = np.load(path_res + "order_E1_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    ord_E1_dict[ii] = ord_E1_deg_ii

    ord_H1_deg_ii = np.load(path_res + "order_H1_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    ord_H1_dict[ii] = ord_H1_deg_ii

    ord_E21_deg_ii = np.load(path_res + "order_E21_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    ord_E21_dict[ii] = ord_E21_deg_ii

    ord_H21_deg_ii = np.load(path_res + "order_H21_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    ord_H21_dict[ii] = ord_H21_deg_ii


plt.figure()
for ii in deg_vec:
    h = h_dict[ii]
    errL2_E2 = err_E2_dict[ii][:, 0]
    plt.plot(np.log(h), np.log(errL2_E2), '-.+', label=r'RT$_' + str(ii) + '$')
    plt.plot(np.log(h), np.log(h**ii) + \
             + 1.15*(np.log(errL2_E2)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')

plt.xlabel(r'$\log(h)$')
plt.ylabel(r'$\log||E^2_h - E^2_{\mathrm{ex}}||_{L^2}$')
plt.title(r'Error $E^2_h$')

plt.legend()

if save_plots:
    plt.savefig(path_fig + "E_2" + geo_case + bc_case + ".pdf", format="pdf")


plt.figure()
for ii in deg_vec:
    h = h_dict[ii]
    errL2_H2 = err_H2_dict[ii][:, 0]
    plt.plot(np.log(h), np.log(errL2_H2), '-.+', label=r'RT$_' + str(ii)+ '$')
    plt.plot(np.log(h), np.log(h**ii) + \
             + 1.15*(np.log(errL2_H2)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')

plt.xlabel(r'$\log(h)$')
plt.ylabel(r'$\log||H^2_h - H^2_{\mathrm{ex}}||_{L^2}$')
plt.title(r'Error $H^2_h$')

plt.legend()

if save_plots:
    plt.savefig(path_fig + "H_2" + geo_case + bc_case + ".pdf", format="pdf")


plt.figure()
for ii in deg_vec:
    h = h_dict[ii]
    errL2_E1 = err_E1_dict[ii][:, 0]
    plt.plot(np.log(h), np.log(errL2_E1), '-.+', label=r'NED$^1_' + str(ii) + '$')
    plt.plot(np.log(h), np.log(h**ii) + \
             + 1.1*(np.log(errL2_E1)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')

plt.xlabel(r'$\log(h)$')
plt.ylabel(r'$\log||E^1_h - E^1_{\mathrm{ex}}||_{L^2}$')
plt.title(r'Error $E^1_h$')

plt.legend()

if save_plots:
    plt.savefig(path_fig + "E_1" + geo_case + bc_case + ".pdf", format="pdf")

# plt.figure()
# for ii in deg_vec:
#     h = h_dict[ii]
#     errHcurl_E1 = err_E1_dict[ii][:, 1]
#     plt.plot(np.log(h), np.log(errHcurl_E1), '-.+', label=r'NED$_' + str(ii) + '$')
#     plt.plot(np.log(h), np.log(h**ii) + \
#              + 1.1*(np.log(errHcurl_E1)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')
#
# plt.xlabel(r'$h$')
# plt.title(r'$||E1||_{H(\mathrm{curl})}$')
# plt.legend()


plt.figure()
for ii in deg_vec:
    h = h_dict[ii]
    errL2_H1 = err_H1_dict[ii][:, 0]
    plt.plot(np.log(h), np.log(errL2_H1), '-.+', label=r'NED$^1_' + str(ii) + '$')
    plt.plot(np.log(h), np.log(h**ii) + \
             + 1.15*(np.log(errL2_H1)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')

plt.xlabel(r'$\log(h)$')
plt.ylabel(r'$\log||H^1_h - H^1_{\mathrm{ex}}||_{L^2}$')
plt.title(r'Error $H^1_h$')

plt.legend()

if save_plots:
    plt.savefig(path_fig + "H_1" + geo_case + bc_case + ".pdf", format="pdf")


# plt.figure()
# for ii in deg_vec:
#     h = h_dict[ii]
#     errHcurl_H1 = err_H1_dict[ii][:, 1]
#     plt.plot(np.log(h), np.log(errHcurl_H1), '-.+', label=r'NED$_' + str(ii) + '$')
#     plt.plot(np.log(h), np.log(h**ii) + \
#              + 1.1*(np.log(errHcurl_H1)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')
#
# plt.xlabel(r'$h$')
# plt.title(r'$||H^1||_{H(\mathrm{curl})}$')
# plt.legend()

plt.figure()
for ii in deg_vec:
    h = h_dict[ii]
    errL2_E21 = err_E21_dict[ii]
    plt.plot(np.log(h), np.log(errL2_E21), '-.+', label=r'$s=' + str(ii) + '$')
    plt.plot(np.log(h), np.log(h**ii) + \
             + 1.15*(np.log(errL2_E21)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')

plt.xlabel(r'$\log(h)$')
plt.ylabel(r'$\log||E^1_h - E^2_h||_{L^2}$')
plt.title(r'Error between $E^1_h$ and $E^2_h$')

plt.legend()

if save_plots:
    plt.savefig(path_fig + "E_21" + geo_case + bc_case + ".pdf", format="pdf")

plt.figure()
for ii in deg_vec:
    h = h_dict[ii]
    errL2_H21 = err_H21_dict[ii]
    plt.plot(np.log(h), np.log(errL2_H21), '-.+', label=r'$s=' + str(ii) + '$')
    plt.plot(np.log(h), np.log(h**ii) + \
             + 1.15*(np.log(errL2_H21)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')

plt.xlabel(r'$\log(h)$')
plt.ylabel(r'$\log||H^1_h - H^2_h||_{L^2}$')
plt.title(r'Error between $H^1_h$ and $H^2_h$')

plt.legend()

if save_plots:
    plt.savefig(path_fig + "H_21" + geo_case + bc_case + ".pdf", format="pdf")

plt.show()

for ii in range(1, 4):
    order_E1_deg_ii = np.load(path_res + "order_E1_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    print("Order E1 for deg " + str(ii))
    print(order_E1_deg_ii)

    order_H1_deg_ii = np.load(path_res + "order_H1_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    print("Order H1 for deg " + str(ii))
    print(order_H1_deg_ii)

    order_E2_deg_ii = np.load(path_res + "order_E2_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    print("Order E2 for deg " + str(ii))
    print(order_E2_deg_ii)

    order_H2_deg_ii = np.load(path_res + "order_H2_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    print("Order H2 for deg " + str(ii))
    print(order_H2_deg_ii)

    order_E21_deg_ii = np.load(path_res + "order_E21_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    print("Order E21 for deg " + str(ii))
    print(order_E21_deg_ii)

    order_H21_deg_ii = np.load(path_res + "order_H21_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    print("Order H2 for deg " + str(ii))
    print(order_H21_deg_ii)