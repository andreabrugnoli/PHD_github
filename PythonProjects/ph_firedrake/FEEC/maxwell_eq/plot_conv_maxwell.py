import numpy as np

import matplotlib.pyplot as plt
from tools_plotting import setup
save_plots = True
path_fig = "/home/andrea/Pictures/PythonPlots/DualField_Maxwell3D/"
path_res = "results_maxwell/"
bc_case = "_EH"
geo_case = "_3D"

deg_vec = [1] # np.arange(1, 4)

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
             + 1.1*(np.log(errL2_E2)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')

plt.xlabel(r'$\log(h)$')
plt.ylabel(r'$\log||E^2 - E^2_{\mathrm{ex}}||_{L^2}$')
plt.title(r'Error $E^2$')

plt.legend()

if save_plots:
    plt.savefig(path_fig + "E_2" + geo_case + bc_case + ".eps", format="eps")


plt.figure()
for ii in deg_vec:
    h = h_dict[ii]
    errL2_H2 = err_H2_dict[ii][:, 0]
    plt.plot(np.log(h), np.log(errL2_H2), '-.+', label=r'RT$_' + str(ii)+ '$')
    plt.plot(np.log(h), np.log(h**ii) + \
             + 1.1*(np.log(errL2_H2)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')

plt.xlabel(r'$\log(h)$')
plt.ylabel(r'$\log||H^2 - H^2_{\mathrm{ex}}||_{L^2}$')
plt.title(r'Error $H^2$')

plt.legend()

if save_plots:
    plt.savefig(path_fig + "H_2" + geo_case + bc_case + ".eps", format="eps")


plt.figure()
for ii in deg_vec:
    h = h_dict[ii]
    errL2_E1 = err_E1_dict[ii][:, 0]
    plt.plot(np.log(h), np.log(errL2_E1), '-.+', label=r'NED$^1_' + str(ii) + '$')
    plt.plot(np.log(h), np.log(h**ii) + \
             + 1.1*(np.log(errL2_E1)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')

plt.xlabel(r'$\log(h)$')
plt.ylabel(r'$\log||E^1 - E^1_{\mathrm{ex}}||_{L^2}$')
plt.title(r'Error $E^1$')

plt.legend()

if save_plots:
    plt.savefig(path_fig + "E_1" + geo_case + bc_case + ".eps", format="eps")

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
             + 1.1*(np.log(errL2_H1)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')

plt.xlabel(r'$\log(h)$')
plt.ylabel(r'$\log||H^1 - H^1_{\mathrm{ex}}||_{L^2}$')
plt.title(r'Error $H^1$')

plt.legend()

if save_plots:
    plt.savefig(path_fig + "H_1" + geo_case + bc_case + ".eps", format="eps")


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
    plt.plot(np.log(h), np.log(errL2_E21), '-.+', label=r'NED$^1_' + str(ii) + '$-RT$_' + str(ii) + '$')
    plt.plot(np.log(h), np.log(h**ii) + \
             + 1.1*(np.log(errL2_E21)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')

plt.xlabel(r'$\log(h)$')
plt.ylabel(r'$\log||E^1 - E^2||_{L^2}$')
plt.title(r'Error between $E^1$ and $E^2$')

plt.legend()

if save_plots:
    plt.savefig(path_fig + "E_21" + geo_case + bc_case + ".eps", format="eps")

plt.figure()
for ii in deg_vec:
    h = h_dict[ii]
    errL2_H21 = err_H21_dict[ii]
    plt.plot(np.log(h), np.log(errL2_H21), '-.+', label=r'NED$^1_' + str(ii) + '$-RT$_' + str(ii) + '$')
    plt.plot(np.log(h), np.log(h**ii) + \
             + 1.1*(np.log(errL2_H21)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')

plt.xlabel(r'$\log(h)$')
plt.ylabel(r'$\log||H^1 - H^2||_{L^2}$')
plt.title(r'Error between $H^1$ and $H^2$')

plt.legend()

if save_plots:
    plt.savefig(path_fig + "H_21" + geo_case + bc_case + ".eps", format="eps")

plt.show()

# order_E2_deg1 = np.zeros((len(h_dict[1]) - 1, 2))
# order_E2_deg2 = np.zeros((len(h_dict[2]) - 1, 2))
# order_E2_deg3 = np.zeros((len(h_dict[3]) - 1, 2))
#
# for i in range(1,len(h_dict[1])):
#
#     order_E2_deg1[i - 1, 0] = np.log(err_E2_dict[1][i, 0] / err_E2_dict[1][i - 1, 0]) / np.log(h_dict[1][i] / h_dict[1][i - 1])
#     order_E2_deg1[i - 1, 1] = np.log(err_E2_dict[1][i, 1] / err_E2_dict[1][i - 1, 1]) / np.log(h_dict[1][i] / h_dict[1][i - 1])
#

#
# print("Estimated order of convergence for p_3: ")
# for ii in deg_vec:
#     print("degree: " + str(ii))
#     print(ord_p3_dict[ii])
#
# print("Estimated order of convergence for p_0: ")
# for ii in deg_vec:
#     print("degree: " + str(ii))
#     print(ord_p0_dict[ii])
#
# print("Estimated order of convergence for u_1: ")
# for ii in deg_vec:
#     print("degree: " + str(ii))
#     print(ord_u1_dict[ii])
#
# print("Estimated order of convergence for u_2: ")
# for ii in deg_vec:
#     print("degree: " + str(ii))
#     print(ord_u2_dict[ii])
#
# print("Estimated order of convergence for p_0 - p_3: ")
# for ii in deg_vec:
#     print("degree: " + str(ii))
#     print(ord_p30_dict[ii])
#
# print("Estimated order of convergence for u_1 - u_2: ")
# for ii in deg_vec:
#     print("degree: " + str(ii))
#     print(ord_u12_dict[ii])
#
#
