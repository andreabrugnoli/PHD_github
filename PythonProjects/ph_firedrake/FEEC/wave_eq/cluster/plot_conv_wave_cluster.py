import numpy as np

import matplotlib.pyplot as plt
from tools_plotting import setup

path_res = "results_wave/"
bc_case = "_DN"
geo_case = "_3D"

deg_vec = np.arange(1, 4)

h_dict = {}
err_p0_dict = {}
err_u1_dict = {}
err_p3_dict = {}
err_u2_dict = {}
err_p30_dict = {}
err_u12_dict = {}


ord_p0_dict = {}
ord_u1_dict = {}
ord_p3_dict = {}
ord_u2_dict = {}
ord_p30_dict = {}
ord_u12_dict = {}

for ii in deg_vec:
    h_deg_ii = np.load(path_res + "h_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    h_dict[ii] = h_deg_ii

    err_p0_deg_ii = np.load(path_res + "p0_err_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    err_p0_dict[ii] = err_p0_deg_ii

    err_p3_deg_ii = np.load(path_res + "p3_err_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    err_p3_dict[ii] = err_p3_deg_ii

    err_u1_deg_ii = np.load(path_res + "u1_err_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    err_u1_dict[ii] = err_u1_deg_ii

    err_u2_deg_ii = np.load(path_res + "u2_err_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    err_u2_dict[ii] = err_u2_deg_ii

    err_p30_deg_ii = np.load(path_res + "p30_err_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    err_p30_dict[ii] = err_p30_deg_ii

    err_u12_deg_ii = np.load(path_res + "u12_err_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    err_u12_dict[ii] = err_u12_deg_ii

    ord_p0_deg_ii = np.load(path_res + "order_p0_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    ord_p0_dict[ii] = ord_p0_deg_ii

    ord_p3_deg_ii = np.load(path_res + "order_p3_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    ord_p3_dict[ii] = ord_p3_deg_ii

    ord_u1_deg_ii = np.load(path_res + "order_u1_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    ord_u1_dict[ii] = ord_u1_deg_ii

    ord_u2_deg_ii = np.load(path_res + "order_u2_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    ord_u2_dict[ii] = ord_u2_deg_ii

    ord_p30_deg_ii = np.load(path_res + "order_p30_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    ord_p30_dict[ii] = ord_p30_deg_ii

    ord_u12_deg_ii = np.load(path_res + "order_u12_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    ord_u12_dict[ii] = ord_u12_deg_ii


plt.figure()
for ii in deg_vec:
    h = h_dict[ii]
    errL2_p3 = err_p3_dict[ii]
    plt.plot(np.log(h), np.log(errL2_p3), '-.+', label=r'DG$_' + str(ii) + '$')
    plt.plot(np.log(h), np.log(h**ii) + \
             + 1.1*(np.log(errL2_p3)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')

plt.xlabel(r'$h$')
plt.title(r'$||p_3||_{L^2}$')
plt.legend()

plt.figure()
for ii in deg_vec:
    h = h_dict[ii]
    errL2_p0 = err_p0_dict[ii][:, 0]
    plt.plot(np.log(h), np.log(errL2_p0), '-.+', label=r'CG$_' + str(ii)+ '$')
    plt.plot(np.log(h), np.log(h**ii) + \
             + 1.1*(np.log(errL2_p0)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')

plt.xlabel(r'$h$')
plt.title(r'$||p_0||_{L^2}$')
plt.legend()

# plt.figure()
# for ii in deg_vec:
#     h = h_dict[ii]
#     errH1_p0 = err_p0_dict[ii][:, 1]
#     plt.plot(np.log(h), np.log(errH1_p0), '-.+', label=r'CG$_' + str(ii)+ '$')
#     plt.plot(np.log(h), np.log(h**ii) + \
#              + 1.1*(np.log(errH1_p0)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')
#
# plt.xlabel(r'$h$')
# plt.title(r'$||p_0||_{H^1}$')
# plt.legend()

plt.figure()
for ii in deg_vec:
    h = h_dict[ii]
    errL2_u1 = err_u1_dict[ii][:, 0]
    plt.plot(np.log(h), np.log(errL2_u1), '-.+', label=r'NED$_' + str(ii) + '$')
    plt.plot(np.log(h), np.log(h**ii) + \
             + 1.1*(np.log(errL2_u1)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')

plt.xlabel(r'$h$')
plt.title(r'$||u_1||_{L^2}$')
plt.legend()

# plt.figure()
# for ii in deg_vec:
#     h = h_dict[ii]
#     errHcurl_u1 = err_u1_dict[ii][:, 1]
#     plt.plot(np.log(h), np.log(errHcurl_u1), '-.+', label=r'NED$_' + str(ii) + '$')
#     plt.plot(np.log(h), np.log(h**ii) + \
#              + 1.1*(np.log(errHcurl_u1)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')
#
# plt.xlabel(r'$h$')
# plt.title(r'$||u_1||_{H(\mathrm{curl})}$')
# plt.legend()


plt.figure()
for ii in deg_vec:
    h = h_dict[ii]
    errL2_u2 = err_u2_dict[ii][:, 0]
    plt.plot(np.log(h), np.log(errL2_u2), '-.+', label=r'RT$_' + str(ii) + '$')
    plt.plot(np.log(h), np.log(h**ii) + \
             + 1.1*(np.log(errL2_u2)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')

plt.xlabel(r'$h$')
plt.title(r'$||u_2||_{L^2}$')
plt.legend()

# plt.figure()
# for ii in deg_vec:
#     h = h_dict[ii]
#     errHdiv_u2 = err_u1_dict[ii][:, 1]
#     plt.plot(np.log(h), np.log(errHdiv_u2), '-.+', label=r'RT$_' + str(ii) + '$')
#     plt.plot(np.log(h), np.log(h**ii) + \
#              + 1.1*(np.log(errHdiv_u2)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')
#
# plt.xlabel(r'$h$')
# plt.title(r'$||u_2||_{H(\mathrm{div})}$')
# plt.legend()

plt.figure()
for ii in deg_vec:
    h = h_dict[ii]
    errL2_u12 = err_u12_dict[ii]
    plt.plot(np.log(h), np.log(errL2_u12), '-.+', label=r'NED$_' + str(ii) + '$-RT$_' + str(ii) + '$')
    plt.plot(np.log(h), np.log(h**ii) + \
             + 1.1*(np.log(errL2_u12)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')

plt.xlabel(r'$h$')
plt.title(r'$||u_1 - u_2||_{L^2}$')
plt.legend()

plt.figure()
for ii in deg_vec:
    h = h_dict[ii]
    errL2_p30 = err_p30_dict[ii]
    plt.plot(np.log(h), np.log(errL2_p30), '-.+', label=r'DG$_' + str(ii-1) + '$-CG$_' + str(ii) + '$')
    plt.plot(np.log(h), np.log(h**ii) + \
             + 1.1*(np.log(errL2_p30)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')

plt.xlabel(r'$h$')
plt.title(r'$||p_3 - p_0||_{L^2}$')
plt.legend()

plt.show()

order_p0_deg1 = np.zeros((len(h_dict[1]) - 1, 2))
order_p0_deg2 = np.zeros((len(h_dict[2]) - 1, 2))
order_p0_deg3 = np.zeros((len(h_dict[3]) - 1, 2))

for i in range(1,len(h_dict[1])):

    order_p0_deg1[i - 1, 0] = np.log(err_p0_dict[1][i, 0] / err_p0_dict[1][i - 1, 0]) / np.log(h_dict[1][i] / h_dict[1][i - 1])
    order_p0_deg1[i - 1, 1] = np.log(err_p0_dict[1][i, 1] / err_p0_dict[1][i - 1, 1]) / np.log(h_dict[1][i] / h_dict[1][i - 1])

for i in range(1, len(h_dict[2])):

    order_p0_deg2[i - 1, 0] = np.log(err_p0_dict[2][i, 0] / err_p0_dict[2][i - 1, 0]) / np.log(
        h_dict[2][i] / h_dict[2][i - 1])
    order_p0_deg2[i - 1, 1] = np.log(err_p0_dict[2][i, 1] / err_p0_dict[2][i - 1, 1]) / np.log(
        h_dict[2][i] / h_dict[2][i - 1])

for i in range(1, len(h_dict[3])):

    order_p0_deg3[i - 1, 0] = np.log(err_p0_dict[3][i, 0] / err_p0_dict[3][i - 1, 0]) / np.log(
        h_dict[3][i] / h_dict[3][i - 1])
    order_p0_deg3[i - 1, 1] = np.log(err_p0_dict[3][i, 1] / err_p0_dict[3][i - 1, 1]) / np.log(
        h_dict[3][i] / h_dict[3][i - 1])

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
