import numpy as np

import matplotlib.pyplot as plt
from tools_plotting import setup
save_plots = input("Save plots? ")
path_fig = "/home/andrea/Pictures/PythonPlots/Hybridization_wave/"
path_res = "results_hybrid/"
bc_case = "_DN"
geo_case = "_3D"

deg_vec = np.arange(1, 4)

h_dict = {}

# 01
err_p0_dict = {}
err_u1_dict = {}
err_u0nor_dict = {}
err_p0tan_dict = {}
# 32
err_p3_dict = {}
err_u2_dict = {}
err_u2tan_dict = {}
err_p2nor_dict = {}

# Postprocessing
err_p0_pp_dict = {}
err_u1_pp_dict = {}
err_p3_pp_dict = {}
err_u2_pp_dict = {}

# Dual Field
err_p30_dict = {}
err_u12_dict = {}

err_p30_pp_dict = {}
err_u12_pp_dict = {}

# Orders
# 01
ord_p0_dict = {}
ord_u1_dict = {}
ord_u0nor_dict = {}
ord_p0tan_dict = {}
# 32
ord_p3_dict = {}
ord_u2_dict = {}
ord_u2tan_dict = {}
ord_p2nor_dict = {}
# Orders post-processed
ord_p0_pp_dict = {}
ord_u1_pp_dict = {}
ord_p3_pp_dict = {}
ord_u2_pp_dict = {}

# Dual field
ord_p30_dict = {}
ord_u12_dict = {}
ord_p30_pp_dict = {}
ord_u12_pp_dict = {}

for ii in deg_vec:
    h_deg_ii = np.load(path_res + "h_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    h_dict[ii] = h_deg_ii

    # 01
    err_p0_deg_ii = np.load(path_res + "p0_err_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    err_p0_dict[ii] = err_p0_deg_ii

    err_u1_deg_ii = np.load(path_res + "u1_err_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    err_u1_dict[ii] = err_u1_deg_ii

    err_u0nor_deg_ii = np.load(path_res + "u0nor_err_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    err_u0nor_dict[ii] = err_u0nor_deg_ii

    err_p0tan_deg_ii = np.load(path_res + "p0tan_err_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    err_p0tan_dict[ii] = err_p0tan_deg_ii

    # 32
    err_p3_deg_ii = np.load(path_res + "p3_err_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    err_p3_dict[ii] = err_p3_deg_ii

    err_u2_deg_ii = np.load(path_res + "u2_err_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    err_u2_dict[ii] = err_u2_deg_ii

    err_p2nor_deg_ii = np.load(path_res + "p2nor_err_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    err_p2nor_dict[ii] = err_p2nor_deg_ii

    err_u2tan_deg_ii = np.load(path_res + "u2tan_err_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    err_u2tan_dict[ii] = err_u2tan_deg_ii

    # Post-processed
    err_p0_pp_deg_ii = np.load(path_res + "p0_pp_err_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    err_p0_pp_dict[ii] = err_p0_pp_deg_ii

    err_u1_pp_deg_ii = np.load(path_res + "u1_pp_err_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    err_u1_pp_dict[ii] = err_u1_pp_deg_ii

    err_p3_pp_deg_ii = np.load(path_res + "p3_pp_err_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    err_p3_pp_dict[ii] = err_p3_pp_deg_ii

    err_u2_pp_deg_ii = np.load(path_res + "u2_pp_err_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    err_u2_pp_dict[ii] = err_u2_pp_deg_ii

    # Dual field
    err_p30_deg_ii = np.load(path_res + "p30_err_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    err_p30_dict[ii] = err_p30_deg_ii

    err_u12_deg_ii = np.load(path_res + "u12_err_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    err_u12_dict[ii] = err_u12_deg_ii

    err_p30_pp_deg_ii = np.load(path_res + "p30_pp_err_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    err_p30_pp_dict[ii] = err_p30_deg_ii

    err_u12_pp_deg_ii = np.load(path_res + "u12_pp_err_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    err_u12_pp_dict[ii] = err_u12_pp_deg_ii


    # Orders
    # 01
    ord_p0_deg_ii = np.load(path_res + "order_p0_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    ord_p0_dict[ii] = ord_p0_deg_ii

    ord_u1_deg_ii = np.load(path_res + "order_u1_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    ord_u1_dict[ii] = ord_u1_deg_ii

    ord_u0nor_deg_ii = np.load(path_res + "order_u0nor_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    ord_u0nor_dict[ii] = ord_u0nor_deg_ii

    ord_p0tan_deg_ii = np.load(path_res + "order_p0tan_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    ord_p0tan_dict[ii] = ord_p0tan_deg_ii
    # 32
    ord_p3_deg_ii = np.load(path_res + "order_p3_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    ord_p3_dict[ii] = ord_p3_deg_ii

    ord_u2_deg_ii = np.load(path_res + "order_u2_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    ord_u2_dict[ii] = ord_u2_deg_ii

    ord_p2nor_deg_ii = np.load(path_res + "order_p2nor_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    ord_p2nor_dict[ii] = ord_p2nor_deg_ii

    ord_u2tan_deg_ii = np.load(path_res + "order_u2tan_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    ord_u2tan_dict[ii] = ord_u2tan_deg_ii

    # Postprocessing
    ord_p0_pp_deg_ii = np.load(path_res + "order_p0_pp_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    ord_p0_pp_dict[ii] = ord_p0_pp_deg_ii

    ord_u1_pp_deg_ii = np.load(path_res + "order_u1_pp_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    ord_u1_pp_dict[ii] = ord_u1_pp_deg_ii

    ord_p3_pp_deg_ii = np.load(path_res + "order_p3_pp_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    ord_p3_pp_dict[ii] = ord_p3_pp_deg_ii

    ord_u2_pp_deg_ii = np.load(path_res + "order_u2_pp_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    ord_u2_pp_dict[ii] = ord_u2_pp_deg_ii

    # Dual Field
    ord_p30_deg_ii = np.load(path_res + "order_p30_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    ord_p30_dict[ii] = ord_p30_deg_ii

    ord_u12_deg_ii = np.load(path_res + "order_u12_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    ord_u12_dict[ii] = ord_u12_deg_ii

    ord_p30_pp_deg_ii = np.load(path_res + "order_p30_pp_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    ord_p30_pp_dict[ii] = ord_p30_pp_deg_ii

    ord_u12_pp_deg_ii = np.load(path_res + "order_u12_pp_deg" + str(ii) + bc_case + str(geo_case) + ".npy")
    ord_u12_pp_dict[ii] = ord_u12_pp_deg_ii


# 01 system

plt.figure()
for ii in deg_vec:
    h = h_dict[ii]
    errL2_p0 = err_p0_dict[ii][:, 0]
    plt.plot(np.log(h), np.log(errL2_p0), '-.+', label=r'CG$_' + str(ii)+ '$')
    plt.plot(np.log(h), np.log(h**(ii)) + \
             + 1.1*(np.log(errL2_p0)[-1] - np.log(h**(ii))[-1]), '-v', label=r'$h^' + str(ii) + '$')

plt.xlabel(r'$\log(h)$')
plt.ylabel(r'$\log||p^0_h - p^0_{\mathrm{ex}}||_{L^2}$')
plt.title(r'Error $p^0_h$')

plt.legend()

if save_plots:
    plt.savefig(path_fig + "p_0" + geo_case + bc_case + ".pdf", format="pdf")

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
    plt.plot(np.log(h), np.log(errL2_u1), '-.+', label=r'NED$^1_' + str(ii) + '$')
    plt.plot(np.log(h), np.log(h**ii) + \
             + 1.5*(np.log(errL2_u1)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')

plt.xlabel(r'$\log(h)$')
plt.ylabel(r'$\log||u^1_h - u^1_{\mathrm{ex}}||_{L^2}$')
plt.title(r'Error $u^1_h$')

plt.legend()

if save_plots:
    plt.savefig(path_fig + "u_1" + geo_case + bc_case + ".pdf", format="pdf")

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
    errL2_u0nor = err_u0nor_dict[ii][:]
    plt.plot(np.log(h), np.log(errL2_u0nor), '-.+', label=r'CG$_' + str(ii) + '$')
    if ii==1 or ii==2:
        plt.plot(np.log(h), np.log(h ** ii) + \
                 + 1.2 * (np.log(errL2_u0nor)[-1] - 0.9*np.log(h ** ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')
    else:
        plt.plot(np.log(h), np.log(h**ii) + \
             + 1.2*(np.log(errL2_u0nor)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')

plt.xlabel(r'$\log(h)$')
plt.ylabel(r'$\log|||u^{0, \bm{n}}_h - P_h u^{0,\bm{n}}_{\mathrm{ex}}|||_{\mathcal{T}_h}$')
plt.title(r'Error $u^{0, \bm{n}}_h$')

plt.legend()

if save_plots:
    plt.savefig(path_fig + "u_0nor" + geo_case + bc_case + ".pdf", format="pdf")


plt.figure()
for ii in deg_vec:
    h = h_dict[ii]
    errL2_p0tan = err_p0tan_dict[ii][:]
    plt.plot(np.log(h), np.log(errL2_p0tan), '-.+', label=r'CG$_' + str(ii) + '$')
    plt.plot(np.log(h), np.log(h**ii) + \
             + 1.5*(np.log(errL2_p0tan)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')

plt.xlabel(r'$\log(h)$')
plt.ylabel(r'$\log|||p^{0, \bm{t}}_h - p^{0, \bm{t}}_{\mathrm{ex}}|||_{\mathcal{T}_h}$')
plt.title(r'Error $p^{0, \bm{t}}_h$')

plt.legend()

if save_plots:
    plt.savefig(path_fig + "p_0tan" + geo_case + bc_case + ".pdf", format="pdf")

# 32 system
plt.figure()
for ii in deg_vec:
    h = h_dict[ii]
    errL2_p3 = err_p3_dict[ii]
    plt.plot(np.log(h), np.log(errL2_p3), '-.+', label=r'DG$_' + str(ii) + '$')
    plt.plot(np.log(h), np.log(h**ii) + \
             + 1.5*(np.log(errL2_p3)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')

plt.xlabel(r'$\log(h)$')
plt.ylabel(r'$\log||\widehat{p}^3_h - \widehat{p}^3_{\mathrm{ex}}||_{L^2}$')
plt.title(r'Error $\widehat{p}^3_h$')

plt.legend()

if save_plots:
    plt.savefig(path_fig + "p_3" + geo_case + bc_case + ".pdf", format="pdf")



plt.figure()
for ii in deg_vec:
    h = h_dict[ii]
    errL2_u2 = err_u2_dict[ii][:, 0]
    plt.plot(np.log(h), np.log(errL2_u2), '-.+', label=r'RT$_' + str(ii) + '$')
    plt.plot(np.log(h), np.log(h**ii) + \
             + 1.2*(np.log(errL2_u2)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')

plt.xlabel(r'$\log(h)$')
plt.ylabel(r'$\log||\widehat{u}^2_h - \widehat{u}^2_{\mathrm{ex}}||_{L^2}$')
plt.title(r'Error $\widehat{u}^2_h$')

plt.legend()

if save_plots:
    plt.savefig(path_fig + "u_2" + geo_case + bc_case + ".pdf", format="pdf")

# plt.figure()
# for ii in deg_vec:
#     h = h_dict[ii]
#     errHdiv_u2 = err_u2_dict[ii][:, 1]
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
    errL2_p2nor = err_p2nor_dict[ii][:]
    plt.plot(np.log(h), np.log(errL2_p2nor), '-.+', label=r'RT$_' + str(ii) + '$')
    if ii==1:
        plt.plot(np.log(h), np.log(h ** (ii+1)) + \
                 + 2 * (np.log(errL2_p2nor)[-1] - np.log(h ** (ii+1))[-1]), '-v', label=r'$h^' + str(ii+1) + '$')
    elif ii==2:
        pass
    else:
        plt.plot(np.log(h), np.log(h**(ii)) + \
             + 1.5*(np.log(errL2_p2nor)[-1] - np.log(h**(ii))[-1]), '-v', label=r'$h^' + str(ii) + '$')

plt.xlabel(r'$\log(h)$')
plt.ylabel(r'$\log|||\widehat{p}^{2, \bm{n}}_h - P_h \widehat{p}^{2, \bm{n}}_{\mathrm{ex}}|||_{\mathcal{T}_h}$')
plt.title(r'Error $\widehat{p}^{2, \bm{n}}_h$')

plt.legend()

if save_plots:
    plt.savefig(path_fig + "p_2nor" + geo_case + bc_case + ".pdf", format="pdf")


plt.figure()
for ii in deg_vec:
    h = h_dict[ii]
    errL2_u2tan = err_u2tan_dict[ii][:]
    plt.plot(np.log(h), np.log(errL2_u2tan), '-.+', label=r'RT$_' + str(ii) + '$')
    plt.plot(np.log(h), np.log(h**ii) + \
             + 1.5*(np.log(errL2_u2tan)[-1] - 0.9*np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')

plt.xlabel(r'$\log(h)$')
plt.ylabel(r'$\log|||\widehat{u}^{2, \bm{t}}_h - \widehat{u}^{2, \bm{t}}_{\mathrm{ex}}|||_{\mathcal{T}_h}$')
plt.title(r'Error $\widehat{u}^{2, \bm{t}}_h$')

plt.legend()

if save_plots:
    plt.savefig(path_fig + "u_2tan" + geo_case + bc_case + ".pdf", format="pdf")
#
# Post-processing
#
# plt.figure()
# for ii in deg_vec:
#     h = h_dict[ii]
#     errL2_p0_pp = err_p0_pp_dict[ii][:, 0]
#     plt.plot(np.log(h), np.log(errL2_p0_pp), '-.+', label=r'CG$_' + str(ii)+ '$')
#     plt.plot(np.log(h), np.log(h**ii) + \
#              + 1.1*(np.log(errL2_p0_pp)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')
#
# plt.xlabel(r'$\log(h)$')
# plt.ylabel(r'$\log||p^{*0}_h - p^0_{\mathrm{ex}}||_{L^2}$')
# plt.title(r'Error $p^{*0}_h$')
#
# plt.legend()
#
# if save_plots:
#     plt.savefig(path_fig + "p_0_pp" + geo_case + bc_case + ".pdf", format="pdf")
#
#
#
# plt.figure()
# for ii in deg_vec:
#     h = h_dict[ii]
#     errH1_p0_pp = err_p0_pp_dict[ii][:, 1]
#     plt.plot(np.log(h), np.log(errH1_p0_pp), '-.+', label=r'CG$_' + str(ii)+ '$')
#     plt.plot(np.log(h), np.log(h**(ii-1)) + \
#          + 1.1*(np.log(errH1_p0_pp)[-1] - np.log(h**(ii-1))[-1]), '-v', label=r'$h^' + str(ii-1) + '$')
#
# plt.xlabel(r'$\log(h)$')
# plt.ylabel(r'$\log||\mathrm{d}(p^{*0}_h - p^0_{\mathrm{ex}})||_{L^2}$')
# plt.title(r'Error $\mathrm{d}p^{*0}_h$')
#
# plt.legend()
#
# if save_plots:
#     plt.savefig(path_fig + "grad_p_0_pp" + geo_case + bc_case + ".pdf", format="pdf")
#
#
# plt.figure()
# for ii in deg_vec:
#     h = h_dict[ii]
#     errL2_p3_pp = err_p3_pp_dict[ii][:]
#     plt.plot(np.log(h), np.log(errL2_p3_pp), '-.+', label=r'DG$_' + str(ii)+ '$')
#     plt.plot(np.log(h), np.log(h**ii) + \
#              + 1.1*(np.log(errL2_p3_pp)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')
#
# plt.xlabel(r'$\log(h)$')
# plt.ylabel(r'$\log||p^{*3}_h - p^3_{\mathrm{ex}}||_{L^2}$')
# plt.title(r'Error $p^{*3}_h$')
#
# plt.legend()
#
# if save_plots:
#     plt.savefig(path_fig + "p_3_pp" + geo_case + bc_case + ".pdf", format="pdf")
#
#
# plt.figure()
# for ii in deg_vec:
#     h = h_dict[ii]
#     errL2_u1_pp = err_u1_pp_dict[ii][:, 0]
#     plt.plot(np.log(h), np.log(errL2_u1_pp), '-.+', label=r'NED$^1_' + str(ii) + '$')
#     plt.plot(np.log(h), np.log(h**ii) + \
#              + 1.1*(np.log(errL2_u1_pp)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')
#
# plt.xlabel(r'$\log(h)$')
# plt.ylabel(r'$\log||u^{*1}_h - u^1_{\mathrm{ex}}||_{L^2}$')
# plt.title(r'Error $u^{*1}_h$')
#
# plt.legend()
#
# if save_plots:
#     plt.savefig(path_fig + "u_1_pp" + geo_case + bc_case + ".pdf", format="pdf")
#
#
# plt.figure()
# for ii in deg_vec:
#     h = h_dict[ii]
#     errHcurl_u1_pp = err_u1_pp_dict[ii][:, 1]
#     plt.plot(np.log(h), np.log(errHcurl_u1_pp), '-.+', label=r'NED$^1_' + str(ii) + '$')
#     if ii>1:
#         plt.plot(np.log(h), np.log(h**(ii-1)) + \
#              + 1.1*(np.log(errHcurl_u1_pp)[-1] - np.log(h**(ii-1))[-1]), '-v', label=r'$h^' + str(ii-1) + '$')
#
# plt.xlabel(r'$\log(h)$')
# plt.ylabel(r'$\log||\mathrm{d}(u^{*1}_h - u^1_{\mathrm{ex}})||_{L^2}$')
# plt.title(r'Error $\mathrm{d} u^{*1}_h$')
#
# plt.legend()
#
# if save_plots:
#     plt.savefig(path_fig + "curl_u_1_pp" + geo_case + bc_case + ".pdf", format="pdf")
#
#
#
#
# plt.figure()
# for ii in deg_vec:
#     h = h_dict[ii]
#     errL2_u2_pp = err_u2_pp_dict[ii][:, 0]
#     plt.plot(np.log(h), np.log(errL2_u2_pp), '-.+', label=r'RT$^1_' + str(ii) + '$')
#     plt.plot(np.log(h), np.log(h**ii) + \
#              + 1.1*(np.log(errL2_u2_pp)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')
#
# plt.xlabel(r'$\log(h)$')
# plt.ylabel(r'$\log||u^{*2}_h - u^2_{\mathrm{ex}}||_{L^2}$')
# plt.title(r'Error $u^{*2}_h$')
#
# plt.legend()
#
# if save_plots:
#     plt.savefig(path_fig + "u_2_pp" + geo_case + bc_case + ".pdf", format="pdf")
#
#
# plt.figure()
# for ii in deg_vec:
#     h = h_dict[ii]
#     errHdiv_u2_pp = err_u2_pp_dict[ii][:, 1]
#     plt.plot(np.log(h), np.log(errHdiv_u2_pp), '-.+', label=r'RT$^1_' + str(ii) + '$')
#     plt.plot(np.log(h), np.log(h**(ii-1)) + \
#          + 1.1*(np.log(errHdiv_u2_pp)[-1] - np.log(h**(ii-1))[-1]), '-v', label=r'$h^' + str((ii-1)) + '$')
#
# plt.xlabel(r'$\log(h)$')
# plt.ylabel(r'$\log||\mathrm{d}(u^{*2}_h - u^2_{\mathrm{ex}})||_{L^2}$')
# plt.title(r'Error $\mathrm{d} u^{*2}_h$')
#
# plt.legend()
#
# if save_plots:
#     plt.savefig(path_fig + "div_u_2_pp" + geo_case + bc_case + ".pdf", format="pdf")


# Dual Field
#

plt.figure()
for ii in deg_vec:
    h = h_dict[ii]
    errL2_p30 = err_p30_dict[ii]
    plt.plot(np.log(h), np.log(errL2_p30), '-.+', label=r'$s=' + str(ii) + '$')
    plt.plot(np.log(h), np.log(h**ii) + \
             + 1.5*(np.log(errL2_p30)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')

plt.xlabel(r'$\log(h)$')
plt.ylabel(r'$\log||\widehat{p}^3_h - p^0_h||_{L^2}$')
plt.title(r'Error between $\widehat{p}^3$ and $p^0$')

plt.legend()

if save_plots:
    plt.savefig(path_fig + "p_30" + geo_case + bc_case + ".pdf", format="pdf")

plt.figure()
for ii in deg_vec:
    h = h_dict[ii]
    errL2_u12 = err_u12_dict[ii]
    plt.plot(np.log(h), np.log(errL2_u12), '-.+', label=r'$s=' + str(ii) + '$')
    plt.plot(np.log(h), np.log(h**ii) + \
             + 1.5*(np.log(errL2_u12)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')

plt.xlabel(r'$\log(h)$')
plt.ylabel(r'$\log||u^1_h - \widehat{u}^2_h||_{L^2}$')
plt.title(r'Error between $u^1_h$ and $\widehat{u}^2_h$')

plt.legend()

if save_plots:
    plt.savefig(path_fig + "u_12" + geo_case + bc_case + ".pdf", format="pdf")


# plt.figure()
# for ii in deg_vec:
#     h = h_dict[ii]
#     errL2_p30_pp = err_p30_pp_dict[ii]
#     plt.plot(np.log(h), np.log(errL2_p30_pp), '-.+', label=r'$s=' + str(ii) + '$')
#     plt.plot(np.log(h), np.log(h**ii) + \
#              + 1.1*(np.log(errL2_p30_pp)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')
#
# plt.xlabel(r'$\log(h)$')
# plt.ylabel(r'$\log||p^{*3}_h - p^{*0}_h||_{L^2}$')
# plt.title(r'Error between $p^{*3}$ and $p^{*0}$')
#
# plt.legend()
#
# if save_plots:
#     plt.savefig(path_fig + "p_30_pp" + geo_case + bc_case + ".pdf", format="pdf")
#
#
# plt.figure()
# for ii in deg_vec:
#     h = h_dict[ii]
#     errL2_u12_pp = err_u12_pp_dict[ii]
#     plt.plot(np.log(h), np.log(errL2_u12_pp), '-.+', label=r'$s=' + str(ii) + '$')
#     plt.plot(np.log(h), np.log(h**ii) + \
#              + 1.15*(np.log(errL2_u12_pp)[-1] - np.log(h**ii)[-1]), '-v', label=r'$h^' + str(ii) + '$')
#
# plt.xlabel(r'$\log(h)$')
# plt.ylabel(r'$\log||u^{*1}_h - u^{*2}_h||_{L^2}$')
# plt.title(r'Error between $u^{*1}_h$ and $u^{*2}_h$')
#
# plt.legend()
#
# if save_plots:
#     plt.savefig(path_fig + "u_12_pp" + geo_case + bc_case + ".pdf", format="pdf")

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
