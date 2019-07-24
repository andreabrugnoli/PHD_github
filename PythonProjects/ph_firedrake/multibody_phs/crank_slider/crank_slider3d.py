import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
fntsize = 15

import numpy as np
import quaternion
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from modules_phdae.classes_phsystem import SysPhdaeRig
from system_components.beams import SpatialBeam, draw_deformation3D
from math import pi

from assimulo.solvers import IDA
from assimulo.implicit_ode import Implicit_Problem

L_crank = 0.15
L_coupler = 0.3
d = 0.006

ecc = 0.1
rho_coupler = 7.87 * 10 ** 3
E_coupler = 200 * 10**9
A_coupler = pi * d**2 / 4
I_coupler = pi * d**4 / 64
Jxx_coupler = 2 * I_coupler * rho_coupler
n_elem = 2

omega_cr = 150

nr_coupler = 6
nr_slider = 3
nr_tot = nr_coupler + nr_slider


coupler = SpatialBeam(n_elem, L_coupler, rho_coupler, A_coupler, E_coupler, I_coupler, Jxx_coupler)

mass_coupler = rho_coupler * A_coupler * L_coupler
mass_slider = 0.5 * mass_coupler
M_slider = mass_slider * np.eye(nr_slider)

J_slider = np.zeros((nr_slider, nr_slider))
B_slider = np.eye(nr_slider)

slider = SysPhdaeRig(nr_slider, 0, nr_slider, 0, 0, E=M_slider, J=J_slider, B=B_slider)

sys = SysPhdaeRig.transformer_ordered(coupler, slider, [6, 7, 8], [0, 1, 2], np.eye(nr_slider))

# plt.spy(sys.E); plt.show()

# M_sys = sys.E
# J_sys = sys.J
# G_coupler = sys.B[:, [0, 1, 2]]
#
# n_sys = sys.n
# n_e = sys.n_e
# n_p = sys.n_p
# n_pu = int(n_p/5)
# n_pv = 2*n_pu
# n_pw = 2*n_pu
#
# n_tot = n_sys + 9  # 5 lambda (3 coupler, 2 slider) and quaternions
# order = []
#
# dx = L_coupler/n_elem
#
#
# def dae_closed_phs(t, y, yd):
#
#     yd_sys = yd[:n_sys]
#     dquat_cl = yd[-4:]
#
#     y_sys = y[:n_sys]
#     lmd_cl = y[n_sys:n_sys + 3]
#     lmd_mass = y[n_sys + 3: n_sys + 5]
#
#     quat_cl = y[-4:]
#     omega_cl = y[3:6]
#
#     p_coupler = M_sys[:3, :n_e] @ y_sys[:n_e]
#     pi_coupler = M_sys[3:6, :n_e] @ y_sys[:n_e]
#
#     p_slider = M_sys[nr_coupler:nr_tot, :n_e] @ y_sys[:n_e]
#
#     p_v = M_sys[nr_tot+n_pu:nr_tot+n_pu+n_pv, :n_e] @ y_sys[:n_e]
#     p_v[1::2] = 0
#     p_vdis = np.array([p_v[i] for i in range(len(p_v)) if i % 2 == 0])
#
#     p_w = M_sys[nr_tot+n_pu+n_pv:nr_tot + n_p, :n_e] @ y_sys[:n_e]
#     p_w[1::2] = 0
#     p_wdis = np.array([p_w[i] for i in range(len(p_w)) if i % 2 == 0])
#
#     p_udis = M_sys[nr_tot:nr_tot + n_pu, :n_e] @ y_sys[:n_e]
#     p_u = np.zeros_like(p_w)
#     p_u[::2] = p_udis
#
#     J_sys[:3, 3:6] = skew(p_coupler)
#     J_sys[3:6, :3] = skew(p_coupler)
#     J_sys[3:6, 3:6] = skew(pi_coupler)
#
#     J_sys[nr_coupler:nr_coupler+3, 3:6] = p_slider
#
#     alflex_cross = skew_flex(p_u, p_v, p_w, p_udis, p_vdis, p_wdis)
#     J_sys[nr_tot:nr_tot+n_p, 3:6] = alflex_cross
#     J_sys[3:6, nr_tot:nr_tot+n_p] = -alflex_cross.T
#
#     Rot_cl = quaternion.as_rotation_matrix(np.quaternion(quat_cl))
#
#     G_slider = np.zeros((n_sys, 2))
#     G_slider[nr_coupler:nr_tot, :] = Rot_cl[[1, 2], :].T
#
#     vC_cr = omega_cr * L_crank * np.array([0, -np.cos(omega_cr * t), -np.sin(omega_cr * t)])
#
#     res_sys = M_sys @ yd_sys - J_sys @ y_sys - G_coupler @ lmd_cl - G_slider * lmd_mass
#     res_cl = - G_coupler.T @ y_sys + Rot_cl.T @ vC_cr
#     res_mass = G_slider.T @ y_sys
#
#     Omegacl_mat = np.array([[0, -omega_cl[0], -omega_cl[1], -omega_cl[2]],
#                             [omega_cl[0],   0, omega_cl[2], -omega_cl[1]],
#                             [omega_cl[1],   -omega_cl[2], 0, omega_cl[0]],
#                             [omega_cl[2],  omega_cl[1], -omega_cl[0], 0]])
#
#     res_th = dquat_cl - 0.5 * Omegacl_mat @ quat_cl
#
#     res = np.concatenate((res_sys, res_cl, res_mass, res_th), axis=0)
#
#     return res
#
#
# def handle_result(solver, t, y, yd):
#
#     order.append(solver.get_last_order())
#
#     solver.t_sol.extend([t])
#     solver.y_sol.extend([y])
#     solver.yd_sol.extend([yd])
#
#
# alpha1 = np.arctan2(L_crank, ecc)
# alpha2 = np.arcsin(np.sqrt(ecc**2 + L_crank**2)/L_coupler)
#
# Rot_alpha1 = np.array([[1, 0, 0],
#                        [0, np.cos(alpha1), -np.sin(alpha1)],
#                        [0, np.sin(alpha1), np.cos(alpha1)]])
#
# Rot_alpha2 = np.array([[np.cos(alpha2), -np.sin(alpha1), 0],
#                        [np.sin(alpha1), +np.cos(alpha1), 0]
#                        [0, 0, 1]])
#
# T_I2B = Rot_alpha2 @ Rot_alpha1.T
#
# y0 = np.zeros(n_tot)  # Initial conditions
# yd0 = np.zeros(n_tot)  # Initial conditions
#
# v0P_I = [0, -omega_cr * L_crank, 0]
# v0P = T_I2B @ v0P_I
# y0[:3] = v0P
#
# rP_I = [0, 0, L_crank]
#
# xC_I = np.sqrt(L_coupler**2 - L_crank**2 - ecc**2)
# rC_I = [xC_I, -ecc, 0]
#
# dir_couplerI = rP_I - rC_I
#
# dir_om_cl = np.cross(dir_couplerI, v0P_I)
# dir_om_cl = dir_om_cl/np.linalg.norm(dir_om_cl)
# om_cl_norm = omega_cr * L_crank / L_coupler
# om_clI = om_cl_norm * dir_om_cl
#
# y0[3:6] = T_I2B @ om_clI
# y0[-9:-6] = T_I2B @ [0, 0, - mass_coupler * omega_cr ** 2 * L_crank]
# y0[-4:] = quaternion.as_quat_array(T_I2B.T)
#
# yd0[:3] = T_I2B @ [0, 0, - omega_cr ** 2 * L_crank]
# y0[-4:] = quaternion.as_float_array(quaternion.from_rotation_matrix(T_I2B.T))
#
# # Create an Assimulo implicit problem
# imp_mod = Implicit_Problem(dae_closed_phs, y0, yd0, name='dae_closed_pHs')
# imp_mod.handle_result = handle_result
#
# # Set the algebraic components
# imp_mod.algvar = list(np.concatenate((np.ones(n_e), np.zeros(n_tot - n_e - 1), np.ones(1))))
#
# # Create an Assimulo implicit solver (IDA)
# imp_sim = IDA(imp_mod)  # Create a IDA solver
#
# # Sets the paramters
# imp_sim.atol = 1e-6  # Default 1e-6
# imp_sim.rtol = 1e-6  # Default 1e-6
# imp_sim.suppress_alg = True  # Suppress the algebraic variables on the error test
# imp_sim.report_continuously = True
# # imp_sim.maxh = 1e-6
#
# # Let Sundials find consistent initial conditions by use of 'IDA_YA_YDP_INIT'
# imp_sim.make_consistent('IDA_YA_YDP_INIT')
#
# # Simulate
# t_final = 8/150
# n_ev = 1000
# t_ev = np.linspace(0, t_final, n_ev)
# t_sol, y_sol, yd_sol = imp_sim.simulate(t_final, 0, t_ev)
# e_sol = y_sol[:, :n_e].T
# lmb_sol = y_sol[:, n_e:-1].T
# th_cl_sol = y_sol[:, -1]
#
# er_cl_sol = e_sol[:nr_coupler, :]
# ep_sol = e_sol[nr_tot:nr_tot + n_p, :]
# om_cl_sol = er_cl_sol[3:6, :]
#
#
# nu = int(n_p/5)
# nv = int(2*n_p/5)
# nw = int(2*n_p/5)
#
# up_sol = ep_sol[:nu, :]
# vp_sol = ep_sol[nu:nu + nv, :]
# wp_sol = ep_sol[nu + nv:n_p, :]
#
# n_ev = len(t_sol)
# dt_vec = np.diff(t_sol)
#
# n_plot = 11
# eu_plot = np.zeros((n_plot, n_ev))
# ev_plot = np.zeros((n_plot, n_ev))
# ew_plot = np.zeros((n_plot, n_ev))
#
# x_plot = np.linspace(0, L_coupler, n_plot)
# w_plot = np.zeros((n_plot, n_ev))
#
# t_plot = t_sol
#
# for i in range(n_ev):
#     eu_plot[:, i], ev_plot[:, i], ew_plot[:, i] = draw_deformation3D(n_plot, [0, 0, 0], ep_sol[:, i], L_coupler)[1:4]
#
# ind_midpoint = int((n_plot-1)/2)
#
# euM_B = eu_plot[ind_midpoint, :]
# evM_B = ev_plot[ind_midpoint, :]
# ewM_B = ew_plot[ind_midpoint, :]
#
# omega_cr_int = interp1d(t_ev, om_cl_sol, kind='linear')
# euM_B_int = interp1d(t_ev, euM_B, kind='linear')
# evM_B_int = interp1d(t_ev, evM_B, kind='linear')
# ewM_B_int = interp1d(t_ev, ewM_B, kind='linear')
#
# def sys(t,y):
#
#     dudt = euM_B_int(t) # + omega_cr_int(t) * y[1]
#     dvdt = evM_B_int(t) # - omega_cr_int(t) * y[0]
#     dwdt = ewM_B_int(t) # - omega_cr_int(t) * y[0]
#
#     dydt = np.array([dudt, dvdt, dwdt])
#     return dydt
#
# uM0 = 0
# vM0 = 0
# wM0 = 0
#
# r_sol = solve_ivp(sys, [0, t_final], [uM0, vM0, wM0], method='RK45', t_eval=t_ev)
#
# uM_B = r_sol.y[0, :]
# vM_B = r_sol.y[1, :]
# wM_B = r_sol.y[2, :]
#
# fntsize = 16
# fig = plt.figure()
# plt.plot(omega_cr*t_ev, -vM_B/L_coupler, 'b-', label="v midpoint")
# plt.plot(omega_cr*t_ev, -wM_B/L_coupler, 'b-', label="w midpoint")
# plt.xlabel(r'Crank angle [rad]', fontsize = fntsize)
# plt.ylabel(r'w normalized', fontsize = fntsize)
# plt.title(r"Midpoint deflection", fontsize=fntsize)
# plt.legend(loc='upper left')
# axes = plt.gca()
# # axes.set_ylim([-0.015, 0.02])
#
# plt.show()
#
# def skew(v):
#     assert len(v) == 3
#     skew_mat = np.array([[0, -v[2], v[1]],
#                         [v[2], 0 -v[0]],
#                         [-v[1], v[0], 0]])
#
#     return skew_mat
#
#
# def skew_flex(al_u, al_v, al_w, al_udis, al_vdis, al_wdis):
#
#     nu_dis = len(al_udis)
#     nv = len(al_v)
#     nw = len(al_w)
#
#     assert len(al_u)/2 == nu_dis
#     assert nv/2 == len(al_vdis)
#     assert nw/2 == len(al_wdis)
#
#     n_rows = len(al_udis) + len(al_v) + len(al_w)
#     skew_mat_flex = np.zeros((n_rows, 3))
#
#     nv_end = nu_dis + nv
#     nw_end = nu_dis + nv + nw
#     skew_mat_flex[:nu_dis, :] = np.concatenate((np.zeros(nu_dis), - al_wdis, al_vdis), axis=1)
#     skew_mat_flex[nu_dis:nv_end, :] = np.concatenate((al_w, np.zeros(nv), -al_u), axis=1)
#     skew_mat_flex[nv_end: nw_end, :] = np.concatenate((-al_v, al_u, np.zeros(nw)), axis=1)
#
#     return skew_mat_flex
