import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
fntsize = 15

import numpy as np
import quaternion
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import scipy.linalg as la
from scipy.sparse.linalg import lsqr

from modules_ph.classes_phsystem import SysPhdaeRig
from system_components.beams import SpatialBeam, draw_deformation3D, matrices_j3d
from math import pi

from tools_plotting.animate_lines import animate_line3d
from assimulo.solvers import ODASSL, Radau5DAE, IDA

from assimulo.implicit_ode import Implicit_Problem


def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


L_crank = 0.15
L_coupler = 0.3
d = 0.006

ecc = 0.1
offset_cr = 0

rho_coupler = 7.87 * 10 ** 3
E_coupler = 200 * 10**9
A_coupler = pi * d**2 / 4
I_coupler = pi * d**4 / 64


Jxx_coupler = 2 * I_coupler * rho_coupler * L_coupler
omega_cr = 150

nr_coupler = 6
nr_slider = 3
# nr_tot = nr_coupler + nr_slider

n_elem = 2
bc = 'SS'
coupler = SpatialBeam(n_elem, L_coupler, rho_coupler, A_coupler, E_coupler, I_coupler, Jxx_coupler, bc=bc)

mass_coupler = rho_coupler * A_coupler * L_coupler
mass_slider = 0.5 * mass_coupler
M_slider = mass_slider * np.eye(nr_slider)

J_slider = np.zeros((nr_slider, nr_slider))
B_slider = np.eye(nr_slider)

slider = SysPhdaeRig(nr_slider, 0, nr_slider, 0, 0, E=M_slider, J=J_slider, B=B_slider)

sys = SysPhdaeRig.transformer_ordered(coupler, slider, [6, 7, 8], [0, 1, 2], np.eye(nr_slider))

n_sys = sys.n
n_e = sys.n_e
n_p = sys.n_p
n_pu = int(n_p/5)
n_pv = 2*n_pu
n_pw = 2*n_pu
n_f = coupler.n_f
nr_tot = sys.n_r

# E_sys = sys.E
M_e = sys.M_e
invM_e = la.inv(M_e)
J_e = sys.J_e

# E_sys = sys.E
# J_sys = sys.J

nlmd_cl = 3
nlmd_sys = sys.n_lmb
nlmd_sl = 2
nquat = 4

G_e = sys.G_e
G_coupler = np.zeros((n_e, nlmd_cl))
G_coupler[:nlmd_cl, :nlmd_cl] = np.eye(nlmd_cl)
G_slider = np.zeros((n_e, nlmd_sl))

n_tot = n_e + nlmd_cl + nlmd_sys + nlmd_sl + nquat  # 4 coupler, 2 slider, 3 coupler-slider and quaternions
dx = L_coupler/n_elem


t_final = abs(8/omega_cr)


Jf_tx, Jf_ty, Jf_tz, Jf_ry, Jf_rz, Jf_fx, Jf_fy, Jf_fz = matrices_j3d(n_elem, L_coupler, rho_coupler, A_coupler, bc=bc)

def dae_closed_phs(t, y, yd):

    print(t/t_final*100)

    # dy_sys = yd[:n_sys]
    # y_sys = y[:n_sys]

    de_sys = yd[:n_e]
    dquat_cl = yd[-nquat:]

    e_sys = y[:n_e]

    vx_coupler = e_sys[0]
    vy_coupler = e_sys[1]
    vz_coupler = e_sys[2]

    omx_coupler = e_sys[3]
    omy_coupler = e_sys[4]
    omz_coupler = e_sys[5]

    efx_coupler = e_sys[nr_tot:nr_tot+n_pu]
    efy_coupler = e_sys[nr_tot+n_pu:nr_tot+n_pu+n_pv]
    efz_coupler = e_sys[nr_tot+n_pu+n_pv:nr_tot+n_p]

    lmd_sys = y[n_e:n_sys]
    lmd_cl = y[n_sys:n_sys + nlmd_cl]
    lmd_mass = y[n_sys + nlmd_cl: n_sys + nlmd_cl + nlmd_sl]

    quat_cl = y[-nquat:]/np.linalg.norm(y[-nquat:])

    omega_cl = y[3:6]

    p_coupler = M_e[:3, :] @ e_sys + M_e[:3, nr_tot:n_e] @ e_sys[nr_tot:]
    pi_coupler = M_e[3:6, :] @ e_sys + M_e[3:6, nr_tot:n_e] @ e_sys[nr_tot:]

    p_slider = M_e[nr_coupler:nr_tot, :] @ e_sys

    J_e[:3, 3:6] = skew(p_coupler)
    J_e[3:6, :3] = skew(p_coupler)
    J_e[3:6, 3:6] = skew(pi_coupler)
    J_e[nr_coupler:nr_tot, 3:6] = skew(p_slider)

    Jf_om = + Jf_tx * vx_coupler + Jf_ty * vy_coupler + Jf_tz * vz_coupler \
            + Jf_ry * omy_coupler + Jf_rz * omz_coupler  \
            + Jf_fy @ efy_coupler + Jf_fz @ efz_coupler + Jf_fx @ efx_coupler

    Jf_om_cor = Jf_fy @ efy_coupler + Jf_fz @ efz_coupler + Jf_fx @ efx_coupler

    J_e[nr_tot:nr_tot + n_p, 3:6] = Jf_om + Jf_om_cor
    J_e[3:6, nr_tot:nr_tot + n_p] = -2*Jf_om.T

    act_quat = np.quaternion(quat_cl[0], quat_cl[1], quat_cl[2], quat_cl[3])
    Rot_cl = quaternion.as_rotation_matrix(act_quat)

    G_slider[nr_coupler:nr_tot, :] = Rot_cl[[1, 2], :].T

    vP_cl = omega_cr * L_crank * np.array([0, -np.cos(omega_cr * t), -np.sin(omega_cr * t)])

    # res_sys = E_sys @ dy_sys - J_sys @ y_sys - G_coupler @ lmd_cl - G_slider @ lmd_mass

    res_e = M_e @ de_sys - J_e @ e_sys - G_coupler @ lmd_cl - G_slider @ lmd_mass - G_e @ lmd_sys

    # dvP_cl = omega_cr**2 * L_crank * np.array([0, np.sin(omega_cr * t), -np.cos(omega_cr * t)])
    # deE_sys = invM_e @ (J_e @ e_sys + G_coupler @ lmd_cl + G_slider @ lmd_mass + G_e @ lmd_sys)
    # dres_lmb = G_e.T @ deE_sys
    # dres_cl_v = - deE_sys[:3] - skew(omega_cl) @ Rot_cl.T @ vP_cl + Rot_cl.T @ dvP_cl
    # # dres_cl_v = - Rot_cl @ deE_sys[:3] - Rot_cl @ skew(omega_cl) @ e_sys[:3] + dvP_cl
    # dres_slider = Rot_cl[[1, 2], :] @ skew(omega_cl) @ e_sys[nr_coupler:nr_tot] + Rot_cl[[1, 2], :] @ deE_sys[nr_coupler:nr_tot]

    res_lmb = G_e.T @ e_sys
    res_cl_v = - Rot_cl @ e_sys[:3] + vP_cl
    # res_cl_v = - e_sys[:3] + Rot_cl.T @ vP_cl
    res_slider = Rot_cl[[1, 2], :] @ e_sys[nr_coupler:nr_tot]

    Omegacl_mat = np.array([[0, -omega_cl[0], -omega_cl[1], -omega_cl[2]],
                            [omega_cl[0],   0, omega_cl[2], -omega_cl[1]],
                            [omega_cl[1],   -omega_cl[2], 0, omega_cl[0]],
                            [omega_cl[2],  omega_cl[1], -omega_cl[0], 0]])

    res_quat = dquat_cl - 0.5 * Omegacl_mat @ quat_cl

    res = np.concatenate((res_e, res_lmb, res_cl_v, res_slider, res_quat), axis=0)
    # res = np.concatenate((res_e, dres_lmb, dres_cl_v, dres_slider, res_quat), axis=0)

    return res


order = []


def handle_result(solver, t, y, yd):

    order.append(solver.get_last_order())

    solver.t_sol.extend([t])
    solver.y_sol.extend([y])
    solver.yd_sol.extend([yd])


z0_cr = L_crank + offset_cr
theta_cr = pi/2
theta1_cl = np.arcsin(ecc/np.sqrt(L_coupler**2 - z0_cr**2))
theta2_cl = np.arcsin(z0_cr/L_coupler)

A_vel = np.array([[L_coupler*np.cos(theta2_cl)*np.sin(theta1_cl), L_coupler*np.sin(theta2_cl)*np.cos(theta1_cl), 1],
                  [-L_coupler*np.cos(theta2_cl)*np.cos(theta1_cl), L_coupler*np.sin(theta2_cl)*np.sin(theta1_cl), 0],
                  [0, L_coupler*np.cos(theta2_cl), 0]])

b_vel = np.array([0, L_crank*np.sin(theta_cr)*omega_cr, L_crank*np.cos(theta_cr)*omega_cr])

dtheta1_cl, dtheta2_cl, dx_sl = la.solve(A_vel, b_vel)

# omx_B = np.sin(theta2_cl) * dtheta1_cl
omx_B = 0

omy_B = dtheta2_cl
omz_B = -np.cos(theta2_cl) * dtheta1_cl

Rot_theta1 = np.array([[np.cos(theta1_cl), -np.sin(theta1_cl), 0],
                       [np.sin(theta1_cl), +np.cos(theta1_cl), 0],
                       [0, 0, 1]])

Rot_theta2 = np.array([[np.cos(theta2_cl), 0,  np.sin(theta2_cl)],
                       [0, 1, 0],
                       [-np.sin(theta2_cl), 0, np.cos(theta2_cl)]])

T_I2B = Rot_theta2.T @ Rot_theta1
R_B2I = T_I2B.T

y0 = np.zeros(n_tot)  # Initial conditions
yd0 = np.zeros(n_tot)  # Initial conditions

v0P_I = np.array([0, -omega_cr * L_crank, 0])
dv0P_I = omega_cr**2 * L_crank * np.array([0, 0, -1])
ddv0P_I = omega_cr**3 * L_crank * np.array([0, 1, 0])

r0P_I = np.array([0, 0, z0_cr])
x0C_I = np.sqrt(L_coupler**2 - z0_cr**2 - ecc**2)
r0C_I = np.array([x0C_I, -ecc, 0])

rCP_I = r0P_I - r0C_I
assert L_coupler == np.linalg.norm(rCP_I)

v0P_B = T_I2B @ v0P_I
rCP_B = T_I2B @ rCP_I

om0_clB = np.array([omx_B, omy_B, omz_B])
om0_clI = R_B2I @ om0_clB

v0C_I = v0P_I + skew(rCP_I) @ om0_clI

tol = 1e-14
assert abs(v0C_I[0] - dx_sl) < tol
assert abs(v0C_I[1]) < tol
assert abs(v0C_I[2]) < tol


v0P_B = T_I2B @ v0P_I

e0_cl = np.concatenate((v0P_B, om0_clB))
e0_sl = T_I2B @ np.array([dx_sl, 0, 0])
e0_fl = np.zeros((sys.n_f, ))
e0_sys = np.concatenate((e0_cl, e0_sl, e0_fl))
quat0_sys = quaternion.as_float_array(quaternion.from_rotation_matrix(R_B2I))


def find_initial_condition(e0, quat0):
    G0_slider = np.zeros((n_e, 2))
    G0_slider[nr_coupler:nr_tot, :] = R_B2I[[1, 2], :].T

    dG0_slider = np.zeros((n_e, 2))
    dG0_slider[nr_coupler:nr_tot, :] = (R_B2I[[1, 2], :] @ skew(om0_clB)).T

    G0_lmb = np.concatenate((G_e, G_coupler, G0_slider), axis=1)
    dG0_lmb = np.concatenate((G_e, G_coupler, dG0_slider), axis=1)

    e0_sl = e0[nr_coupler:nr_tot]

    b_e = np.zeros((3, ))
    b_cl_v = T_I2B @ dv0P_I - skew(om0_clB) @ T_I2B @ v0P_I
    b_sl = R_B2I[[1, 2], :] @ skew(om0_clB) @ e0_sl

    b_ext = np.concatenate((b_e, b_cl_v, b_sl), axis=0)
    b_sys = G0_lmb.T @ invM_e @ J_e @ e0

    A_lmb = G0_lmb.T @ invM_e @ G0_lmb

    lmb0 = np.linalg.solve(A_lmb, b_ext - b_sys)
    de0 = invM_e @ (J_e @ e0 + G0_lmb @ lmb0)

    dA_lmb = dG0_lmb.T @ invM_e @ G0_lmb + G0_lmb.T @ invM_e @ dG0_lmb

    dom0_cl = de0[3:6]
    de0_sl = de0[6:9]

    db_e = np.zeros((3,))
    db_cl_v = - skew(om0_clB) @ T_I2B @ dv0P_I + T_I2B @ ddv0P_I - skew(dom0_cl) @ T_I2B @ v0P_I + \
        skew(om0_clB) @ skew(om0_clB) @ T_I2B @ v0P_I - skew(om0_clB) @ T_I2B @ dv0P_I
    db_sl = R_B2I[[1, 2], :] @ skew(om0_clB) @ skew(om0_clB) @ e0_sl + \
            R_B2I[[1, 2], :] @ skew(dom0_cl) @ e0_sl + R_B2I[[1, 2], :] @ skew(om0_clB) @ de0_sl

    db_ext = np.concatenate((db_e, db_cl_v, db_sl), axis=0)

    db_sys = G0_lmb.T @ invM_e @ J_e @ de0 + dG0_lmb.T @ invM_e @ J_e @ e0

    dlmb0 = np.linalg.solve(A_lmb, db_ext - db_sys - dA_lmb @ lmb0)

    Om0_cl_mat = np.array([[          0, -om0_clB[0], -om0_clB[1], -om0_clB[2]],
                            [om0_clB[0],           0,  om0_clB[2], -om0_clB[1]],
                            [om0_clB[1], -om0_clB[2],           0,  om0_clB[0]],
                            [om0_clB[2],  om0_clB[1], -om0_clB[0],          0]])

    dquat0 = 0.5 * Om0_cl_mat @ quat0

    return lmb0, de0, dlmb0, dquat0


lmb0_sys, de0_sys, dlmb0_sys, dquat0_sys = find_initial_condition(e0_sys, quat0_sys)

y0[:n_e] = e0_sys
y0[n_e:-nquat] = lmb0_sys
y0[-nquat:] = quat0_sys

# yd0[:n_e] = de0_sys
# y0[n_e:-nquat] = dlmb0_sys
yd0[-nquat:] = dquat0_sys

# Create an Assimulo implicit problem
imp_mod = Implicit_Problem(dae_closed_phs, y0, yd0, name='dae_closed_pHs')
imp_mod.handle_result = handle_result

# Set the algebraic components
imp_mod.algvar = list(np.concatenate((np.ones(n_e), np.zeros(n_tot - n_e - nquat), np.ones(nquat))))
# Create an Assimulo implicit solver
# imp_sim = IDA(imp_mod)  # Create a IDA solver
imp_sim = IDA(imp_mod)  # Create a IDA solver

# Sets the paramters
imp_sim.atol = 1e-6  # Default 1e-6
imp_sim.rtol = 1e-6  # Default 1e-6
imp_sim.suppress_alg = True  # Suppress the algebraic variables on the error test
imp_sim.report_continuously = True
imp_sim.verbosity = 10
imp_sim.maxh = 1e-6

# Let Sundials find consistent initial conditions by use of 'IDA_YA_YDP_INIT'
imp_sim.make_consistent('IDA_YA_YDP_INIT')

# Simulate
n_ev = 5000
t_ev = np.linspace(0, t_final, n_ev)
t_sol, y_sol, yd_sol = imp_sim.simulate(t_final, 0, t_ev)
t_sol = np.array(t_sol)
# t_sol, y_sol, yd_sol = imp_sim.simulate(t_final)
# t_sol = t_sol[0:]
# y_sol = y_sol[0:]
# yd_sol = yd_sol[0:]
e_sol = y_sol[:, :n_e].T
lmb_sol = y_sol[:, n_e:-nquat].T
quat_cl_sol = quaternion.as_quat_array(y_sol[:, -nquat:])

er_cl_sol = e_sol[:nr_coupler, :]
ep_sol = e_sol[nr_tot:nr_tot + n_p, :]
om_cl_sol = er_cl_sol[3:6, :]

plt.figure()
plt.plot(t_sol, om_cl_sol.T)
plt.xlabel(r"Time")
plt.ylabel(r"Omega")

nu = int(n_p/5)
nv = int(2*n_p/5)
nw = int(2*n_p/5)

up_sol = ep_sol[:nu, :]
vp_sol = ep_sol[nu:nu + nv, :]
wp_sol = ep_sol[nu + nv:n_p, :]

if bc == 'CF':
    euM_B = up_sol[int(n_elem / 2) - 1, :]
    evM_B = vp_sol[n_elem - 2, :]
    ewM_B = wp_sol[n_elem - 2, :]
else:
    euM_B = up_sol[int(n_elem / 2) - 1, :]
    evM_B = vp_sol[n_elem - 1, :]
    ewM_B = wp_sol[n_elem - 1, :]


# n_ev = len(t_sol)
# dt_vec = np.diff(t_sol)
#
# n_plot = 11
# eu_plot = np.zeros((n_plot, n_ev))
# ev_plot = np.zeros((n_plot, n_ev))
# ew_plot = np.zeros((n_plot, n_ev))
#
# x_plot = np.linspace(0, L_coupler, n_plot)


# zeros_rig = [0,0,0,0,0,0]
# for i in range(n_ev):
#     eu_plot[:, i], ev_plot[:, i], ew_plot[:, i] = draw_deformation3D(n_plot, zeros_rig, ep_sol[:, i], L_coupler)[1:4]
#
# ind_midpoint = int((n_plot-1)/2)
#
# euM_B = eu_plot[ind_midpoint, :]
# evM_B = ev_plot[ind_midpoint, :]
# ewM_B = ew_plot[ind_midpoint, :]
#
# print(euM_B.shape, evM_B.shape, ewM_B.shape)
#
eM_B = np.column_stack((euM_B, evM_B, ewM_B)).T

euM_B_int = interp1d(t_sol, euM_B, kind='linear')
evM_B_int = interp1d(t_sol, evM_B, kind='linear')
ewM_B_int = interp1d(t_sol, ewM_B, kind='linear')

omega_cl_int = interp1d(t_sol, om_cl_sol, kind='linear')
eM_B_int = interp1d(t_sol, eM_B, kind='linear')

# er_cl_sol = e_sol[:nr_coupler, :]
# ef_cl_sol = e_sol[nr_tot:nr_tot + n_f, :]
# erf_cl_sol = np.concatenate((er_cl_sol, ef_cl_sol), axis=0)
#
# vP_cl_sol = er_cl_sol[:3, :]
# om_cl_sol = er_cl_sol[3:6, :]
# vslB_sol = e_sol[nr_coupler:nr_tot, :]
#
# e1I_sol = np.zeros((3, n_ev))
# rP_cl = np.zeros((3, n_ev))
#
#
# vslI_sol = np.zeros((3, n_ev))
# vclCI_sol = np.zeros((3, n_ev))
# vclCB_sol = np.zeros((3, n_ev))
#
# PC_skew = skew([-L_coupler, 0, 0])

# for i in range(n_ev):
#     vclCB_sol[:, i] = coupler.B[:, [6,7,8]].T @ erf_cl_sol[:, i]
#     vclCI_sol[:, i] = quaternion.as_rotation_matrix(quat_cl_sol[i]) @ vclCB_sol[:, i]
#
#     vslI_sol[:, i] = quaternion.as_rotation_matrix(quat_cl_sol[i]) @ vslB_sol[:, i]
#
#     rP_cl[:, i] = np.array([0, -L_crank * np.sin(omega_cr*t_sol[i]), offset_cr + L_crank * np.cos(omega_cr*t_sol[i])])
#
#
# vslI_int = interp1d(t_sol, vslI_sol, kind='linear')
# vclCI_int = interp1d(t_sol, vclCI_sol, kind='linear')
#
# def sys(t, y):
#
#     dydt_sl = vslI_int(t)
#     dydt_cl = vclCI_int(t)
#
#     return np.concatenate((dydt_sl, dydt_cl))
#
#
# r_sol = solve_ivp(sys, [0, t_final], np.concatenate((r0C_I, r0C_I)), method='RK45', t_eval=t_ev)
#
#
# n_ev = len(t_sol)
# dt_vec = np.diff(t_sol)
#
# xP_sl = r_sol.y[0, :]
# yP_sl = r_sol.y[1, :]
# zP_sl = r_sol.y[2, :]
#
# xC_cl = r_sol.y[3, :]
# yC_cl = r_sol.y[4, :]
# zC_cl = r_sol.y[5, :]
#
# data = np.array([[rP_cl[0], rP_cl[1], rP_cl[2]], [xC_cl, yC_cl, zC_cl], [xP_sl, yP_sl, zP_sl]])


def sys(t,y):

    dydt = eM_B_int(t) - skew(omega_cl_int(t)) @ y
    # dudt = euM_B_int(t)
    # dvdt = evM_B_int(t)
    # dwdt = ewM_B_int(t)
    #
    # dydt = np.array([dudt, dvdt, dwdt])
    return dydt


uM0 = 0
vM0 = 0
wM0 = 0

r_sol = solve_ivp(sys, [0, t_final], [uM0, vM0, wM0], method='RK45', t_eval=t_sol)

uM_B = r_sol.y[0, :]
vM_B = r_sol.y[1, :]
wM_B = r_sol.y[2, :]

# uM_I = np.zeros(n_ev)
# vM_I = np.zeros(n_ev)
# wM_I = np.zeros(n_ev)
#
#
# for i in range(n_ev):
#
#     vecM_b = np.array([uM_B[i], vM_B[i], wM_B[i]])
#     Rot_i = quaternion.as_rotation_matrix(quat_cl_sol[i])
#     uM_I[i] = Rot_i[0, :] @ vecM_b
#     vM_I[i] = Rot_i[1, :] @ vecM_b
#     wM_I[i] = Rot_i[2, :] @ vecM_b


fntsize = 16
#
# fig = plt.figure()
# plt.plot(omega_cr*t_sol, uM_I/L_coupler, 'b-', label="u midpoint")
# plt.xlabel(r'Crank angle [rad]', fontsize = fntsize)
# plt.ylabel(r'w normalized', fontsize = fntsize)
# plt.title(r"Midpoint deflection in I", fontsize=fntsize)
# plt.legend(loc='upper left')
#
# fig = plt.figure()
# plt.plot(omega_cr*t_sol, vM_I/L_coupler, 'b-', label="y midpoint")
# plt.xlabel(r'Crank angle [rad]', fontsize = fntsize)
# plt.ylabel(r'w normalized', fontsize = fntsize)
# plt.title(r"Midpoint deflection in I", fontsize=fntsize)
# plt.legend(loc='upper left')
#
# fig = plt.figure()
# plt.plot(omega_cr*t_sol, wM_I/L_coupler, 'b-', label="z midpoint")
# plt.xlabel(r'Crank angle [rad]', fontsize = fntsize)
# plt.ylabel(r'w normalized', fontsize = fntsize)
# plt.title(r"Midpoint deflection in I", fontsize=fntsize)
# plt.legend(loc='upper left')

# plt.show()

# fig = plt.figure()
# plt.plot(omega_cr*t_sol, uM_B/L_coupler, 'b-', label="u midpoint")
# plt.xlabel(r'Crank angle [rad]', fontsize = fntsize)
# plt.ylabel(r'w normalized', fontsize = fntsize)
# plt.title(r"Midpoint deflection in B", fontsize=fntsize)
# plt.legend(loc='upper left')

fig = plt.figure()
plt.plot(omega_cr*t_sol, vM_B/L_coupler, 'b-', label="y midpoint")
plt.xlabel(r'Crank angle [rad]', fontsize = fntsize)
plt.ylabel(r'w normalized', fontsize = fntsize)
plt.title(r"Midpoint deflection in B", fontsize=fntsize)
plt.legend(loc='upper left')

fig = plt.figure()
plt.plot(omega_cr*t_sol, wM_B/L_coupler, 'b-', label="z midpoint")
plt.xlabel(r'Crank angle [rad]', fontsize = fntsize)
plt.ylabel(r'w normalized', fontsize = fntsize)
plt.title(r"Midpoint deflection in B", fontsize=fntsize)
plt.legend(loc='upper left')

plt.show()

# anim = animate_line3d(data, t_sol)
# plt.show()

#
# min_angle = 0*pi
# max_angle = 2*pi
# n_fig = 10
# th_rot_vec = np.linspace(min_angle, max_angle, n_fig)
#
# for i in range(n_fig):
#     fig = plt.figure()
#     th_rot = th_rot_vec[i]
#     vcos_th = vM_B*np.cos(th_rot)/L_coupler
#     vsin_th = wM_B*np.sin(th_rot)/L_coupler
#     # plt.plot(omega_cr*t_sol, (vM_B)/L_coupler, 'r-', label="theta midpoint " + str(th_rot*180/pi))
#     # plt.plot(omega_cr*t_sol, (wM_B)/L_coupler, 'r-', label="theta midpoint " + str(th_rot*180/pi))
#     plt.plot(omega_cr*t_sol, vcos_th + vsin_th, 'r-', label="theta midpoint " + str(th_rot*180/pi))
#     plt.xlabel(r'Crank angle [rad]', fontsize = fntsize)
#     plt.ylabel(r'v normalized', fontsize = fntsize)
#     plt.title(r"Midpoint deflection", fontsize=fntsize)
#     plt.legend(loc='upper left')
#
# plt.show()
