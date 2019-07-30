import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
fntsize = 15

import numpy as np
import quaternion
from scipy.spatial.transform import Rotation
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import scipy.linalg as la

from modules_phdae.classes_phsystem import SysPhdaeRig
from system_components.beams import SpatialBeam, draw_deformation3D
from math import pi

from assimulo.solvers import IDA
from assimulo.implicit_ode import Implicit_Problem


def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def skew_flex(al_u, al_v, al_w, al_udis, al_vdis, al_wdis):

    nu_dis = len(al_udis)
    nv = len(al_v)
    nw = len(al_w)

    n_rows = len(al_udis) + len(al_v) + len(al_w)
    skew_mat_flex = np.zeros((n_rows, 3))

    nv_end = nu_dis + nv
    nw_end = nu_dis + nv + nw
    skew_mat_flex[:nu_dis, :] = np.column_stack((np.zeros((nu_dis,)), - al_wdis, al_vdis))
    skew_mat_flex[nu_dis:nv_end, :] = np.column_stack((al_w, np.zeros((nv,)), -al_u))
    skew_mat_flex[nv_end: nw_end, :] = np.column_stack((-al_v, al_u, np.zeros((nw, ))))

    return skew_mat_flex


L_crank = 0.15
L_coupler = 0.3
d = 0.006

ecc = 0.1
offset_cr = 0

rho_coupler = 7.87 * 10 ** 3
E_coupler = 200 * 10**9
A_coupler = pi * d**2 / 4
I_coupler = pi * d**4 / 64
Jxx_coupler = 2 * I_coupler * rho_coupler
n_elem = 2

omega_cr = 150

nr_coupler = 6
nr_slider = 3
# nr_tot = nr_coupler + nr_slider


coupler = SpatialBeam(n_elem, L_coupler, rho_coupler, A_coupler, E_coupler, I_coupler, Jxx_coupler)

mass_coupler = rho_coupler * A_coupler * L_coupler
mass_slider = 0.5 * mass_coupler
M_slider = mass_slider * np.eye(nr_slider)

J_slider = np.zeros((nr_slider, nr_slider))
B_slider = np.eye(nr_slider)

slider = SysPhdaeRig(nr_slider, 0, nr_slider, 0, 0, E=M_slider, J=J_slider, B=B_slider)

sys = SysPhdaeRig.transformer_ordered(coupler, slider, [6, 7, 8], [0, 1, 2], np.eye(nr_slider))

# plt.spy(sys.E); plt.show()

n_sys = sys.n
n_e = sys.n_e
n_p = sys.n_p
n_pu = int(n_p/5)
n_pv = 2*n_pu
n_pw = 2*n_pu
nr_tot = sys.n_r


# E_sys = sys.E
M_e = sys.M_e
invM_e = la.inv(M_e)
J_e = sys.J_e

# E_sys = sys.E
# J_sys = sys.J

G_e = sys.G_e
G_coupler = sys.B_e[:, [0, 1, 2]]
G_slider = np.zeros((n_e, 2))

n_tot = n_e + 12  # 2*8 lambda (3 coupler, 2 slider, 3 coupler-slider) and quaternions
order = []

dx = L_coupler/n_elem

t_final = 8/omega_cr


def dae_closed_phs(t, y, yd):

    print(t/t_final*100)

    # dy_sys = yd[:n_sys]
    # y_sys = y[:n_sys]

    de_sys = yd[:n_e]
    dquat_cl = yd[-4:]

    e_sys = y[:n_e]

    lmd_sys = y[n_e:n_sys]
    lmd_cl = y[n_sys:n_sys + 3]
    lmd_mass = y[n_sys + 3: -4]

    quat_cl = y[-4:]/np.linalg.norm(y[-4:])

    omega_cl = y[3:6]

    p_coupler = M_e[:3, :] @ e_sys
    pi_coupler = M_e[3:6, :] @ e_sys

    p_slider = M_e[nr_coupler:nr_tot, :] @ e_sys

    p_v = M_e[nr_tot+n_pu:nr_tot+n_pu+n_pv, :] @ e_sys
    p_v[1::2] = 0
    p_vdis = np.array([p_v[i] for i in range(len(p_v)) if i % 2 == 0])

    p_w = M_e[nr_tot+n_pu+n_pv:nr_tot + n_p, :] @ e_sys
    p_w[1::2] = 0
    p_wdis = np.array([p_w[i] for i in range(len(p_w)) if i % 2 == 0])

    p_udis = M_e[nr_tot:nr_tot + n_pu, :] @ e_sys
    p_u = np.zeros_like(p_w)
    p_u[::2] = p_udis

    alflex_cross = skew_flex(p_u, p_v, p_w, p_udis, p_vdis, p_wdis)

    J_e[:3, 3:6] = skew(p_coupler)
    J_e[3:6, :3] = skew(p_coupler)
    J_e[3:6, 3:6] = skew(pi_coupler)
    J_e[nr_coupler:nr_tot, 3:6] = skew(p_slider)
    J_e[nr_tot:nr_tot+n_p, 3:6] = alflex_cross
    J_e[3:6, nr_tot:nr_tot+n_p] = -alflex_cross.T

    act_quat = np.quaternion(quat_cl[0], quat_cl[1], quat_cl[2], quat_cl[3])
    Rot_cl = quaternion.as_rotation_matrix(act_quat)

    G_slider[nr_coupler:nr_tot, :] = Rot_cl[[1, 2], :].T

    vP_cl = omega_cr * L_crank * np.array([0, -np.cos(omega_cr * t), -np.sin(omega_cr * t)])
    dvP_cl = omega_cr**2 * L_crank * np.array([0, np.sin(omega_cr * t), -np.cos(omega_cr * t)])

    # res_sys = E_sys @ dy_sys - J_sys @ y_sys - G_coupler @ lmd_cl - G_slider @ lmd_mass

    res_e = M_e @ de_sys - J_e @ e_sys - G_coupler @ lmd_cl - G_slider @ lmd_mass - G_e @ lmd_sys

    deE_sys = invM_e @ (J_e @ e_sys + G_coupler @ lmd_cl + G_slider @ lmd_mass + G_e @ lmd_sys)
    dres_lmb = G_e.T @ deE_sys
    dres_cl = - G_coupler.T @ deE_sys - skew(omega_cl) @ Rot_cl.T @ vP_cl + Rot_cl.T @ dvP_cl
    # dres_cl = - Rot_cl @ deE_sys[:3] - Rot_cl @ skew(omega_cl) @ e_sys[:3] + dvP_cl
    dres_slider = Rot_cl[[1, 2], :] @ skew(omega_cl) @ e_sys[nr_coupler:nr_tot] + Rot_cl[[1, 2], :] @ deE_sys[nr_coupler:nr_tot]

    # res_lmb = G_e.T @ e_sys
    # # res_cl = - Rot_cl @ e_sys[:3] + vP_cl
    # res_cl = - G_coupler.T @ e_sys + Rot_cl.T @ vP_cl
    # res_slider = Rot_cl[[1, 2], :] @ e_sys[nr_coupler:nr_tot]

    Omegacl_mat = np.array([[0, -omega_cl[0], -omega_cl[1], -omega_cl[2]],
                            [omega_cl[0],   0, omega_cl[2], -omega_cl[1]],
                            [omega_cl[1],   -omega_cl[2], 0, omega_cl[0]],
                            [omega_cl[2],  omega_cl[1], -omega_cl[0], 0]])

    res_quat = dquat_cl - 0.5 * Omegacl_mat @ quat_cl

    # res = np.concatenate((res_e, res_lmb, res_cl, res_slider, res_quat), axis=0)
    res = np.concatenate((res_e, dres_lmb, dres_cl, dres_slider, res_quat), axis=0)

    return res


def handle_result(solver, t, y, yd):

    order.append(solver.get_last_order())

    solver.t_sol.extend([t])
    solver.y_sol.extend([y])
    solver.yd_sol.extend([yd])


z0_cr = L_crank + offset_cr
theta1 = np.arcsin(ecc/np.sqrt(L_coupler**2 - z0_cr**2))
theta2 = np.arcsin(z0_cr/L_coupler)

Rot_theta1 = np.array([[np.cos(theta1), -np.sin(theta1), 0],
                       [np.sin(theta1), +np.cos(theta1), 0],
                       [0, 0, 1]])

Rot_theta2 = np.array([[np.cos(theta2), 0,  np.sin(theta2)],
                       [0, 1, 0],
                       [-np.sin(theta2), 0, np.cos(theta2)]])

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

A_com = np.concatenate((skew(rCP_I), np.array([-1, 0, 0]).reshape((-1, 1))), axis=1)
om_vx = np.linalg.lstsq(A_com, -v0P_I)[0]

om0_clI = om_vx[:3]
vx0C_I = om_vx[-1]

v0C_I = v0P_I + skew(rCP_I) @ om0_clI
assert v0C_I[1] < 1e-14 and v0C_I[2] < 1e-14

v0P_B = T_I2B @ v0P_I
om0_clB = T_I2B @ om0_clI

e0_cl = np.concatenate((v0P_B, om0_clB))
e0_sl = T_I2B @ v0C_I
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
    b_cl = T_I2B @ dv0P_I - skew(om0_clB) @ T_I2B @ v0P_I
    b_sl = R_B2I[[1, 2], :] @ skew(om0_clB) @ e0_sl

    b_ext = np.concatenate((b_e, b_cl, b_sl), axis=0)
    b_sys = G0_lmb.T @ invM_e @ J_e @ e0

    A_lmb = G0_lmb.T @ invM_e @ G0_lmb

    lmb0 = np.linalg.solve(A_lmb, b_ext - b_sys)
    de0 = invM_e @ (J_e @ e0 + G0_lmb @ lmb0)

    dA_lmb = dG0_lmb.T @ invM_e @ G0_lmb + G0_lmb.T @ invM_e @ dG0_lmb

    dom0_cl = de0[3:6]
    de0_sl = de0[6:9]

    db_e = np.zeros((3,))
    db_cl = - skew(om0_clB) @ T_I2B @ dv0P_I + T_I2B @ ddv0P_I - skew(dom0_cl) @ T_I2B @ v0P_I + \
        skew(om0_clB) @ skew(om0_clB) @ T_I2B @ v0P_I - skew(om0_clB) @ T_I2B @ dv0P_I
    db_sl = R_B2I[[1, 2], :] @ skew(om0_clB) @ skew(om0_clB) @ e0_sl + \
            R_B2I[[1, 2], :] @ skew(dom0_cl) @ e0_sl + R_B2I[[1, 2], :] @ skew(om0_clB) @ de0_sl

    db_ext = np.concatenate((db_e, db_cl, db_sl), axis=0)

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
y0[n_e:-4] = lmb0_sys
y0[-4:] = quat0_sys

yd0[:n_e] = de0_sys
y0[n_e:-4] = dlmb0_sys
yd0[-4:] = dquat0_sys

# Create an Assimulo implicit problem
imp_mod = Implicit_Problem(dae_closed_phs, y0, yd0, name='dae_closed_pHs')
imp_mod.handle_result = handle_result

# Set the algebraic components
imp_mod.algvar = list(np.concatenate((np.ones(n_e), np.zeros(n_tot - n_e - 4), np.ones(4))))
# Create an Assimulo implicit solver (IDA)
imp_sim = IDA(imp_mod)  # Create a IDA solver

# Sets the paramters
imp_sim.atol = 1e-6  # Default 1e-6
imp_sim.rtol = 1e-6  # Default 1e-6
# imp_sim.suppress_alg = True  # Suppress the algebraic variables on the error test
imp_sim.report_continuously = True
imp_sim.verbosity = 10
# imp_sim.maxh = 1e-6

# Let Sundials find consistent initial conditions by use of 'IDA_YA_YDP_INIT'
imp_sim.make_consistent('IDA_YA_YDP_INIT')

# Simulate
n_ev = 1000
t_ev = np.linspace(0, t_final, n_ev)
t_sol, y_sol, yd_sol = imp_sim.simulate(t_final, 0, t_ev)
e_sol = y_sol[:, :n_e].T
lmb_sol = y_sol[:, n_e:-4].T
quat_cl_sol = y_sol[:, -4:]

euler_angles = Rotation.from_quat(quat_cl_sol).as_euler('ZYX', degrees=True)

er_cl_sol = e_sol[:nr_coupler, :]
ep_sol = e_sol[nr_tot:nr_tot + n_p, :]
om_cl_sol = er_cl_sol[3:6, :]


nu = int(n_p/5)
nv = int(2*n_p/5)
nw = int(2*n_p/5)

up_sol = ep_sol[:nu, :]
vp_sol = ep_sol[nu:nu + nv, :]
wp_sol = ep_sol[nu + nv:n_p, :]

n_ev = len(t_sol)
dt_vec = np.diff(t_sol)

n_plot = 21
eu_plot = np.zeros((n_plot, n_ev))
ev_plot = np.zeros((n_plot, n_ev))
ew_plot = np.zeros((n_plot, n_ev))

x_plot = np.linspace(0, L_coupler, n_plot)

t_plot = t_sol

zeros_rig = [0,0,0,0,0,0]
for i in range(n_ev):
    eu_plot[:, i], ev_plot[:, i], ew_plot[:, i] = draw_deformation3D(n_plot, zeros_rig, ep_sol[:, i], L_coupler)[1:4]

ind_midpoint = int((n_plot-1)/2)

euM_B = eu_plot[ind_midpoint, :]
evM_B = ev_plot[ind_midpoint, :]
ewM_B = ew_plot[ind_midpoint, :]

eM_B = np.column_stack((euM_B, evM_B, ewM_B)).T

# euM_B_int = interp1d(t_ev, euM_B, kind='linear')
# evM_B_int = interp1d(t_ev, evM_B, kind='linear')
# ewM_B_int = interp1d(t_ev, ewM_B, kind='linear')

omega_cl_int = interp1d(t_ev, om_cl_sol, kind='linear')
eM_B_int = interp1d(t_ev, eM_B, kind='linear')

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

r_sol = solve_ivp(sys, [0, t_final], [uM0, vM0, wM0], method='RK45', t_eval=t_ev)

uM_B = r_sol.y[0, :]
vM_B = r_sol.y[1, :]
wM_B = r_sol.y[2, :]


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


pts = 10
vM_B_smooth = smooth(vM_B, pts)
wM_B_smooth = smooth(wM_B, pts)


fntsize = 16

min_angle = 0*pi/180
max_angle = 0*pi/180
n_fig = 1
th_rot_vec = np.linspace(min_angle, max_angle, n_fig)

for i in range(n_fig):
    fig = plt.figure()
    th_rot = th_rot_vec[i]
    plt.plot(omega_cr*t_ev, (vM_B_smooth*np.cos(th_rot) + wM_B_smooth*np.sin(th_rot))/L_coupler, 'r-', label="theta midpoint " + str(th_rot*180/pi))
    plt.xlabel(r'Crank angle [rad]', fontsize = fntsize)
    plt.ylabel(r'v normalized', fontsize = fntsize)
    plt.title(r"Midpoint deflection", fontsize=fntsize)
    plt.legend(loc='upper left')

#
# fig = plt.figure()
# plt.plot(omega_cr*t_ev, vM_B/L_coupler, 'b-', label="y midpoint")
# plt.xlabel(r'Crank angle [rad]', fontsize = fntsize)
# plt.ylabel(r'w normalized', fontsize = fntsize)
# plt.title(r"Midpoint deflection", fontsize=fntsize)
# plt.legend(loc='upper left')
# axes = plt.gca()

plt.show()

