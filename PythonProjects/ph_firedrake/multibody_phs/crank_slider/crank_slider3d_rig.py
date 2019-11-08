import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
fntsize = 15

import numpy as np
import quaternion
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import scipy.linalg as la

from modules_phdae.classes_phsystem import SysPhdaeRig
from system_components.beams import SpatialBeam, draw_deformation3D
from math import pi

from assimulo.solvers import IDA
from assimulo.implicit_ode import Implicit_Problem

from tools_plotting.animate_lines import animate_line3d
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
mass_coupler = rho_coupler * A_coupler * L_coupler
mass_slider = 0.5 * mass_coupler
Jxx_coupler = 2 * I_coupler * rho_coupler * L_coupler
Jyy_coupler = 1 / 3 * mass_coupler * L_coupler ** 2
Jzz_coupler = 1 / 3 * mass_coupler * L_coupler ** 2
n_elem = 2

omega_cr = 150

nr_coupler = 6
nr_slider = 3
nr_tot = nr_coupler + nr_slider

M_coupler = la.block_diag(mass_coupler * np.eye(3), np.diag([Jxx_coupler, Jyy_coupler, Jzz_coupler]))

s_moment = mass_coupler * L_coupler/2 * np.array([[0, 0, 0],
                                          [0,  0, 1],
                                          [0, -1, 0]])

M_coupler[:3, 3:6] = s_moment
M_coupler[3:6, :3] = s_moment.T

J_coupler = np.zeros((nr_coupler, nr_coupler))
tau_CP = np.eye(nr_coupler)
tau_CP[1, 5] = L_coupler
tau_CP[2, 4] = - L_coupler

B_coupler = np.concatenate((np.eye(nr_coupler), tau_CP.T), axis=1)


coupler = SysPhdaeRig(nr_coupler, 0, nr_coupler, 0, 0, E=M_coupler, J=J_coupler, B=B_coupler)

M_slider = mass_slider * np.eye(nr_slider)

J_slider = np.zeros((nr_slider, nr_slider))
B_slider = np.eye(nr_slider)

slider = SysPhdaeRig(nr_slider, 0, nr_slider, 0, 0, E=M_slider, J=J_slider, B=B_slider)

sys = SysPhdaeRig.transformer_ordered(coupler, slider, [6, 7, 8], [0, 1, 2], np.eye(nr_slider))

n_sys = sys.n
n_e = sys.n_e

E_sys = sys.E
M_sys = E_sys[:n_e, :n_e]
invM_sys = la.inv(M_sys)
J_e = sys.J_e


G_e = sys.G_e

G_coupler = sys.B_e[:, [0, 1, 2]]
G_slider = np.zeros((n_e, 2))
n_tot = n_e + 12 # 3 coupler, 2 slider, 3 coupler-slider and quaternions
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

    p_coupler = M_sys[:3, :] @ e_sys
    pi_coupler = M_sys[3:6, :] @ e_sys
    p_slider = M_sys[nr_coupler:nr_tot, :] @ e_sys

    # J_e[:3, 3:6] = skew(p_coupler)
    # J_e[3:6, :3] = skew(p_coupler)
    # J_e[3:6, 3:6] = skew(pi_coupler)
    # J_e[nr_coupler:nr_tot, 3:6] = skew(p_slider)

    act_quat = np.quaternion(quat_cl[0], quat_cl[1], quat_cl[2], quat_cl[3])
    Rot_cl = quaternion.as_rotation_matrix(act_quat)

    G_slider[nr_coupler:nr_tot, :] = Rot_cl[[1, 2], :].T

    vP_cl = omega_cr * L_crank * np.array([0, -np.cos(omega_cr * t), -np.sin(omega_cr * t)])
    dvP_cl = omega_cr**2 * L_crank * np.array([0, np.sin(omega_cr * t), -np.cos(omega_cr * t)])

    res_sys = M_sys @ de_sys - J_e @ e_sys - G_coupler @ lmd_cl - G_slider @ lmd_mass - G_e @ lmd_sys
    deE_sys = invM_sys @ (J_e @ e_sys + G_coupler @ lmd_cl + G_slider @ lmd_mass + G_e @ lmd_sys)

    res_lmb = G_e.T @ e_sys
    # res_cl = - G_coupler.T @ e_sys + Rot_cl.T @ vP_cl
    res_cl = - Rot_cl @ e_sys[:3] + vP_cl
    res_slider = Rot_cl[[1, 2], :] @ e_sys[nr_coupler:nr_tot]

    # dres_lmb = G_e.T @ deE_sys
    # dres_cl = - G_coupler.T @ deE_sys - skew(omega_cl) @ Rot_cl.T @ vP_cl + Rot_cl.T @ dvP_cl
    # # dres_cl = - Rot_cl @ deE_sys[:3] - Rot_cl @ skew(omega_cl) @ e_sys[:3] + dvP_cl
    #
    # dres_slider = Rot_cl[[1, 2], :] @ skew(omega_cl) @ e_sys[nr_coupler:nr_tot] +\
    #               Rot_cl[[1, 2], :] @ deE_sys[nr_coupler:nr_tot]

    Omegacl_mat = np.array([[0, -omega_cl[0], -omega_cl[1], -omega_cl[2]],
                            [omega_cl[0],   0, omega_cl[2], -omega_cl[1]],
                            [omega_cl[1],   -omega_cl[2], 0, omega_cl[0]],
                            [omega_cl[2],  omega_cl[1], -omega_cl[0], 0]])

    res_quat = dquat_cl - 0.5 * Omegacl_mat @ quat_cl

    res = np.concatenate((res_sys, res_lmb, res_cl, res_slider, res_quat), axis=0)

    return res


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
    b_cl = T_I2B @ dv0P_I - skew(om0_clB) @ T_I2B @ v0P_I
    b_sl = R_B2I[[1, 2], :] @ skew(om0_clB) @ e0_sl

    b_ext = np.concatenate((b_e, b_cl, b_sl), axis=0)
    b_sys = G0_lmb.T @ invM_sys @ J_e @ e0

    A_lmb = G0_lmb.T @ invM_sys @ G0_lmb
    dA_lmb = dG0_lmb.T @ invM_sys @ G0_lmb + G0_lmb.T @ invM_sys @ dG0_lmb

    lmb0 = np.linalg.solve(A_lmb, b_ext - b_sys)
    de0 = invM_sys @ (J_e @ e0 + G0_lmb @ lmb0)

    dom0_cl = de0[3:6]
    de0_sl = de0[6:9]

    db_e = np.zeros((3,))
    db_cl = - skew(om0_clB) @ T_I2B @ dv0P_I + T_I2B @ ddv0P_I - skew(dom0_cl) @ T_I2B @ v0P_I + \
        skew(om0_clB) @ skew(om0_clB) @ T_I2B @ v0P_I - skew(om0_clB) @ T_I2B @ dv0P_I
    db_sl = R_B2I[[1, 2], :] @ skew(om0_clB) @ skew(om0_clB) @ e0_sl + \
            R_B2I[[1, 2], :] @ skew(dom0_cl) @ e0_sl + R_B2I[[1, 2], :] @ skew(om0_clB) @ de0_sl

    db_ext = np.concatenate((db_e, db_cl, db_sl), axis=0)

    db_sys = G0_lmb.T @ invM_sys @ J_e @ de0 + dG0_lmb.T @ invM_sys @ J_e @ e0

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
imp_sim.suppress_alg = True  # Suppress the algebraic variables on the error test
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
quat_cl_sol = quaternion.as_quat_array(y_sol[:, -4:])
e1B_quat = np.quaternion(0, 1, 0, 0)

er_cl_sol = e_sol[:nr_coupler, :]
vP_cl_sol = er_cl_sol[:3, :]
om_cl_sol = er_cl_sol[3:6, :]
vslB_sol = e_sol[nr_coupler:nr_tot, :]

plt.figure()
plt.plot(t_sol, om_cl_sol.T)
plt.xlabel(r"Time")
plt.ylabel(r"Omega")

e1I_sol = np.zeros((3, n_ev))
rP_cl = np.zeros((3, n_ev))

vslI_sol = np.zeros((3, n_ev))
vclCI_sol = np.zeros((3, n_ev))
vclCB_sol = np.zeros((3, n_ev))

PC_skew = skew([-L_coupler, 0, 0])

for i in range(n_ev):
    # e1I_quat = np.multiply(quat_cl_sol[i], np.multiply(e1B_quat, np.conjugate(quat_cl_sol[i])))
    # e1I_sol[:, i] = L_coupler * quaternion.as_float_array(e1I_quat)[1:]
    #
    # vslB_quat = np.quaternion(0, er_sl_sol[0, i], er_sl_sol[1, i], er_sl_sol[2, i])
    # vslI_quat = np.multiply(quat_cl_sol[i], np.multiply(vslB_quat, np.conjugate(quat_cl_sol[i])))

    # vslI_sol[:, i] = quaternion.as_float_array(vslI_quat)[1:]
    vclCB_sol[:, i] = vP_cl_sol[:, i] + PC_skew @ om_cl_sol[:, i]
    vclCI_sol[:, i] = quaternion.as_rotation_matrix(quat_cl_sol[i]) @ vclCB_sol[:, i]

    e1I_sol[:, i] = quaternion.as_rotation_matrix(quat_cl_sol[i]) @ np.array([L_coupler, 0, 0])
    vslI_sol[:, i] = quaternion.as_rotation_matrix(quat_cl_sol[i]) @ vslB_sol[:, i]

    rP_cl[:, i] = np.array([0, -L_crank * np.sin(omega_cr*t_ev[i]), offset_cr + L_crank * np.cos(omega_cr*t_ev[i])])

# plt.plot(t_ev, (G_e.T @ e_sol)[0], 'r', t_ev, (G_e.T @ e_sol)[1], 'b', t_ev, (G_e.T @ e_sol)[2], 'g')

plt.show()

vslI_int = interp1d(t_ev, vslI_sol, kind='linear')
vclCI_int = interp1d(t_ev, vclCI_sol, kind='linear')

def sys(t, y):

    dydt_sl = vslI_int(t)
    dydt_cl = vclCI_int(t)

    return np.concatenate((dydt_sl, dydt_cl))


r_sol = solve_ivp(sys, [0, t_final], np.concatenate((r0C_I, r0C_I)), method='RK45', t_eval=t_ev)


n_ev = len(t_sol)
dt_vec = np.diff(t_sol)

xP_sl = r_sol.y[0, :]
yP_sl = r_sol.y[1, :]
zP_sl = r_sol.y[2, :]

xC_cl = r_sol.y[3, :]
yC_cl = r_sol.y[4, :]
zC_cl = r_sol.y[5, :]

# rC_cl = rP_cl + e1I_sol
# xC_cl = rC_cl[0]
# yC_cl = rC_cl[1]
# zC_cl = rC_cl[2]

data = np.array([[rP_cl[0], rP_cl[1], rP_cl[2]], [xC_cl, yC_cl, zC_cl], [xP_sl, yP_sl, zP_sl]])
print(data.shape)

fntsize = 16

anim = animate_line3d(data, t_ev)

plt.show()
