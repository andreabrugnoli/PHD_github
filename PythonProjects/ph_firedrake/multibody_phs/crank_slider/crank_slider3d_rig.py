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


L_crank = 0.15
L_coupler = 0.3
d = 0.006

ecc = 0.1
rho_coupler = 7.87 * 10 ** 3
E_coupler = 200 * 10**9
A_coupler = pi * d**2 / 4
I_coupler = pi * d**4 / 64
mass_coupler = rho_coupler * A_coupler * L_coupler
mass_slider = 0.5 * mass_coupler
Jxx_coupler = 2 * I_coupler * rho_coupler
Jyy_coupler = 1 / 3 * mass_coupler * L_coupler ** 2
Jzz_coupler = 1 / 3 * mass_coupler * L_coupler ** 2
n_elem = 2

omega_cr = 150

nr_coupler = 6
nr_slider = 3
nr_tot = nr_coupler + nr_slider

M_coupler = la.block_diag(mass_coupler * np.eye(3), np.diag([Jxx_coupler, Jyy_coupler, Jzz_coupler]))

s = mass_coupler* L_coupler/2 * np.array([[0, 0, 0],
                                    [0,  0, 1],
                                    [0, -1, 0]])

M_coupler[:3, 3:6] = s
M_coupler[3:6, :3] = s.T

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

# plt.spy(sys.E); plt.show()

n_sys = sys.n
n_e = sys.n_e

E_sys = sys.E
M_sys = E_sys[:n_e, :n_e]
invM_sys = la.inv(M_sys)
J_e = sys.J_e


G_e = sys.G_e
G_coupler = sys.B_e[:, [0, 1, 2]]
G_slider = np.zeros((n_e, 2))

n_tot = n_e + 12  # 2*8 lambda (3 coupler, 2 slider, 3 coupler-slider) and quaternions
order = []

dx = L_coupler/n_elem

t_final = 2*pi/150


def dae_closed_phs(t, y, yd):

    print(t/t_final*100)

    # dy_sys = yd[:n_sys]
    # y_sys = y[:n_sys]

    de_sys = yd[:n_e]
    dquat_cl = yd[-4:]

    e_sys = y[:n_e]

    lmd_sys = y[n_e:n_sys]
    lmd_cl = y[n_sys:n_sys + 3]
    lmd_mass = y[n_sys + 3: n_sys + 5]

    quat_cl = y[-4:]/np.linalg.norm(y[-4:])

    omega_cl = y[3:6]

    p_coupler = M_sys[:3, :] @ e_sys
    pi_coupler = M_sys[3:6, :] @ e_sys
    p_slider = M_sys[nr_coupler:nr_tot, :] @ e_sys

    J_e[:3, 3:6] = skew(p_coupler)
    J_e[3:6, :3] = skew(p_coupler)
    J_e[3:6, 3:6] = skew(pi_coupler)
    J_e[nr_coupler:nr_tot, 3:6] = skew(p_slider)

    act_quat = np.quaternion(quat_cl[0], quat_cl[1], quat_cl[2], quat_cl[3])
    Rot_cl = quaternion.as_rotation_matrix(act_quat)
    # Rot_cl = Rotation.from_quat(quat_cl).as_dcm()
    # print(Rotation.from_dcm(Rot_cl).as_euler('ZYX', degrees=True))

    G_slider[nr_coupler:nr_tot, :] = Rot_cl[[1, 2], :].T

    vP_cl = omega_cr * L_crank * np.array([0, -np.cos(omega_cr * t), -np.sin(omega_cr * t)])
    dvP_cl = omega_cr**2 * L_crank * np.array([0, np.sin(omega_cr * t), -np.cos(omega_cr * t)])

    res_sys = M_sys @ de_sys - J_e @ e_sys - G_coupler @ lmd_cl - G_slider @ lmd_mass - G_e @ lmd_sys
    deE_sys = invM_sys @ (J_e @ e_sys + G_coupler @ lmd_cl + G_slider @ lmd_mass + G_e @ lmd_sys)

    dres_lmb = G_e.T @ deE_sys
    dres_cl = - G_coupler.T @ deE_sys - skew(omega_cl) @ Rot_cl.T @ vP_cl + Rot_cl.T @ dvP_cl
    dres_slider = Rot_cl[[1, 2], :] @ skew(omega_cl) @ e_sys[nr_coupler:nr_tot] + Rot_cl[[1, 2], :] @ deE_sys[nr_coupler:nr_tot]

    res_lmb = G_e.T @ e_sys
    res_cl = - G_coupler.T @ e_sys + Rot_cl.T @ vP_cl
    res_slider = Rot_cl[[1, 2], :] @ e_sys[nr_coupler:nr_tot]

    Omegacl_mat = np.array([[0, -omega_cl[0], -omega_cl[1], -omega_cl[2]],
                            [omega_cl[0],   0, omega_cl[2], -omega_cl[1]],
                            [omega_cl[1],   -omega_cl[2], 0, omega_cl[0]],
                            [omega_cl[2],  omega_cl[1], -omega_cl[0], 0]])

    res_quat = dquat_cl - 0.5 * Omegacl_mat @ quat_cl

    res = np.concatenate((res_sys, dres_lmb, dres_cl, dres_slider, res_quat), axis=0)

    return res


def handle_result(solver, t, y, yd):

    order.append(solver.get_last_order())

    solver.t_sol.extend([t])
    solver.y_sol.extend([y])
    solver.yd_sol.extend([yd])


# alpha1 = np.arctan2(L_crank, ecc)
# alpha2 = np.arcsin(np.sqrt(ecc**2 + L_crank**2)/L_coupler)
#
# Rot_alpha1 = np.array([[1, 0, 0],
#                        [0, np.cos(alpha1), -np.sin(alpha1)],
#                        [0, np.sin(alpha1), np.cos(alpha1)]])
#
# Rot_alpha2 = np.array([[np.cos(alpha2), -np.sin(alpha2), 0],
#                        [np.sin(alpha2), +np.cos(alpha2), 0],
#                        [0, 0, 1]])
#
# T_I2B = Rot_alpha2 @ Rot_alpha1.T

theta1 = np.arcsin(ecc/np.sqrt(L_coupler**2 - L_crank**2))
theta2 = np.arcsin(L_crank/L_coupler)

Rot_theta1 = np.array([[np.cos(theta1), -np.sin(theta1), 0],
                       [np.sin(theta1), +np.cos(theta1), 0],
                       [0, 0, 1]])

Rot_theta2 = np.array([[np.cos(theta2), 0,  np.sin(theta2)],
                       [0, 1, 0],
                       [-np.sin(theta2), 0, np.cos(theta2)]])

T_I2B = Rot_theta2.T @ Rot_theta1

print(theta1*180/pi, theta2*180/pi)
print(Rotation.from_dcm(T_I2B.T).as_euler('ZYX', degrees=True))

y0 = np.zeros(n_tot)  # Initial conditions
yd0 = np.zeros(n_tot)  # Initial conditions

v0P_I = np.array([0, -omega_cr * L_crank, 0])
rP_I = np.array([0, 0, L_crank])

xC_I = np.sqrt(L_coupler**2 - L_crank**2 - ecc**2)
rC_I = np.array([xC_I, -ecc, 0])

dir_couplerI = rP_I - rC_I

dir_om_cl = np.cross(dir_couplerI, v0P_I)
dir_om_cl = dir_om_cl/np.linalg.norm(dir_om_cl)

om_cl_norm = omega_cr * L_crank / L_coupler
om_clI = om_cl_norm * dir_om_cl

y0[:3] = T_I2B @ v0P_I
y0[3:6] = T_I2B @ om_clI
quat_1 = quaternion.as_float_array(quaternion.from_rotation_matrix(T_I2B.T))
quat_2 = Rotation.from_dcm(T_I2B.T).as_quat()

print(quat_1, quat_2)
y0[-4:] = quat_1


y0[-9:-6] = T_I2B @ np.array([0, 0, - mass_coupler * omega_cr ** 2 * L_crank])
yd0[:3] = T_I2B @ np.array([0, 0, - omega_cr ** 2 * L_crank])

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
quat_cl_sol = y_sol[:, -4:]

euler_angles = Rotation.from_quat(quat_cl_sol).as_euler('ZYX', degrees=True)

er_cl_sol = e_sol[:nr_coupler, :]
om_cl_sol = er_cl_sol[3:6, :]


n_ev = len(t_sol)
dt_vec = np.diff(t_sol)


fntsize = 16

fig = plt.figure()
plt.plot(omega_cr*t_ev*180/pi, euler_angles[:,0], 'r-', label="angle z")
plt.plot(omega_cr*t_ev*180/pi, euler_angles[:,1], 'b', label="angle y")
plt.plot(omega_cr*t_ev*180/pi, euler_angles[:,2], 'g-', label="angle x")
plt.legend(loc='upper left')
axes = plt.gca()
plt.xlabel(r'Crank angle [deg]', fontsize = fntsize)
plt.ylabel(r'Angles', fontsize = fntsize)
plt.title(r"ZYX", fontsize=fntsize)

plt.show()

