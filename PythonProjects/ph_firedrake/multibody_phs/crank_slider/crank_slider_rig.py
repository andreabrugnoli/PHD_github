import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
fntsize = 15

import numpy as np
import scipy.linalg as la

from modules_phdae.classes_phsystem import SysPhdaeRig
from math import pi

from assimulo.solvers import IDA
from assimulo.implicit_ode import Implicit_Problem

# L_crank = 0.2
# L_coupler = 0.5

# rpm = 60
# omega_cr = rpm * 2*pi/60
L_crank = 0.15
L_coupler = 0.3
omega_cr = 150

mass_coupler = 3
mass_slider = 2
J_coupler = 1/3 * mass_coupler * L_coupler**2

nr_coupler = 3
nr_slider = 2
nr_tot = nr_coupler + nr_slider

M_coupler = np.array([[mass_coupler, 0, 0],
                      [0, mass_coupler, mass_coupler * L_coupler/2],
                      [0, mass_coupler * L_coupler/2, mass_coupler]])

J_coupler = np.zeros((nr_coupler, nr_coupler))
tau_CP = np.array([[1, 0, 0], [0, 1, L_coupler], [0, 0, 1]])
B_coupler = np.concatenate((np.eye(nr_coupler), tau_CP.T), axis=1)

coupler = SysPhdaeRig(nr_coupler, 0, nr_coupler, 0, 0, E=M_coupler, J=J_coupler, B=B_coupler)

M_slider = mass_slider*np.eye(nr_slider)
J_slider = np.zeros((nr_slider, nr_slider))
B_slider = np.eye(nr_slider)

mass = SysPhdaeRig(nr_slider, 0, nr_slider, 0, 0, E=M_slider, J=J_slider, B=B_slider)

sys = SysPhdaeRig.transformer_ordered(coupler, mass, [3, 4], [0, 1], np.eye(2))

# plt.spy(sys.E); plt.show()

M_sys = sys.E
J_sys = sys.J
G_coupler = sys.B[:, [0, 1]]

n_sys = sys.n
n_e = sys.n_e
n_tot = n_sys + 4  # 3 lambda et theta
order = []


def dae_closed_phs(t, y, yd):

    yd_sys = yd[:n_sys]
    y_sys = y[:n_sys]
    lmd_cl = y[n_sys:-2]
    lmd_mass = y[-2]
    theta_cl = y[-1]
    omega_cl = y[2]

    p_crank = M_sys[:2, :3] @ y_sys[:3]
    p_mass = M_sys[3:5, 3:5] @ y_sys[3:5]

    J_sys[:2, 2] = [+p_crank[1], -p_crank[0]]
    J_sys[2, :2] = [-p_crank[1], +p_crank[0]]
    J_sys[3:5, 2] = [+p_mass[1], -p_mass[0]]

    R_th = np.array([[np.cos(theta_cl), -np.sin(theta_cl)],
                    [np.sin(theta_cl), np.cos(theta_cl)]])

    G_mass = np.zeros(n_sys)
    G_mass[nr_coupler:nr_coupler+2] = R_th[1, :]

    vC_cr = omega_cr * L_crank * np.array([-np.sin(omega_cr * t), np.cos(omega_cr * t)])

    res_sys = M_sys @ yd_sys - J_sys @ y_sys - G_coupler @ lmd_cl - G_mass * lmd_mass
    res_cl = - G_coupler.T @ y_sys + R_th.T @ vC_cr
    res_mass = np.reshape(G_mass, (1, -1)) @ y_sys
    res_th = np.reshape(yd[-1] - omega_cl, (1, ))

    res = np.concatenate((res_sys, res_cl, res_mass, res_th), axis=0)
    return res


def handle_result(solver, t, y, yd):

    order.append(solver.get_last_order())

    solver.t_sol.extend([t])
    solver.y_sol.extend([y])
    solver.yd_sol.extend([yd])

    # The initial conditons


y0 = np.zeros(n_tot)  # Initial conditions
yd0 = np.zeros(n_tot)  # Initial conditions

y0[1] = omega_cr * L_crank
y0[2] = - omega_cr * L_crank / L_coupler
y0[-4] = - mass_coupler * omega_cr ** 2 * L_crank
yd0[0] = - omega_cr ** 2 * L_crank
# Create an Assimulo implicit problem
imp_mod = Implicit_Problem(dae_closed_phs, y0, yd0, name='dae_closed_pHs')
imp_mod.handle_result = handle_result

# Set the algebraic components
imp_mod.algvar = list(np.concatenate((np.ones(n_e), np.zeros(n_tot - n_e - 1), np.ones(1))))

# Create an Assimulo implicit solver (IDA)
imp_sim = IDA(imp_mod)  # Create a IDA solver

# Sets the paramters
imp_sim.atol = 1e-6  # Default 1e-6
imp_sim.rtol = 1e-6  # Default 1e-6
imp_sim.suppress_alg = True  # Suppress the algebraic variables on the error test
imp_sim.report_continuously = True
# imp_sim.maxh = 1e-6

# Let Sundials find consistent initial conditions by use of 'IDA_YA_YDP_INIT'
imp_sim.make_consistent('IDA_YA_YDP_INIT')

# Simulate
t_final = 8/omega_cr
n_ev = 1000
t_ev = np.linspace(0, t_final, n_ev)
t_sol, y_sol, yd_sol = imp_sim.simulate(t_final, 0, t_ev)
e_sol = y_sol[:, :n_e].T
lmb_sol = y_sol[:, n_e:-1].T

th_cl_sol = y_sol[:, -1]
ercl_sol = e_sol[:nr_coupler, :]
ersl_sol = e_sol[nr_coupler:nr_tot, :]
om_cl_sol = ercl_sol[2, :]

n_ev = len(t_sol)
dt_vec = np.diff(t_sol)

ersl_I = np.zeros_like(ersl_sol)
wsl_I = np.zeros_like(ersl_sol)
wsl_I[:, 0] = [L_coupler + L_crank, 0]

for i in range(n_ev):
    theta_cl = th_cl_sol[i]
    R_th = np.array([[np.cos(theta_cl), -np.sin(theta_cl)],
                     [np.sin(theta_cl), np.cos(theta_cl)]])
    ersl_I[:, i] = R_th @ ersl_sol[:, i]

    if i>0:
        wsl_I[:, i] = wsl_I[:, i-1] + 0.5 * (ersl_I[:, i] + ersl_I[:, i-1]) * dt_vec[i-1]

fntsize = 16

fig = plt.figure()
plt.plot(omega_cr*t_ev, om_cl_sol, 'b-')
plt.xlabel(r'Crank angle [rad]', fontsize = fntsize)
plt.ylabel(r'Omega coupler [rad/s]', fontsize = fntsize)
plt.title(r"Omega coupler", fontsize=fntsize)

# fig = plt.figure()
# plt.plot(t_ev, ersl_I[0,:], 'b-', t_ev, ersl_I[1,:], 'r-')
# plt.xlabel(r'{Time} (s)', fontsize = fntsize)
# plt.ylabel(r'$v$ slider', fontsize = fntsize)
# plt.title(r"Velocity slider", fontsize=fntsize)
#
# fig = plt.figure()
# plt.plot(t_ev, wsl_I[0,:], 'b-', t_ev, wsl_I[1,:], 'r-')
# plt.xlabel(r'{Time} (s)', fontsize = fntsize)
# plt.ylabel(r'$w$ slider', fontsize = fntsize)
# plt.title(r"Displacement slider", fontsize=fntsize)

plt.show()
# plt.legend(loc='upper left')