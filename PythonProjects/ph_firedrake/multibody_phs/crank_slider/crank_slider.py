import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
fntsize = 15

import numpy as np
import scipy.linalg as la
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from modules_phdae.classes_phsystem import SysPhdaeRig
from system_components.beams import FloatFlexBeam, draw_deformation
from math import pi

from assimulo.solvers import IDA
from assimulo.implicit_ode import Implicit_Problem

L_crank = 0.15
L_coupler = 0.3
d = 0.006
A_coupler = pi * d**2 / 4
I_coupler = pi * d**4 / 64


rho_coupler = 7.87 * 10 ** 3
E_coupler = 200 * 10**9

# L_crank = 150
# L_coupler = 300
# d = 6
# A_coupler = pi * d**2 / 4
# I_coupler = pi * d**4 / 64
#
#
# rho_coupler = 7.87 * 10 ** (-6)
# E_coupler = 200 * 10**3

omega_cr = 150

nr_coupler = 3
nr_mass = 2
nr_tot = nr_coupler + nr_mass

n_elem = 2
coupler = FloatFlexBeam(n_elem, L_coupler, rho_coupler, A_coupler, E_coupler, I_coupler)

mass_coupler = rho_coupler * A_coupler * L_coupler
M_mass = 0.5 * la.block_diag(mass_coupler, mass_coupler)
J_mass = np.zeros((nr_mass, nr_mass))
B_mass = np.eye(nr_mass)

mass = SysPhdaeRig(nr_mass, 0, nr_mass, 0, 0, E=M_mass, J=J_mass, B=B_mass)

sys = SysPhdaeRig.transformer_ordered(coupler, mass, [3, 4], [0, 1], np.eye(2))

# plt.spy(sys.E); plt.show()

M_sys = sys.E
J_sys = sys.J
G_coupler = sys.B[:, [0, 1]]

n_sys = sys.n
n_e = sys.n_e
n_p = sys.n_p
n_pu = int(n_p/3)
n_pw = n_p - n_pu

n_tot = n_sys + 4  # 3 lambda et theta
order = []


def dae_closed_phs(t, y, yd):

    yd_sys = yd[:n_sys]
    y_sys = y[:n_sys]
    lmd_cl = y[n_sys:-2]
    lmd_mass = y[-2]
    theta_cl = y[-1]
    omega_cl = y[2]

    p_coupler = M_sys[:2, :n_e] @ y_sys[:n_e]
    p_mass = M_sys[nr_coupler:nr_tot, :n_e] @ y_sys[:n_e]

    p_u = M_sys[nr_tot:nr_tot+n_pu, :n_e] @ y_sys[:n_e]
    p_w = M_sys[nr_tot+n_pu:nr_tot+n_p, :n_e] @ y_sys[:n_e]

    p_wdis = np.array([p_w[i] for i in range(len(p_w)) if i % 2 == 0])
    p_udis = np.zeros_like(p_w)
    p_udis[::2] = p_u

    J_sys[:2, 2] = [+p_coupler[1], -p_coupler[0]]
    J_sys[2, :2] = [-p_coupler[1], +p_coupler[0]]
    J_sys[3:5, 2] = [+p_mass[1], -p_mass[0]]
    J_sys[nr_tot:nr_tot+n_p, 2] = np.concatenate((p_wdis, -p_udis))
    J_sys[2, nr_tot:nr_tot+n_p] = np.concatenate((-p_wdis, +p_udis))

    R_th = np.array([[np.cos(theta_cl), -np.sin(theta_cl)],
                    [np.sin(theta_cl), np.cos(theta_cl)]])

    G_mass = np.zeros(n_sys)
    G_mass[nr_coupler:nr_tot] = R_th[1, :]

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
t_final = 8/150
n_ev = 2000
t_ev = np.linspace(0, t_final, n_ev)
t_sol, y_sol, yd_sol = imp_sim.simulate(t_final, 0, t_ev)
e_sol = y_sol[:, :n_e].T
lmb_sol = y_sol[:, n_e:-1].T
th_cl_sol = y_sol[:, -1]

er_cl_sol = e_sol[:nr_coupler, :]
ep_sol = e_sol[nr_tot:nr_tot + n_p, :]
om_cl_sol = er_cl_sol[2, :]


nw = int(2*n_p/3)
nu = int(n_p/3)

up_sol = ep_sol[:nu, :]
wp_sol = ep_sol[nu:, :]

n_ev = len(t_sol)
dt_vec = np.diff(t_sol)

n_plot = 11
eu_plot = np.zeros((n_plot, n_ev))
ew_plot = np.zeros((n_plot, n_ev))

x_plot = np.linspace(0, L_coupler, n_plot)
w_plot = np.zeros((n_plot, n_ev))

t_plot = t_sol

for i in range(n_ev):
    eu_plot[:, i], ew_plot[:, i] = draw_deformation(n_plot, [0, 0, 0], ep_sol[:, i], L_coupler)[1:3]

ind_midpoint = int((n_plot-1)/2)

euM_B = eu_plot[ind_midpoint, :]
ewM_B = ew_plot[ind_midpoint, :]

omega_cr_int = interp1d(t_ev, om_cl_sol, kind='linear')
euM_B_int = interp1d(t_ev, euM_B, kind='linear')
ewM_B_int = interp1d(t_ev, ewM_B, kind='linear')

def sys(t,y):

    dudt = euM_B_int(t) #+ omega_cr_int(t) * y[1]
    dwdt = ewM_B_int(t) #- omega_cr_int(t) * y[0]

    dydt = np.array([dudt, dwdt])
    return dydt


wM0 = 0
uM0 = 0

r_sol = solve_ivp(sys, [0, t_final], [uM0, wM0], method='RK45', t_eval=t_ev)

uM_B = r_sol.y[0, :]
wM_B = r_sol.y[1, :]

fntsize = 16
fig = plt.figure()
plt.plot(omega_cr*t_ev, -wM_B/L_coupler, 'b-')
plt.xlabel(r'Crank angle [rad]', fontsize = fntsize)
plt.ylabel(r'w normalized', fontsize = fntsize)
plt.title(r"Midpoint deflection", fontsize=fntsize)
axes = plt.gca()
# axes.set_ylim([-0.015, 0.02])

plt.show()
# plt.legend(loc='upper left')