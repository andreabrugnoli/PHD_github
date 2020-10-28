import matplotlib.pyplot as plt
plt.rc('text', usetex=True)

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

import numpy as np
import scipy.linalg as la
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from modules_ph.classes_phsystem import SysPhdaeRig
from system_components.beams import FloatFlexBeam, draw_deformation, matrices_j2d
from math import pi
from assimulo.solvers import IDA
from assimulo.solvers import Radau5DAE
from assimulo.implicit_ode import Implicit_Problem
import time

n_elem = 2

# L_crank = 0.15
# L_coupler = 0.3

L_crank = 0.152
L_coupler = 0.304
rho_coupler = 7.83 * 10 ** 3
E_coupler = 206.8 * 10**9


d = 0.006
A_coupler = pi * d**2 / 4
I_coupler = pi * d**4 / 64


# rho_coupler = 7.87 * 10 ** 3
# E_coupler = 0.2 * 10**12

omega_cr = 150
# omega_cr = 124.8


# L_crank = 150
# L_coupler = 300
# d = 6
# A_coupler = pi * d**2 / 4
# I_coupler = pi * d**4 / 64
#
#
# rho_coupler = 7.87 * 10 ** (-6)
# E_coupler = 200 * 10**3


nr_coupler = 3
nr_mass = 2
nr_tot = nr_coupler + nr_mass

bc = 'SS'
coupler = FloatFlexBeam(n_elem, L_coupler, rho_coupler, A_coupler, E_coupler, I_coupler, bc=bc)

mass_coupler = rho_coupler * A_coupler * L_coupler
M_mass = 0.5 * la.block_diag(mass_coupler, mass_coupler)
J_mass = np.zeros((nr_mass, nr_mass))
B_mass = np.eye(nr_mass)

mass = SysPhdaeRig(nr_mass, 0, nr_mass, 0, 0, E=M_mass, J=J_mass, B=B_mass)

sys = SysPhdaeRig.transformer_ordered(coupler, mass, [3, 4], [0, 1], np.eye(2))

Jf_tx, Jf_ty, Jf_rz, Jf_fx, Jf_fy = matrices_j2d(n_elem, L_coupler, rho_coupler, A_coupler, bc=bc)

M_sys = sys.E
J_sys = sys.J
G_coupler = sys.B[:, [0, 1]]

n_sys = sys.n
n_e = sys.n_e
n_p = sys.n_p
n_pu = int(n_p/3)
n_pw = n_p - n_pu

# Mxx = M_sys[nr_tot:nr_tot+n_pu, nr_tot:nr_tot+n_pu]
# Myy = M_sys[nr_tot+n_pu:nr_tot+n_p, nr_tot+n_pu:nr_tot+n_p]
#
# M_gyro = np.zeros((n_sys, n_pw + n_pu))
# M_gyro[0, :n_pw] = Jf_tx
# M_gyro[0, n_pw:] = Jf_ty
# M_gyro[1, :n_pw] = Jf_tx
# M_gyro[1, n_pw:] = Jf_ty
#
# M_gyro[2, :n_pw] = M_sys[nr_tot + n_pu:nr_tot + n_p, 2]
# M_gyro[2, n_pw:] = Jf_rz
#
# M_gyro[nr_tot:nr_tot + n_pu, :n_pw] = Jf_fy
# M_gyro[nr_tot:nr_tot + n_pu, n_pw:] = Mxx
#
# M_gyro[nr_tot + n_pu:nr_tot + n_p, :n_pw] = Myy
# M_gyro[nr_tot + n_pu:nr_tot + n_p, n_pw:] = Jf_fx

n_tot = n_sys + 4  # 3 lambda et theta
order = []
t_final = 8/omega_cr

t_step = []


def dae_closed_phs(t, y, yd):

    print(t/t_final*100)

    t_step.append(t)

    yd_sys = yd[:n_sys]
    y_sys = y[:n_sys]
    lmd_cl = y[n_sys:-2]
    lmd_mass = y[-2]
    theta_cl = y[-1]
    omega_cl = y[2]

    p_coupler = M_sys[:2, :n_e] @ y_sys[:n_e] + M_sys[:2, nr_tot:n_e] @ y_sys[nr_tot:n_e]
    p_mass = M_sys[nr_coupler:nr_tot, :n_e] @ y_sys[:n_e]

    vxP_coupler = y_sys[0]
    vyP_coupler = y_sys[1]
    omzP_coupler = y_sys[2]

    eu_coupler = y_sys[nr_tot:nr_tot+n_pu]
    ew_coupler = y_sys[nr_tot+n_pu:nr_tot+n_p]

    jf_u = (Jf_tx * vxP_coupler + Jf_fx @ eu_coupler)
    jf_w = Jf_ty * vyP_coupler + Jf_rz * omzP_coupler + Jf_fy @ ew_coupler

    jf_u_cor = Jf_fx @ eu_coupler
    jf_w_cor = Jf_fy @ ew_coupler

    J_sys[:2, 2] = [+p_coupler[1], -p_coupler[0]]
    J_sys[2, :2] = [-p_coupler[1], +p_coupler[0]]
    J_sys[3:5, 2] = [+p_mass[1], -p_mass[0]]

    J_sys[nr_tot:nr_tot + n_p, 2] = np.concatenate((jf_w, -jf_u)) + np.concatenate((jf_w_cor, -jf_u_cor))
    J_sys[2, nr_tot:nr_tot + n_p] = 2*np.concatenate((-jf_w, +jf_u))

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

# y0[nr_tot:nr_tot + n_pu] = 2.2*10**(-5) * np.array(list(range(1, n_elem+1)))
y0[-4] = - mass_coupler * omega_cr ** 2 * L_crank
yd0[0] = - omega_cr ** 2 * L_crank

# Create an Assimulo implicit problem
imp_mod = Implicit_Problem(dae_closed_phs, y0, yd0, name='dae_closed_pHs')
imp_mod.handle_result = handle_result

# Set the algebraic components
imp_mod.algvar = list(np.concatenate((np.ones(n_e), np.zeros(n_tot - n_e - 1), np.ones(1))))

# Create an Assimulo implicit solver (IDA)
imp_sim = IDA(imp_mod)  # Create a IDA solver
# imp_sim = Radau5DAE(imp_mod)  # Create a IDA solver

# Sets the paramters
imp_sim.atol = 1e-6  # Default 1e-6
imp_sim.rtol = 1e-6  # Default 1e-6
imp_sim.suppress_alg = True  # Suppress the algebraic variables on the error test
imp_sim.report_continuously = True
imp_sim.maxh = 1e-6

# Let Sundials find consistent initial conditions by use of 'IDA_YA_YDP_INIT'
imp_sim.make_consistent('IDA_YA_YDP_INIT')

# Simulate
n_ev = 5000
t_ev = np.linspace(0, t_final, n_ev)

ti_sim = time.time()
t_sol, y_sol, yd_sol = imp_sim.simulate(t_final, 0, t_ev)
tf_sim = time.time()

elapsed_t = tf_sim - ti_sim

dt_vec = np.diff(t_step)

dt_avg = np.mean(dt_vec)

print("elapsed: " + str(elapsed_t))
print("dt_avg: " + str(dt_avg))

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

assert n_elem % 2 == 0

if bc == 'CF':
    euM_B = up_sol[int(n_elem / 2) - 1, :]
    ewM_B = wp_sol[n_elem - 2, :]
else:
    euM_B = up_sol[int(n_elem / 2) - 1, :]
    ewM_B = wp_sol[n_elem - 1, :]


path_fig = "/home/a.brugnoli/Plots/Python/Plots/Multibody_PH/CrankSlider/"

H_vec = np.zeros((n_ev,))
for i in range(n_ev):
    H_vec[i] = 0.5 * (e_sol[:, i].T @ M_sys[:n_e, :n_e] @ e_sol[:, i])

fig = plt.figure()
plt.plot(t_ev, H_vec, 'b-')
plt.xlabel(r'{Time} (s)')
plt.ylabel(r'{Hamiltonian} (J)')
plt.title(r"Hamiltonian crank slider")
# plt.savefig(path_fig + 'Hamiltonian.eps', format="eps")

# n_ev = len(t_sol)
# dt_vec = np.diff(t_sol)
#
# n_plot = 21
# eu_plot = np.zeros((n_plot, n_ev))
# ew_plot = np.zeros((n_plot, n_ev))
#
# x_plot = np.linspace(0, L_coupler, n_plot)
# w_plot = np.zeros((n_plot, n_ev))
#
# t_plot = t_sol
#
# for i in range(n_ev):
#     eu_plot[:, i], ew_plot[:, i] = draw_deformation(n_plot, [0, 0, 0], ep_sol[:, i], L_coupler)[1:3]
#
# ind_midpoint = int((n_plot-1)/2)
#
# euM_B = eu_plot[ind_midpoint, :]
# ewM_B = ew_plot[ind_midpoint, :]

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

r_sol = solve_ivp(sys, [0, t_final], [uM0, wM0], method='RK45', t_eval=t_sol)

uM_B = r_sol.y[0, :]
wM_B = r_sol.y[1, :]


fig = plt.figure()
plt.plot(omega_cr*t_ev, om_cl_sol, 'b-')
plt.xlabel(r'Crank angle $\mathrm{[rad]}$')
plt.ylabel(r'$\omega^z_{cl}$ [rad/s]')
plt.title(r"Angular velocity coupler")

fig = plt.figure()
plt.plot(omega_cr*t_ev, uM_B/L_coupler, 'b-')
plt.xlabel(r'Crank angle $\mathrm{[rad]}$')
plt.ylabel(r'$u_f^x/L_{cl}$')
plt.title(r"Midpoint horizontal deflection")
# plt.savefig(path_fig + 'uM_disp.eps', format="eps")

fig = plt.figure()
plt.plot(omega_cr*t_ev, wM_B/L_coupler, 'b-')
plt.xlabel(r'Crank angle $\mathrm{[rad]}$')
plt.ylabel(r'$u_f^y/L_{cl}$')
plt.title(r"Midpoint vertical deflection")
axes = plt.gca()
axes.set_ylim([-0.015, 0.02])
# plt.savefig(path_fig + 'wM_disp.eps', format="eps")

plt.show()
# plt.legend(loc='upper left')