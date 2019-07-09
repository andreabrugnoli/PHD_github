import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
fntsize = 15

import numpy as np
import scipy.linalg as la

from modules_phdae.classes_phsystem import SysPhdaeRig
from system_components.beams import FloatFlexBeam, draw_deformation
from math import pi

L_crank = 0.15
L_coupler = 0.3
d = 0.006
A_coupler = pi * d**2 / 4
I_coupler = pi * d**4 / 64
n_elem = 4
rho_coupler = 7.87 * 10 ** 3
E_coupler = 0.2 * 10**12

coupler = FloatFlexBeam(n_elem, L_coupler, rho_coupler, A_coupler, E_coupler, I_coupler)

mass_coupler = rho_coupler * A_coupler * L_coupler
M_payload = la.block_diag(mass_coupler, mass_coupler)
J_payload = np.zeros((2, 2))
B_payload = np.eye(2)

payload = SysPhdaeRig(2, 0, 2, 0, 0, E=M_payload, J=J_payload, B=B_payload)

sys = SysPhdaeRig.transformer_ordered(coupler, payload, [3, 4], [0, 1], np.eye(2))

plt.spy(sys.E); plt.show()

M_sys = sys.E
J_sys = sys.J
G_coupler = sys.B[:, [0, 1]]

n_sys = sys.n


order = []


def dae_closed_phs(t, y, yd):

    y_sys = y[:-4]
    yd_sys = yd[:-4]

    lmd_cl = y[-4:-2]
    lmd_pl = y[-2]
    theta = y[-1]

    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    res_sys = M_sys @ yd_sys - J_sys @ y_sys - G_coupler @ lmd_cl

    return res

    # res_e = E_aug[:n_e, :] @ yd - J_aug[:n_e, :] @ y - B_aug[:n_e] * u
    # res_lmb = G.T @ invMM @ (J_aug[:n_e, :] @ y + B_aug[:n_e] * u)
    #
    # return np.concatenate((res_e, res_lmb))


def handle_result(solver, t, y, yd):

    order.append(solver.get_last_order())

    solver.t_sol.extend([t])
    solver.y_sol.extend([y])
    solver.yd_sol.extend([yd])

    # The initial conditons


y0 = np.zeros(n_e + n_lmb)  # Initial conditions
yd0 = np.zeros(n_e + n_lmb)  # Initial conditions

# Create an Assimulo implicit problem
imp_mod = Implicit_Problem(dae_closed_phs, y0, yd0, name='dae_closed_pHs')
imp_mod.handle_result = handle_result

# Set the algebraic components
imp_mod.algvar = list(np.concatenate((np.ones(n_e), np.zeros(n_lmb))))

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
t_final = 1
n_ev = 1000
t_ev = np.linspace(0, t_final, n_ev)
t_sol, y_sol, yd_sol = imp_sim.simulate(t_final, 0, t_ev)
e_sol = y_sol[:, :n_e].T
lmb_sol = y_sol[:, n_e:].T