import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import sys
from modules_phdae.classes_phsystem import SysPhdaeRig
from system_components.beams import FloatingPlanarEB
from math import pi
from scipy import integrate
from scipy.io import savemat
from system_components.tests.manipulator_constants import n_el, rho1, EI1, L1, rho2, EI2, L2, n_rig, J_joint1, J_joint2, J_payload, m_joint2, m_payload


beam1 = FloatingPlanarEB(n_el, L1, rho1, 1, EI1, 1, J_joint=J_joint1)
E_hinged = beam1.E[2:, 2:]
J_hinged = beam1.J[2:, 2:]
B_hinged = beam1.B[2:, 2:]
beam1_hinged = SysPhdaeRig(len(E_hinged), 0, 1, beam1.n_p, beam1.n_q,
                           E=E_hinged, J=J_hinged, B=B_hinged)

beam2 = FloatingPlanarEB(n_el, L2, rho2, 1, EI2, 1, m_joint=m_joint2, J_joint=J_joint2)

M_payload = la.block_diag(m_payload, m_payload, J_payload)
J_payload = np.zeros((n_rig, n_rig))
B_payload = np.eye(n_rig)

payload = SysPhdaeRig(n_rig, 0, n_rig, 0, 0, E=M_payload, J=J_payload, B=B_payload)

ind1 = np.array([1, 2], dtype=int)
ind2_int1 = np.array([0, 1], dtype=int)
n_int = 5 # after first interconnection and pivot
ind2_int2 = np.array([n_int - 3, n_int - 2, n_int - 1], dtype=int)
ind3 = np.array([0, 1, 2], dtype=int)


def build_man(theta2):

    R = np.array([[np.cos(theta2), np.sin(theta2)],
                  [-np.sin(theta2), np.cos(theta2)]])
    sys_int1 = SysPhdaeRig.transformer_ordered(beam1_hinged, beam2, ind1, ind2_int1, R)

    sys_int1.pivot(2, 1)

    sys_dae = SysPhdaeRig.transformer_ordered(sys_int1, payload, ind2_int2, ind3, np.eye(3))

    sys_ode, T, M_ode = sys_dae.dae_to_odeCE()

    J_ode = sys_ode.J
    Q_ode = sys_ode.Q
    B_ode = sys_ode.B

    return J_ode, Q_ode, B_ode, T, M_ode


# sys_int1 = SysPhdae.transformer(beam1, beam2, ind1, ind2_int1, R)
# sys_all = SysPhdae.transformer(sys_int1, payload, ind2_int2, ind3, np.eye(3))

theta1_ref = pi / 3
theta2_ref = 0
Kp1 = 160
Kp2 = 60
Kv1 = 11
Kv2 = 1.1

J_sys, Q_sys, B_sys, T, M_sys = build_man(0)


def sys_manipulator_ode(t,y):

    alpha_v = y[:-2]
    theta_v = y[-2:]

    theta1 = theta_v[0]
    theta2 = theta_v[1]

    theta1_dot = B_sys[:, 0].T @ Q_sys @ alpha_v
    theta2_dot = B_sys[:, 1].T @ Q_sys @ alpha_v

    if t < 1:
        u1 = 0
        u2 = 0
        u_v = np.array([u1, u2])
    else:
        u1 = Kp1 * (theta1_ref - theta1) - Kv1 * theta1_dot
        u2 = Kp2 * (theta2_ref - theta2) - Kv2 * theta2_dot
        u_v = np.array([u1, u2])

    daldt_v = J_sys @ Q_sys @ alpha_v + B_sys @ u_v
    dthdt_v = np.array([theta1_dot, theta2_dot])
    dydt = np.concatenate((daldt_v, dthdt_v))
    return dydt

t0 = 0
t_fin = 4
n_t = 100
t_ev = np.linspace(t0, t_fin, num=n_t)
t_span = [t0, t_fin]
ntot_sys = beam1_hinged.n + beam2.n + payload.n
n_sys_ode = ntot_sys - 5 + 2
print(n_sys_ode)
y0 = np.zeros((n_sys_ode, ))

sol = integrate.solve_ivp(sys_manipulator_ode, t_span, y0, method='RK45', vectorized=False, t_eval=t_ev)

t_ev = sol.t
y_sol = sol.y

alpha_sol = y_sol[:-2, :]

e_sol = np.zeros_like(alpha_sol)
etrue_sol = np.zeros((ntot_sys, n_t))
for i in range(n_t):
    e_sol[:, i] = M_sys @ alpha_sol[:, i]
    etrue_sol[:, i] = T.T @ e_sol[:, i]
    # x_cr, u_cr, w_cr = draw_bending(n_dr, eigcrank_r, eigcrank_p, L_crank)



theta1_sol = y_sol[-2, :]
theta2_sol = y_sol[-1, :]
n_ev = len(t_ev)

plt.figure()
plt.plot(t_ev, theta1_sol*180/pi, 'r')
plt.figure()
plt.plot(t_ev, theta2_sol*180/pi, 'b')

plt.show()
