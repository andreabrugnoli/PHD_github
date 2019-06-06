import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from modules_phdae.classes_phsystem import SysPhdaeRig
from system_components.beams import FloatingPlanarEB, draw_bending
from math import pi
from scipy import integrate
from system_components.manipulator.manipulator_constants import rho1, EI1, L1, rho2, EI2, L2, n_rig, J_joint1, J_joint2, J_payload, m_joint2, m_payload
from tools_plotting.animate_plot import animate_plot

n_el = 2
n_p = n_el * 2
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

    return sys_dae

sys_dae = build_man(0)

J_e = sys_dae.J_e
M_e = sys_dae.M_e
B_e = sys_dae.B_e
G_e = sys_dae.G_e

plt.spy(M_e); plt.show()

Q_e = la.inv(M_e)
A_lmb = - la.inv(G_e.T @ Q_e @ G_e) @ G_e.T @ Q_e

n_e = sys_dae.n_e
n_lmb = sys_dae.n_lmb
ref_deg = 60
theta1_ref = ref_deg*pi/180
theta2r_ref = 0
Kp1 = 160
Kp2 = 60
Kv1 = 11
Kv2 = 1.1


def sys_manipulator_ode(t, y):

    e_v = y[:-2]
    theta_v = y[-2:]

    theta1 = theta_v[0]
    theta2r = theta_v[1]

    theta_dot = B_e.T @ e_v

    theta1_dot, theta2r_dot = theta_dot

    if t < 1:
        u1 = 0
        u2 = 0
        u_v = np.array([u1, u2])
    else:
        u1 = Kp1 * (theta1_ref - theta1) - Kv1 * theta1_dot
        u2 = Kp2 * (theta2r_ref - theta2r) - Kv2 * theta2r_dot
        u_v = np.array([u1, u2])

    lmb = A_lmb @ (J_e @ e_v + B_e @ u_v)
    dedt_v = Q_e @ (J_e @ e_v + B_e @ u_v + G_e @ lmb)
    dthdt_v = theta_dot
    dydt = np.concatenate((dedt_v, dthdt_v))

    return dydt


t0 = 0
t_fin = 4
n_t = 100
t_ev = np.linspace(t0, t_fin, num=n_t)
t_span = [t0, t_fin]

n_res = n_e + 2

y0 = np.zeros((n_res, ))

sol = integrate.solve_ivp(sys_manipulator_ode, t_span, y0, method='RK45', vectorized=False, t_eval=t_ev)

t_ev = sol.t
y_sol = sol.y
n_ev = len(t_ev)

dt_vec = np.diff(t_ev)

e_sol = y_sol[:n_e, :]
theta1_sol = y_sol[-2, :]
theta2r_sol = y_sol[-1, :]
theta2a_sol = theta1_sol + theta2r_sol

v_intI = np.zeros((2, n_ev))

v_intI1 = np.zeros((2, n_ev))
v_intI2 = np.zeros((2, n_ev))

n_dr = 50

xI_1 = np.zeros((n_dr, n_ev))
yI_1 = np.zeros((n_dr, n_ev))
xI_2 = np.zeros((n_dr, n_ev))
yI_2 = np.zeros((n_dr, n_ev))

xI_1[:, 0] = np.linspace(0, L1, n_dr)
xI_2[:, 0] = np.linspace(L1, L1 + L2, n_dr)

vx_I1 = np.zeros((n_dr, n_ev))
vy_I1 = np.zeros((n_dr, n_ev))
vx_I2 = np.zeros((n_dr, n_ev))
vy_I2 = np.zeros((n_dr, n_ev))

vx_B1 = np.zeros((n_dr, n_ev))
vy_B1 = np.zeros((n_dr, n_ev))
vx_B2 = np.zeros((n_dr, n_ev))
vy_B2 = np.zeros((n_dr, n_ev))

theta1Num_sol = np.zeros((n_ev, ))
dtheta1_sol = np.zeros((n_ev, ))
dtheta2r_sol = np.zeros((n_ev, ))

# n_r1 = 1
# n_r2 = n_r1 + 3
# n_r3 = n_r2  #+ 3
#
# n_p1 = n_r3 + n_p
# n_p2 = n_p1 + n_p

n_r1 = beam1_hinged.n_r

n_r2 = beam1_hinged.n_r + beam2.n_r
n_r3 = n_r2 + payload.n_r

n_p1 = n_r3 + beam1_hinged.n_p
n_p2 = n_p1 + beam2.n_p

for i in range(n_ev):

    dtheta1_sol[i] = B_e[:, 0].T @ e_sol[:, i]
    dtheta2r_sol[i] = B_e[:, 1].T @ e_sol[:, i]

    vx_rigB2 = L1 * dtheta1_sol[i] * np.sin(theta2r_sol[i])
    vy_rigB2 = L1 * dtheta1_sol[i] * np.cos(theta2r_sol[i])

    # print(e_sol[1, i] - vx_rigB2)
    # print(e_sol[2, i] - vy_rigB2)


    # vx_rigB2 = L1 * dtheta1_sol[i] * np.sin(theta2r_sol[i])
    # vy_rigB2 = L1 * dtheta1_sol[i] * np.cos(theta2r_sol[i])
    # om_rigB2 = dtheta1_sol[i] + dtheta2r_sol[i]
    # erig_1 = np.array([0, 0, dtheta1_sol[i]])
    # erig_2 = np.array([vx_rigB2, vy_rigB2, om_rigB2])

    erig_1 = np.array([0, 0, e_sol[0, i]])
    erig_2 = np.array([e_sol[1, i], e_sol[2, i], e_sol[3, i]])

    ep_1 = e_sol[n_r1:n_p1, i]
    ep_2 = e_sol[n_r3:n_p2, i]

    # ep_1 = np.zeros((n_p, ))
    # ep_2 = np.zeros((n_p, ))

    x_B1, vx_B1[:, i], vy_B1[:, i] = draw_bending(n_dr, erig_1, ep_1, L1)
    x_B2, vx_B2[:, i], vy_B2[:, i] = draw_bending(n_dr, erig_2, ep_2, L2)

    R_B1_I = np.array([[np.cos(theta1_sol[i]), -np.sin(theta1_sol[i])],
                       [np.sin(theta1_sol[i]),  np.cos(theta1_sol[i])]])

    R_B2_I = np.array([[np.cos(theta2a_sol[i]), -np.sin(theta2a_sol[i])],
                       [np.sin(theta2a_sol[i]),  np.cos(theta2a_sol[i])]])

    for j in range(n_dr):
        vx_I1[j, i], vy_I1[j, i] = R_B1_I @ np.array([vx_B1[j, i], vy_B1[j, i]])

        vx_I2[j, i], vy_I2[j, i] = R_B2_I @ np.array([vx_B2[j, i], vy_B2[j, i]])

        # if j == 0:
        #         v_intI2[:, i] = R_B2_I @ np.array([vx_B2[j, i], vy_B2[j, i]])
        #
        # if j == n_dr-1:
        #         v_intI1[:, i] = R_B1_I @ np.array([vx_B1[j, i], vy_B1[j, i]])

        # assert abs(v_intI1[:, i] - v_intI2[:, i]).all() < 1e-9
        # print(v_intI1[:, i] - v_intI2[:, i])

    if i > 0:
        xI_1[:, i] = xI_1[:, i - 1] + 0.5 * (vx_I1[:, i - 1] + vx_I1[:, i]) * dt_vec[i - 1]
        yI_1[:, i] = yI_1[:, i - 1] + 0.5 * (vy_I1[:, i - 1] + vy_I1[:, i]) * dt_vec[i - 1]

        xI_2[:, i] = xI_2[:, i - 1] + 0.5 * (vx_I2[:, i - 1] + vx_I2[:, i]) * dt_vec[i - 1]
        yI_2[:, i] = yI_2[:, i - 1] + 0.5 * (vy_I2[:, i - 1] + vy_I2[:, i]) * dt_vec[i - 1]

        theta1Num_sol[i] = theta1Num_sol[i - 1] + 0.5 * (e_sol[0, i - 1] + e_sol[0, i]) * dt_vec[i - 1]

#
# plt.figure()
# plt.plot(t_ev, theta1_sol*180/pi, 'r')
# plt.figure()
# plt.plot(t_ev, theta2r_sol*180/pi, 'b')

# plt.figure()
# plt.plot(t_ev, dtheta1_sol*180/pi, 'r')
# plt.figure()
# plt.plot(t_ev, dtheta2r_sol*180/pi, 'b')
#
# plt.figure()
# plt.plot(t_ev, e_sol[0, :]*180/pi, 'r')
# plt.figure()
# plt.plot(t_ev, e_sol[3, ]*180/pi, 'b')

n_rig = sys_dae.n_r
for i in range(n_rig):
    plt.figure()
    plt.plot(t_ev, e_sol[i, ]*180/pi, 'b')

x_manI = np.concatenate((xI_1, xI_2), axis=0)
y_manI = np.concatenate((yI_1, yI_2), axis=0)

# anim = animate_plot(t_ev, xI_2, yI_2, xlabel=None, ylabel=None, title=None)
# anim = animate_plot(t_ev, x_manI, y_manI, xlabel=None, ylabel=None, title=None

# from tools_plotting.animate_plotrigfl import animate_plot
# anim = animate_plot(t_ev, x_manI, y_manI, theta1_sol, theta2a_sol,  xlabel=None, ylabel=None, title=None)

# from tools_plotting.animate_plotrigfl import animate_plot
# anim = animate_plot(t_ev, x_manI, y_manI, theta1_sol, theta2a_sol,  xlabel=None, ylabel=None, title=None)

# fps = 20
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps= fps, metadata=dict(artist='Me'), bitrate=1800)
# anim.save(path_out + 'Kirchh_Rod.mp4', writer=writer)
# plt.show()

plt.show()
