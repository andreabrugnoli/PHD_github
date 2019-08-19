import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
fntsize = 16

import numpy as np
import quaternion
from scipy.spatial.transform import Rotation
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import scipy.linalg as la

from modules_phdae.classes_phsystem import SysPhdaeRig
from system_components.beams import SpatialBeamTorsion, draw_deformation3D
from math import pi

from tools_plotting.animate_lines import animate_line3d

def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def skew_flex(al_u, al_v, al_w):

    nu = len(al_u)
    nv = len(al_v)
    nw = len(al_w)

    n_rows = nu + nv + nw
    skew_mat_flex = np.zeros((n_rows, 3))

    nv_end = nu + nv
    skew_mat_flex[:nu, :] = np.column_stack((np.zeros((nu,)), - al_w, al_v))
    skew_mat_flex[nu:nv_end, :] = np.column_stack((al_w, np.zeros((nv,)), -al_u))
    skew_mat_flex[nv_end:, :] = np.column_stack((-al_v, al_u, np.zeros((nw, ))))

    return skew_mat_flex


L_beam = 141.42
rho_beam = 7.8 * 10 ** (-3)
E_beam = 2.10 * 10**6
nu_beam = 0.3
G_beam = E_beam/(2*(1+nu_beam))
A_beam = 9
I_beam = 6.75
Ir_beam = 2 * I_beam

Mz_max = 200


Fz_max = 100
mass_beam = rho_beam * A_beam * L_beam
Jxx_beam = 2 * I_beam * rho_beam * L_beam

n_elem = 2

beam = SpatialBeamTorsion(n_elem, L_beam, rho_beam, A_beam, E_beam, G_beam, I_beam, Ir_beam)

# dofs2dump = list([0, 1, 2])
# dofs2keep = list(set(range(beam.n)).difference(set(dofs2dump)))

E_hinged = beam.E[3:, 3:]

J_hinged = beam.J[3:, 3:]

B_hinged = beam.B[3:, :]
beam_hinged = SysPhdaeRig(len(E_hinged), 0, 3, beam.n_p, beam.n_q,
                           E=E_hinged, J=J_hinged, B=B_hinged)


n_e = beam_hinged.n_e
n_r = beam_hinged.n_r
n_quat = 4

n_p = beam_hinged.n_p
n_pu, n_pv, n_pw = int(n_p/3), int(n_p/3), int(n_p/3)

n_f = beam_hinged.n_f
n_tot = n_e + n_quat


M = beam_hinged.M_e
invM = la.inv(M)
J = beam_hinged.J
B_Mz0 = beam_hinged.B[:, 5]
B_FzL = beam_hinged.B[:, 8]

B_FxyzL = beam_hinged.B[:, 6:9]


t_load = 0.2
t1 = 10
t2 = t1 + t_load
t3 = 15
t4 = t3 + t_load
t5 = t4 + t_load

t_0 = 0
t_fin = 50


def sys(t,y):

    print(t/t_fin*100)

    if t <= t_load:
        Mz_0 = Mz_max*t/t_load
    elif t>t_load and t<t1:
        Mz_0 = Mz_max
    elif t>=t1 and t<=t2:
        Mz_0 = Mz_max*(1 - (t-t1)/t_load)
    else:
        Mz_0 = 0

    if t>=t3 and t<t4:
        Fz_L = Fz_max * (t-t3) / t_load
    elif t>=t4 and t<=t5:
        Fz_L = Fz_max * (1 - (t - t4) / t_load)
    else:
        Fz_L = 0

    y_e = y[:n_e]
    omega = y[:n_r]
    y_quat = y[-n_quat:]

    pi_beam = M[:n_r, :] @ y_e
    J[:n_r, :n_r] = skew(pi_beam)

    p_u = M[n_r:n_r + n_pu, :] @ y_e
    p_v = M[n_r + n_pu:n_r + n_pu + n_pv, :] @ y_e
    p_w = M[n_r + n_pu + n_pv:n_r + n_p, :] @ y_e

    alflex_cross = skew_flex(p_u, p_v, p_w)

    # p_u[1::2] = 0
    # p_v[1::2] = 0
    # p_w[1::2] = 0
    # alflex_cross_noang = skew_flex(p_u, p_v, p_w)

    J[n_r:n_r + n_p, :n_r] = alflex_cross
    J[:n_r, n_r:n_r + n_p] = -alflex_cross.T

    # dedt = invM @ (J @ y_e + B_Mz0 * Mz_0 + B_FzL * Fz_L)

    act_quat = np.quaternion(y_quat[0], y_quat[1], y_quat[2], y_quat[3])
    Rot_mat = quaternion.as_rotation_matrix(act_quat)
    dedt = invM @ (J @ y_e + B_Mz0 * Mz_0 + B_FxyzL @ Rot_mat.T[:, 2] * Fz_L)

    Omega_mat = np.array([[0, -omega[0], -omega[1], -omega[2]],
                         [omega[0], 0, omega[2], -omega[1]],
                         [omega[1], -omega[2], 0, omega[0]],
                         [omega[2], omega[1], -omega[0], 0]])

    dquat = 0.5 * Omega_mat @ y_quat

    dydt = np.concatenate((dedt, dquat))
    return dydt


y0 = np.zeros(n_tot,)

quat0 = quaternion.as_float_array(quaternion.from_rotation_matrix(np.eye(3)))
y0[-n_quat:] = quat0

t_ev = np.linspace(t_0, t_fin, num=500)
t_span = [t_0, t_fin]

sol = solve_ivp(sys, t_span, y0, method='RK45', vectorized=False, t_eval=t_ev, max_step=0.02)

t_sol = sol.t
y_sol = sol.y
omB_sol = y_sol[:n_r, :]
quat_sol = quaternion.as_quat_array(y_sol[-4:, :].T)

n_ev = len(t_sol)
omI_sol = np.zeros((3, n_ev))


for i in range(n_ev):
    omI_sol[:, i] = quaternion.as_rotation_matrix(quat_sol[i]) @ omB_sol[:, i]

plt.figure()
plt.plot(t_sol, omI_sol[0], 'r')
plt.xlabel("Time $s$", fontsize=fntsize)
plt.ylabel("$\omega_x$", fontsize=fntsize)
plt.title("Angular velocity along x in I", fontsize=fntsize)

plt.figure()
plt.plot(t_sol, omI_sol[1], 'r')
plt.xlabel("Time $s$", fontsize=fntsize)
plt.ylabel("$\omega_y$", fontsize=fntsize)
plt.title("Angular velocity along y in I", fontsize=fntsize)

plt.figure()
plt.plot(t_sol, omI_sol[2], 'r')
plt.xlabel("Time $s$", fontsize=fntsize)
plt.ylabel("$\omega_z$", fontsize=fntsize)
plt.title("Angular velocity along z in I", fontsize=fntsize)

plt.show()

#
#
# plt.figure()
# plt.plot(t_sol, omB_sol[0], 'r')
# plt.xlabel("Time $s$", fontsize=fntsize)
# plt.ylabel("$\omega_x$", fontsize=fntsize)
# plt.title("Angular velocity along x in B", fontsize=fntsize)
#
# plt.figure()
# plt.plot(t_sol, omB_sol[1], 'r')
# plt.xlabel("Time $s$", fontsize=fntsize)
# plt.ylabel("$\omega_y$", fontsize=fntsize)
# plt.title("Angular velocity along y in B", fontsize=fntsize)
#
# plt.figure()
# plt.plot(t_sol, omB_sol[2], 'r')
# plt.xlabel("Time $s$", fontsize=fntsize)
# plt.ylabel("$\omega_z$", fontsize=fntsize)
# plt.title("Angular velocity along z in B", fontsize=fntsize)
# plt.show()
