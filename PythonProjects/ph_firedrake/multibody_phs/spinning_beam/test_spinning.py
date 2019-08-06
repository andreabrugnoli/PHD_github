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
from system_components.beams import SpatialBeam, draw_deformation3D
from math import pi

from tools_plotting.animate_lines import animate_line3d

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


L_beam = 0.14142
rho_beam = 7.8 * 10 ** 3
E_beam = 210 * 10**9
A_beam = 9 * 10**(-6)
I_beam = 6.75 * 10**(-12)
Jxx_beam = 2 * I_beam * rho_beam * L_beam
n_elem = 2

beam = SpatialBeam(n_elem, L_beam, rho_beam, A_beam, E_beam, I_beam, Jxx_beam)

dofs2dump = list([0, 1, 2])
dofs2keep = list(set(range(beam.n)).difference(set(dofs2dump)))

E_hinged = beam.E[dofs2keep, :]
E_hinged = E_hinged[:, dofs2keep]

J_hinged = beam.J[dofs2keep, :]
J_hinged = J_hinged[:, dofs2keep]

B_hinged = beam.B[dofs2keep, :]
beam_hinged = SysPhdaeRig(len(E_hinged), 0, 3, beam.n_p, beam.n_q,
                           E=E_hinged, J=J_hinged, B=B_hinged)


n_e = beam_hinged.n_e
n_r = beam_hinged.n_r
n_quat = 4

n_p = beam_hinged.n_p
n_pu = int(n_p/5)
n_pv = 2*n_pu
n_pw = 2*n_pu
n_f = beam_hinged.n_f
n_tot = n_e + n_quat


M = beam_hinged.M_e
invM = la.inv(M)
J = beam_hinged.J
B_Mz0 = beam_hinged.B[:, 5]
B_FzL = beam_hinged.B[:, 8]

t_load = 0.2
t1 = 10
t2 = t1 + t_load
t3 = t2 + 15
t4 = t3 + t_load
t5 = t4 + t_load

t_0 = 0
t_fin = 50

Mz_max = 0.2
Fz_max = 100

def sys(t,y):

    print(t/t_fin*100)


    Mz_0 = 0
    Fz_L = 0

    if t <= t_load:
        Mz_0 = Mz_max*t/t_load

    if t>t_load and t<t1:
        Mz_0 = Mz_max

    if t>=t1 and t<=t2:
        Mz_0 = Mz_max*(1 - (t-t1)/t_load)

    if t>=t3 and t<t4:
        Fz_L = Fz_max * (t-t3) / t_load

    if t>=t4 and t<=t5:
        Fz_L = Fz_max * (1 - (t - t4) / t_load)

    y_e = y[:n_e]
    omega = y[:n_r]
    y_quat = y[-n_quat:]

    pi_beam = M[:3, :] @ y_e
    J[:n_r, :n_r] = skew(pi_beam)

    p_v = M[n_r + n_pu:n_r + n_pu + n_pv, :] @ y_e
    p_v[1::2] = 0
    p_vdis = np.array([p_v[i] for i in range(len(p_v)) if i % 2 == 0])

    p_w = M[n_r + n_pu + n_pv:n_r + n_p, :] @ y_e
    p_w[1::2] = 0
    p_wdis = np.array([p_w[i] for i in range(len(p_w)) if i % 2 == 0])

    p_udis = M[n_r:n_r + n_pu, :] @ y_e
    p_u = np.zeros_like(p_w)
    p_u[::2] = p_udis

    alflex_cross = skew_flex(p_u, p_v, p_w, p_udis, p_vdis, p_wdis)
    J[n_r:n_r+ n_p, :n_r] = alflex_cross
    J[:n_r, n_r:n_r + n_p] = -alflex_cross.T

    dedt = invM @ (J @ y_e + B_Mz0 * Mz_0 + B_FzL * Fz_L)

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

sol = solve_ivp(sys, t_span, y0, method='RK45', vectorized=False, t_eval=t_ev)

t_sol = sol.t
y_sol = sol.y
om_sol = y_sol[:n_r, :]

plt.plot(t_sol, om_sol[2], 'r')
plt.xlabel("Time $s$", fontsize=fntsize)
plt.ylabel("$\omega_z$", fontsize=fntsize)
plt.title("Angular velocity along z", fontsize=fntsize)
plt.show()