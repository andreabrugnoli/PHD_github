import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
fntsize = 16

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import scipy.linalg as la

from modules_ph.classes_phsystem import SysPhdaeRig
from system_components.beams import FloatFlexBeam, matrices_j2d


L_beam = 141.42
rho_beam = 7.8 * 10 ** (-3)
E_beam = 2.10 * 10**6
A_beam = 9
I_beam = 6.75
Mz_max = 10000


n_elem = 2

beam = FloatFlexBeam(n_elem, L_beam, rho_beam, A_beam, E_beam, I_beam)

# dofs2dump = list([0, 1, 2])
# dofs2keep = list(set(range(beam.n)).difference(set(dofs2dump)))

E_hinged = beam.E[2:, 2:]

J_hinged = beam.J[2:, 2:]

B_hinged = beam.B[2:, :]
n_th = 1

beam_hinged = SysPhdaeRig(len(E_hinged), 0, n_th, beam.n_p, beam.n_q,
                           E=E_hinged, J=J_hinged, B=B_hinged)


n_e = beam_hinged.n_e
n_r = beam_hinged.n_r

n_p = beam_hinged.n_p
n_pu = int(n_p/3)
n_pw = 2*n_pu
n_f = beam_hinged.n_f
n_tot = n_e + n_th


M = beam_hinged.M_e
invM = la.inv(M)
J = beam_hinged.J
B_Mz0 = beam_hinged.B[:, 2]

Jf_rz, Jf_fx, Jf_fy = matrices_j2d(n_elem, L_beam, rho_beam, A_beam)[2:]


t_load = 0.2
t1 = 10
t2 = t1 + t_load
t3 = t2


t_0 = 0
t_fin = 30

Mz_max = 10000


def sys(t, y):

    print(t/t_fin*100)

    Mz_0 = 0

    if t <= t_load:
        Mz_0 = Mz_max*t/t_load

    if t>t_load and t<t1:
        Mz_0 = Mz_max

    if t>=t1 and t<=t2:
        Mz_0 = Mz_max*(1 - (t-t1)/t_load)


    y_e = y[:n_e]
    omega = y[0]
    theta = y[-1]

    # p_u = M[n_r:n_r + n_pu, :] @ y_e
    # p_w = M[n_r + n_pu:n_r + n_p, :] @ y_e
    #
    # p_wdis = np.array([p_w[i] for i in range(len(p_w)) if i % 2 == 0])
    # p_udis = np.zeros_like(p_w)
    # p_udis[::2] = p_u
    #
    # J[n_r:n_r + n_p, 0] = np.concatenate((p_wdis, -p_udis))
    # J[0, n_r:n_r + n_p] = np.concatenate((-p_wdis, +p_udis))

    eu_beam = y[n_r:n_r + n_pu]
    ew_beam = y[n_r + n_pu:n_r+ n_p]

    jf_u = Jf_fx @ eu_beam
    jf_w = Jf_rz * omega + Jf_fy @ ew_beam

    jf_u_cor = Jf_fx @ eu_beam
    jf_w_cor = Jf_fy @ ew_beam

    J[n_r:n_r + n_p, 0] = np.concatenate((+jf_w, -jf_u)) + np.concatenate((jf_w_cor, -jf_u_cor))
    J[0, n_r:n_r + n_p] = 2*np.concatenate((-jf_w, +jf_u))

    dedt = invM @ (J @ y_e + B_Mz0 * Mz_0)
    dth = np.array([omega])

    dydt = np.concatenate((dedt, dth))
    return dydt


y0 = np.zeros(n_tot,)

th0 = 0
y0[-1] = th0

t_ev = np.linspace(t_0, t_fin, num=500)
t_span = [t_0, t_fin]

sol = solve_ivp(sys, t_span, y0, method='RK45', vectorized=False, t_eval=t_ev)

t_sol = sol.t
y_sol = sol.y
om_sol = y_sol[0, :]


plt.plot(t_sol, om_sol, 'r')
plt.xlabel("Time $s$", fontsize=fntsize)
plt.ylabel("$\omega_z$", fontsize=fntsize)
plt.title("Angular velocity along z", fontsize=fntsize)
plt.show()