import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
fntsize = 16

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import scipy.linalg as la

from modules_ph.classes_phsystem import SysPhdaeRig
from system_components.beams import FloatCentBeamNoTr

L_beam = 0.5
rho_beam = 0.0858
E_beam = 5.50
A_beam = 1
I_beam = 1

offset = 0.05
J_beam = 1/3*rho_beam*A_beam*((offset+L_beam)**3 - offset**3)
beta = 0.5
J_joint = beta*J_beam
n_elem = 6

beam = FloatCentBeamNoTr(n_elem, L_beam, rho_beam, A_beam, E_beam, I_beam, J_joint=J_joint, offset=offset)

# dofs2dump = list([0, 1, 2])
# dofs2keep = list(set(range(beam.n)).difference(set(dofs2dump)))

E_hinged = beam.E[2:, 2:]
J_hinged = beam.J[2:, 2:]
Q_hinged = beam.Q[2:, 2:]
B_hinged = beam.B[2:, :]

n_th = 1

beam_hinged = SysPhdaeRig(len(E_hinged), 0, n_th, beam.n_p, beam.n_q,
                           E=E_hinged, J=J_hinged, Q=Q_hinged, B=B_hinged)


n_e = beam_hinged.n_e
n_r = beam_hinged.n_r

n_p = beam_hinged.n_p
n_pw = n_p
n_f = beam_hinged.n_f
n_tot = n_e + n_th

nq_cen = n_elem + 1


M = beam_hinged.E
invM = la.inv(M)
J = beam_hinged.J
Q = beam_hinged.Q

B_Mz0 = beam_hinged.B[:, 2]

print(M.shape, J.shape, Q.shape, B_Mz0.shape)

t1 = 0.05
t2 = 0.1
t3 = 0.15


t_0 = 0
t_fin = 0.3


Mz_max = 1


def sys(t, y):

    print(t/t_fin*100)

    Mz_0 = 0

    if t <= t1:
        Mz_0 = Mz_max

    if t>=t2 and t<=t3:
        Mz_0 = -Mz_max


    y_e = y[:n_e]
    omega = y[0]
    theta = y[-1]

    Om_mat = np.eye(n_e)
    Om_mat[-nq_cen:, -nq_cen:] = omega**2*np.eye(nq_cen)

    dedt = invM @ (J @ Q @ Om_mat @ y_e + B_Mz0 * Mz_0)
    dth = np.array([omega])

    dydt = np.concatenate((dedt, dth))
    return dydt


y0 = np.zeros(n_tot,)

th0 = 0
y0[-1] = th0

n_ev = 500
t_ev = np.linspace(t_0, t_fin, num=n_ev)
t_span = [t_0, t_fin]

sol = solve_ivp(sys, t_span, y0, method='RK45', vectorized=False, t_eval=t_ev)

t_sol = sol.t
y_sol = sol.y
e_sol = y_sol[:-1, :]
om_sol = y_sol[0, :]


ep_sol = y_sol[n_r:n_r + n_p, :]

nw = int(2*n_p/3)
nu = int(n_p/3)

wp_sol = ep_sol

ewF_B = wp_sol[2*(n_elem - 1), :]

path_fig = "/home/a.brugnoli/Plots_Videos/Python/Plots/Multibody_PH/CrankSlider/"
fntsize = 16

H_vec = np.zeros((n_ev,))
for i in range(n_ev):
    H_vec[i] = 0.5 * (e_sol[:, i].T @ M @ e_sol[:, i])

fig = plt.figure()
plt.plot(t_ev, H_vec, 'b-')
plt.xlabel(r'{Time} (s)', fontsize=fntsize)
plt.ylabel(r'{Hamiltonian} (J)', fontsize=fntsize)
plt.title(r"Hamiltonian spinning beam",
          fontsize=fntsize)
# plt.savefig(path_fig + 'Hamiltonian.eps', format="eps")


omega_int = interp1d(t_ev, om_sol, kind='linear')
ewF_B_int = interp1d(t_ev, ewF_B, kind='linear')

def sys(t,y):

    dwdt = ewF_B_int(t) #- omega_int(t) * y[0]

    dydt = dwdt
    return dydt


wF0 = 0

r_sol = solve_ivp(sys, [0, t_fin], [wF0], method='RK45', t_eval=t_sol)

wF_B = r_sol.y[0, :]

fig = plt.figure()
plt.plot(t_sol, om_sol, 'r')
plt.xlabel("Time $s$", fontsize=fntsize)
plt.ylabel("$\omega_z$", fontsize=fntsize)
plt.title("Angular velocity along z", fontsize=fntsize)

fig = plt.figure()
plt.plot(t_sol, wF_B*1000, 'b-')
plt.xlabel(r'Crank angle [s]', fontsize=fntsize)
plt.ylabel(r'$u_y$ [mm]', fontsize=fntsize)
plt.title(r"Midpoint vertical deflection", fontsize=fntsize)
axes = plt.gca()
# plt.savefig(path_fig + 'wM_disp.eps', format="eps")

plt.show()