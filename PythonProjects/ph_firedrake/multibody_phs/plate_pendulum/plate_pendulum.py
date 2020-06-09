import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
fntsize = 15

import numpy as np
import quaternion
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import scipy.linalg as la
from scipy.sparse.linalg import lsqr

from modules_ph.classes_phsystem import SysPhdaeRig
from system_components.plates import FloatingKP6dofs_Bell, find_point
from math import pi

def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


Lx = 0.6
Ly = 0.3
h= 0.01
E = 10**5

# Lx = 1
# Ly = 1
# h= 0.1
# E =2*1e9


nu = 0.3
nx = 2
ny = 2
rho = 7810
g = 9.81

x_H = Lx
y_H = 0

point_H = np.array([x_H, y_H])
plate = FloatingKP6dofs_Bell(Lx, Ly, h, rho, E, nu, nx, ny, [0, 0])
Jf_tx, Jf_ty, Jf_tz, Jf_rx, Jf_ry, Jf_rz, Jf_fx, Jf_fy, Jf_fz = plate.matrices_j3d()

i_H = find_point(plate.coords, point_H)[0]

# print(plate.coords)
# print(i_H)

mat_u = plate.vec_shapefun()

n_rig = 6
np_m1 = plate.np_m1

m_plate = plate.M_e[0, 0]
assert m_plate == rho * h* Lx * Ly

M_om = plate.M_e[3:6, 3:6]
M_f = plate.M_e[n_rig:, n_rig:]

J_f = plate.J_e[n_rig:, n_rig:]


M_omP_ef = plate.M_e[3:, 3:]
J_omP_ef = plate.J_e[3:, 3:]

invM_omP_ef = la.inv(M_omP_ef)

# invM_om = la.inv(M_om)
# invM_f = la.inv(M_f)

np_fl = plate.n_p

t_fin = 1.4

n_quat = 4
n_uf = n_quat+np_fl
n_omP = n_uf + 3
n_vf = n_omP + np_fl

n_tot = n_omP + plate.n_f

C_om = g*m_plate*np.array([Lx/2, Ly/2, 0]).reshape((3, ))



def sys(t,y):

    print(t/t_fin*100)

    y_quat = y[:n_quat]
    quat = np.quaternion(y[0], y[1], y[2], y[3])

    u_f = y[n_quat:n_uf]
    omega = y[n_uf:n_omP]
    v_f = y[n_omP:n_vf]

    omP_ef = y[n_uf:]

    e_f = y[n_omP:]

    efx = e_f[:np_m1-1]
    efy = e_f[np_m1-1:2*np_m1-2]
    efz = e_f[2*np_m1-2:np_fl]

    # print(efx.shape, efy.shape, efz.shape)

    R_mat = quaternion.as_rotation_matrix(quat)

    Rz_mat = R_mat[2, :]

    # pi_beam = M_om @ omega + 2*M_omP_ef[:3, 3:] @ e_f
    # J_omP_ef[:3, :3] = skew(pi_beam)
    Jf_om = Jf_rx * omega[0] + Jf_ry * omega[1] + Jf_rz * omega[2] + Jf_fz @ efz + Jf_fx @ efx  + Jf_fy @ efy
    Jf_om_cor = Jf_fy @ efy + Jf_fz @ efz + Jf_fx @ efx
    J_omP_ef[3:3 + np_fl, :3] = Jf_om + Jf_om_cor
    J_omP_ef[:3, 3:3 + np_fl] = -2 * Jf_om.T

    # Quat_mat = np.array([[-y_quat[1],-y_quat[2],-y_quat[3]],
    #                      [ y_quat[0],-y_quat[3], y_quat[2]],
    #                      [ y_quat[3], y_quat[0],-y_quat[1]],
    #                      [-y_quat[2], y_quat[1], y_quat[0]]
    #                      ])

    Omega_mat = np.array([[0, -omega[0], -omega[1], -omega[2]],
                          [omega[0], 0, omega[2], -omega[1]],
                          [omega[1], -omega[2], 0, omega[0]],
                          [omega[2], omega[1], -omega[0], 0]])

    dt_quat = 0.5 * Omega_mat @ y_quat

    # dt_quat = 0.5 * Quat_mat @ omega

    dt_uf = v_f

    f_omf = np.concatenate((-skew(Rz_mat).T @ C_om ,\
                            - rho * g * h * mat_u @ Rz_mat.T), axis=0)

    # print(M_omP_ef.shape, J_omP_ef.shape, omP_ef.shape, f_omf.shape)
    dt_omP_ef = invM_omP_ef @ (J_omP_ef @ omP_ef + f_omf)

    # dt_omP = -invM_om @ skew(Rz_mat).T @ C_om
    #
    # dt_ef = invM_f @ (J_f @ e_f - rho * g * h * mat_u @ Rz_mat.T)

    dydt = np.concatenate((dt_quat, dt_uf, dt_omP_ef))

    return dydt


y0 = np.zeros(n_tot,)

quat0 = quaternion.as_float_array(quaternion.from_rotation_matrix(np.eye(3)))
y0[:n_quat] = quat0

n_t = 500
t_ev = np.linspace(0, t_fin, num=n_t)
t_span = [0, t_fin]


sol = solve_ivp(sys, t_span, y0, method='LSODA', vectorized=False, t_eval=t_ev)

t_sol = sol.t
y_sol = sol.y

n_ev = len(t_sol)

quat_sol = quaternion.as_quat_array(y_sol[:n_quat, :].T)
uf_sol = y_sol[n_quat:n_uf, :]
om_sol = y_sol[n_quat:n_omP]


ufx_H = uf_sol[i_H-1]
ufy_H = uf_sol[plate.np_m1 + i_H-2]
ufz_H = uf_sol[2*plate.np_m1 + 6*i_H-5]

rB_H = np.concatenate(((x_H + ufx_H).reshape(1, -1), (y_H + ufy_H).reshape(1, -1), ufz_H.reshape(1, -1)), axis=0)
print(rB_H.shape)

rI_H = np.zeros((3, n_ev))

for i in range(n_ev):
    rI_H[:, i] = quaternion.as_rotation_matrix(quat_sol[i]) @ rB_H[:, i]

plt.figure()
plt.plot(t_sol, rI_H[0], 'r', label='x H')
plt.plot(t_sol, rI_H[1], 'b', label='y H')
plt.plot(t_sol, rI_H[2], 'g', label='z H')
plt.legend()

plt.figure()
plt.plot(t_sol, om_sol[0], 'r', label='x')
plt.plot(t_sol, om_sol[1], 'b', label='y')
plt.plot(t_sol, om_sol[2], 'g', label='z')
plt.show()