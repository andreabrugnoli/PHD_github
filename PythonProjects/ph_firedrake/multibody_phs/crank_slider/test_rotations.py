import quaternion
from scipy.spatial.transform import Rotation
import numpy as np
from scipy import linalg as la
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
fntsize = 15
from mpl_toolkits.mplot3d import Axes3D
from math import pi
from modules_phdae.classes_phsystem import check_skew_symmetry
from scipy.optimize import fsolve


def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

L_crank = 0.15
L_coupler = 0.3
ecc = 0.1
omega_cr = 150

theta_cr = pi/2
theta1_cl = np.arcsin(ecc/np.sqrt(L_coupler**2 - L_crank**2))
theta2_cl = np.arcsin(L_crank/L_coupler)
pos_sl = L_coupler*np.cos(theta1_cl)*np.cos(theta2_cl)


def initial_configuration(p):
    th1_cl, th2_cl, x_sl = p
    eq1 = L_coupler*np.cos(th2_cl)*np.cos(th1_cl) - x_sl
    eq2 = -L_coupler*np.cos(th2_cl)*np.sin(th1_cl) + L_crank*np.cos(theta_cr) + ecc
    eq3 = L_coupler*np.sin(th2_cl) - L_crank * np.sin(theta_cr)
    return (eq1, eq2, eq3)


theta1_cl2, theta2_cl2, x_sl2 =  fsolve(initial_configuration, (0, 0, 0))
#
# print(theta1_cl, theta1_cl2)
# print(theta2_cl, theta2_cl2)
# print(pos_sl, x_sl2)
#
# print(initial_configuration((theta1_cl, theta2_cl, pos_sl)))
# print(initial_configuration((theta1_cl2, theta2_cl2, x_sl2)))

A_vel = np.array([[L_coupler*np.cos(theta2_cl)*np.sin(theta1_cl), L_coupler*np.sin(theta2_cl)*np.cos(theta1_cl), 1],
                  [-L_coupler*np.cos(theta2_cl)*np.cos(theta1_cl), L_coupler*np.sin(theta2_cl)*np.sin(theta1_cl), 0],
                  [0, L_coupler*np.cos(theta2_cl), 0]])

b_vel = np.array([0, L_crank*np.sin(theta_cr)*omega_cr, L_crank*np.cos(theta_cr)*omega_cr])

dtheta1_cl, dtheta2_cl, dx_sl = la.solve(A_vel, b_vel)
x_sol = np.array([dtheta1_cl, dtheta2_cl, dx_sl])

omx_B1 = np.sin(theta2_cl) * dtheta1_cl
omy_B1 = dtheta2_cl
omz_B1 = -np.cos(theta2_cl) * dtheta1_cl

om_B1 = np.array([omx_B1, omy_B1, omz_B1])

Rot_theta1 = np.array([[np.cos(theta1_cl), -np.sin(theta1_cl), 0],
                       [np.sin(theta1_cl), +np.cos(theta1_cl), 0],
                       [0,                                 0, 1]])

dRot_theta1 = np.array([[- np.sin(theta1_cl), - np.cos(theta1_cl), 0],
                       [np.cos(theta1_cl),    - np.sin(theta1_cl), 0],
                       [0,                                      0, 0]])

Rot_theta2 = np.array([[np.cos(theta2_cl), 0,  np.sin(theta2_cl)],
                       [0, 1, 0],
                       [-np.sin(theta2_cl), 0, np.cos(theta2_cl)]])

dRot_theta2 = np.array([[- np.sin(theta2_cl), 0,  np.cos(theta2_cl)],
                       [0, 0, 0],
                       [-np.cos(theta2_cl), 0, - np.sin(theta2_cl)]])

T_I2B = Rot_theta2.T @ Rot_theta1

R_B2I = Rot_theta1.T @ Rot_theta2

om_I1 = R_B2I @ om_B1

dR_B2I = dRot_theta1.T @ Rot_theta2 * dtheta1_cl + Rot_theta1.T @ dRot_theta2 * dtheta2_cl
skew_omB = R_B2I.T @ dR_B2I

r0P_I = np.array([0, 0, L_crank])
x0C_I = np.sqrt(L_coupler**2 - L_crank**2 - ecc**2)
r0C_I = np.array([x0C_I, -ecc, 0])

rCP_I = r0P_I - r0C_I
assert L_coupler == np.linalg.norm(rCP_I)

v0P_I = np.array([0, -omega_cr * L_crank, 0])
v0P_B = T_I2B @ v0P_I
rCP_B = T_I2B @ rCP_I

A_com = np.column_stack((skew(rCP_B)[:, [1, 2]], - T_I2B[:, 0]))
om_vx = np.linalg.solve(A_com, -v0P_B)

omy_B3, omz_B3 = om_vx[:2]
dx_sl2 = om_vx[-1]

assert check_skew_symmetry(skew_omB)
omx_B2 = skew_omB[2,1]
omy_B2 = skew_omB[0,2]
omz_B2 = skew_omB[1,0]

assert omx_B1 == omx_B2
assert abs(omy_B1- omy_B2)<1e-14
assert omz_B1 == omz_B2

print(omy_B1, omy_B3)
print(omz_B1, omz_B3)
print(dx_sl, dx_sl2)


# quat = quaternion.from_rotation_matrix(R_B2I)
#
# e1B = np.array([1, 0, 0])
# e2B = np.array([0, 1, 0])
# e3B = np.array([0, 0, 1])
#
# e1B_quat = np.quaternion(0, 1, 0, 0)
# e2B_quat = np.quaternion(0, 0, 1, 0)
# e3B_quat = np.quaternion(0, 0, 0, 1)
#
# e1I_quat = np.multiply(quat, np.multiply(e1B_quat, np.conjugate(quat)))
#
# e1I = quaternion.as_float_array(e1I_quat)
# print(e1I[0])
#
# rP_I = np.array([0, 0, L_crank])
#
# xC_I = np.sqrt(L_coupler**2 - L_crank**2 - ecc**2)
# rC_I = np.array([xC_I, -ecc, 0])
#
# dir_couplerI = -(rP_I - rC_I)/np.linalg.norm(rP_I - rC_I)
#
# origin = np.array([0, 0, 0]) # origin point
# # e1I = R_B2I @ e1B
# origins = np.zeros((3, 3))
# baseI = np.eye(3)
# baseB = R_B2I @ np.eye(3)
#
# basesBI = np.concatenate((np.eye(3), baseI), axis=0)

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.set_xlabel('$x [m]$', fontsize=fntsize)
# ax.set_ylabel('$y [m]$', fontsize=fntsize)
# ax.set_zlabel('$z [m]$', fontsize=fntsize)
# plt.plot([0, 1],[0, 0],[0, 0], '-b')
# plt.plot([0, 0],[0, 1],[0, 0], '-b')
# plt.plot([0, 0],[0, 0],[0, 1], '-b')
# plt.plot([0, baseB[0, 0]], [0, baseB[1, 0]], [0, baseB[2, 0]], '-r')
# plt.plot([0, baseB[0, 1]], [0, baseB[1, 1]], [0, baseB[2, 1]], '-r')
# plt.plot([0, baseB[0, 2]], [0, baseB[1, 2]], [0, baseB[2, 2]], '-r')
#
# plt.show()
#
# dRtheta1 = dRot_theta1.T @ Rot_theta2
# dRtheta2 = Rot_theta1.T @ dRot_theta2
#
#
# print(R_B2I.T @ dRtheta1)
# print(R_B2I.T @ dRtheta2)
#
# print(np.cos(theta2), -np.sin(theta2))