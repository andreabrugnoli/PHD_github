import quaternion
from scipy.spatial.transform import Rotation
import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
fntsize = 15
from mpl_toolkits.mplot3d import Axes3D


L_crank = 0.15
L_coupler = 0.3
ecc = 0.1

# alpha1 = np.arctan2(L_crank, ecc)
# alpha2 = np.arcsin(np.sqrt(ecc**2 + L_crank**2)/L_coupler)
#
# Rot_alpha1 = np.array([[1, 0, 0],
#                        [0, np.cos(alpha1), -np.sin(alpha1)],
#                        [0, np.sin(alpha1), np.cos(alpha1)]])
#
# Rot_alpha2 = np.array([[np.cos(alpha2), -np.sin(alpha2), 0],
#                        [np.sin(alpha2), +np.cos(alpha2), 0],
#                        [0, 0, 1]])
#
# T_I2B = Rot_alpha2 @ Rot_alpha1.T

theta1 = np.arcsin(ecc/np.sqrt(L_coupler**2 - L_crank**2))
theta2 = np.arcsin(L_crank/L_coupler)

Rot_theta1 = np.array([[np.cos(theta1), -np.sin(theta1), 0],
                       [np.sin(theta1), +np.cos(theta1), 0],
                       [0, 0, 1]])

Rot_theta2 = np.array([[np.cos(theta2), 0,  np.sin(theta2)],
                       [0, 1, 0],
                       [-np.sin(theta2), 0, np.cos(theta2)]])

T_I2B = Rot_theta2.T @ Rot_theta1

R_B2I = T_I2B.T

quat = quaternion.from_rotation_matrix(R_B2I)

e1B = np.array([1, 0, 0])
e2B = np.array([0, 1, 0])
e3B = np.array([0, 0, 1])

e1B_quat = np.quaternion(0, 1, 0, 0)
e2B_quat = np.quaternion(0, 0, 1, 0)
e3B_quat = np.quaternion(0, 0, 0, 1)

e1I_quat = np.multiply(quat, np.multiply(e1B_quat, np.conjugate(quat)))

e1I = quaternion.as_float_array(e1I_quat)
print(e1I[0])

rP_I = np.array([0, 0, L_crank])

xC_I = np.sqrt(L_coupler**2 - L_crank**2 - ecc**2)
rC_I = np.array([xC_I, -ecc, 0])

dir_couplerI = -(rP_I - rC_I)/np.linalg.norm(rP_I - rC_I)

origin = np.array([0, 0, 0]) # origin point
# e1I = R_B2I @ e1B
origins = np.zeros((3, 3))
baseI = np.eye(3)
baseB = R_B2I @ np.eye(3)

basesBI = np.concatenate((np.eye(3), baseI), axis=0)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('$x [m]$', fontsize=fntsize)
ax.set_ylabel('$y [m]$', fontsize=fntsize)
ax.set_zlabel('$z [m]$', fontsize=fntsize)
# plt.quiver(origins[0], origins[1], origins[2], baseI[0], baseI[1], baseI[2], color=['b', 'b', 'b'], length=1, normalize=True)
# plt.quiver(origins[0], origins[1], origins[2], baseB[0], baseB[1], baseB[2], color=['r', 'r', 'r'], length=1, normalize=True)
# plt.quiver(origin[0], origin[1], origin[2], baseI[0, 0], baseI[1, 0], baseI[2, 0], color=['b'], length=1, normalize=True)
# plt.quiver(origin[0], origin[1], origin[2], baseB[0, 0], baseB[1, 0], baseB[2, 0], color=['r'], length=1, normalize=True)
# plt.quiver(origin[0], origin[1], origin[2], dir_couplerI[0], dir_couplerI[1], dir_couplerI[2], color=['g'], length=1, normalize=True)
plt.quiver(origin[0], origin[1], origin[2], e1I[1], e1I[2], e1I[3], color=['b'], length=1, normalize=True)

plt.show()