import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import sys
from modules_phdae.classes_phsystem import SysPhdaeRig
from system_components.beams import FloatFlexBeam
from scipy.io import savemat
from system_components.tests.fourbars_constants import *
from math import pi

def configuration(theta2, l1, l2, l3, l4):
    """
    Calculates the angles between bars in a four-bar closed chain.
    The inputs are the cranck angle and the 4 lengths of the bars.
    The outputs are two rotation matrices : two possible configurations of
    the four_bar mechanism
    l1: ground length
    l2: crank length
    l3: coupler length
    l4: follower length
    """

    K1 = l1 / l2
    K2 = l1 / l4
    K3 = (l2**2 - l3**2 + l4**2 + l1**2) / (2 * l4 * l2)
    A = np.cos(theta2) - K1 - K2 * np.cos(theta2) + K3
    B = -2 * np.sin(theta2)
    C = K1 - (K2 + 1) * np.cos(theta2) + K3
    # there might be different configurations(max 2 or maybe no one!)
    if B**2 - 4 * A*C < 0:
        print('This configuration is impossible!')
        R1 = None
        R2 = None
        return R1, R2
    else:
        if A != 0:
            theta4 = 2 * np.arctan((-B - np.sqrt(B**2 - 4 * A * C)) / (2 * A))
            theta42 = 2 * np.arctan((-B + np.sqrt(B**2 - 4 * A * C)) / (2 * A))
            theta3 = np.arctan((l4*np.sin(theta4) - l2*np.sin(theta2))/(l1 - l2*np.cos(theta2) + l4*np.cos(theta4)))
            theta32 = np.arctan((l4*np.sin(theta42) - l2*np.sin(theta2))/(l1 - l2*np.cos(theta2) + l4*np.cos(theta42)))
        else:
            theta4 = 2 * np.arctan(-C / B)
            theta42 = 0
            theta3 = np.arctan((l4*np.sin(theta4) - l2*np.sin(theta2)) / (l1 - l2*np.cos(theta2) + l4*np.cos(theta4)))
            theta32 = 0

    # Matrix R output:
    R1 = np.zeros((3, 3))
    R2 = np.zeros((3, 3))
    a = theta2
    R1[0, :] = np.array([a, 0, 0])
    R2[0, :] = np.array([a, 0, 0])
    b1 = theta3 - theta2
    b2 = theta32 - theta2
    # if (b1 > pi / 2) | (b2 > pi / 2):
    #     b1 = b1 - pi
    #     b2 = b2 - pi

    R1[1, :] = np.array([b1, 0, 0])
    R2[1, :] = np.array([b2, 0, 0])
    c1 = -pi - theta3 + theta4
    c2 = -pi - theta3 + theta42
    # if (c1 > pi / 2) | (c2 > pi / 2):
    #     c1 = c1 - pi
    #     c2 = c2 - pi

    R1[2, :] = np.array([c1, 0, 0])
    R2[2, :] = np.array([c2, 0, 0])

    return R1, R2

n_el = 2


crank = FloatFlexBeam(n_el, L_crank, rho, A_crank, E, I_crank)
E_hinged = crank.E[2:, 2:]
J_hinged = crank.J[2:, 2:]
B_hinged = crank.B[2:, 2:]
crank_hinged = SysPhdaeRig(len(E_hinged), 0, 1, crank.n_p, crank.n_q,
                           E=E_hinged, J=J_hinged, B=B_hinged)


coupler = FloatFlexBeam(n_el, L_coupler, rho, A_coupler, E, I_coupler, m_joint=m_link)
follower = FloatFlexBeam(n_el, L_follower, rho, A_follower, E, I_follower, m_joint=m_link)

theta1 = 0
R1 = np.array([[np.cos(theta1), np.sin(theta1)],
              [-np.sin(theta1), np.cos(theta1)]])

[r1, r2] = configuration(theta1, L_ground, L_crank, L_coupler, L_follower)

theta2 = r1[1, 0]
theta3 = r1[2, 0]

# print(theta2*180/pi)
# print(theta3*180/pi)

R2 = np.array([[np.cos(theta2), np.sin(theta2)],
              [-np.sin(theta2), np.cos(theta2)]])

R3 = np.array([[np.cos(theta3), np.sin(theta3)],
              [-np.sin(theta3), np.cos(theta3)]])


ind_crank = np.array([1, 2], dtype=int)
ind_coupler = np.array([0, 1], dtype=int)

crank_coupler = SysPhdaeRig.transformer_ordered(crank_hinged, coupler, ind_crank, ind_coupler, R2)


n_int = crank_coupler.m
ind_cr_coup = np.array([n_int-3, n_int-2], dtype=int)
ind_follower = np.array([0, 1], dtype=int)

manipulator = SysPhdaeRig.transformer_ordered(crank_coupler, follower, ind_cr_coup, ind_follower, R3)

# print(manipulator.G_f)

E_man = manipulator.E
J_man = manipulator.J

n_man = len(E_man)
rankE = np.linalg.matrix_rank(E_man)


m_man = manipulator.m
print(rankE, n_man, m_man)
G_follower = manipulator.B[:, [m_man-3, m_man-2]]

nlmb_ground = 2
Z_lmb = np.zeros((nlmb_ground, nlmb_ground))

E_mech = la.block_diag(E_man, Z_lmb)
J_mech = la.block_diag(J_man, Z_lmb)

J_mech[:n_man, n_man:] = G_follower
J_mech[n_man:, :n_man] = -G_follower.T

eigenvalues, eigvectors = la.eig(J_mech, E_mech)
omega_all = np.imag(eigenvalues)
index = omega_all > 0
omega = omega_all[index]
eigvec_omega = eigvectors[:, index]
perm = np.argsort(omega)
eigvec_omega = eigvec_omega[:, perm]
omega.sort()

print(omega)



#
# J_dae = sys_dae.J
# E_dae = sys_dae.E
# B_dae = sys_dae.B
#
# sys_ode, T = sys_dae.dae_to_ode()
#
# J_ode = sys_ode.J
# Q_ode = sys_ode.Q
# B_ode = sys_ode.B
#
# pathout = '/home/a.brugnoli/GitProjects/MatlabProjects/PH/PH_TITOP/TwoLinks_Manipulator/Matrices_manipulator/'
# Qode_file = 'Q_ode'; Jode_file = 'J_ode'; Bode_file = 'B_ode'
# savemat(pathout + Qode_file, mdict={Qode_file: Q_ode})
# savemat(pathout + Jode_file, mdict={Jode_file: J_ode})
# savemat(pathout + Bode_file, mdict={Bode_file: B_ode})
#
# Edae_file = 'E_dae'; Jdae_file = 'J_dae'; Bdae_file = 'B_dae'
# savemat(pathout + Edae_file, mdict={Edae_file: E_dae})
# savemat(pathout + Jdae_file, mdict={Jdae_file: J_dae})
# savemat(pathout + Bdae_file, mdict={Bdae_file: B_dae})

# plt.figure(); plt.spy(sys_int1.E)
# plt.figure(); plt.spy(sys_int1.J)
# plt.figure(); plt.spy(sys_all.E)
# plt.figure(); plt.spy(sys_all.J)
# plt.show()

# plt.figure()
# plt.spy(J_all)
# plt.figure()
# plt.spy(M_all)
# plt.show()
#




