import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

from modules_phdae.classes_phsystem import SysPhdaeRig
from system_components.beams import FloatFlexBeam, draw_deformation
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

R2 = np.array([[np.cos(theta2), np.sin(theta2)],
              [-np.sin(theta2), np.cos(theta2)]])

R3 = np.array([[np.cos(theta3), np.sin(theta3)],
              [-np.sin(theta3), np.cos(theta3)]])


ind_crank = np.array([1, 2], dtype=int)
ind_coupler = np.array([0, 1], dtype=int)

crank_coupler = SysPhdaeRig.transformer_ordered(crank_hinged, coupler, ind_crank, ind_coupler, R2)

m_int = crank_coupler.m
ind_cr_coup = np.array([m_int-3, m_int-2], dtype=int)
ind_follower = np.array([0, 1], dtype=int)

manipulator = SysPhdaeRig.transformer_ordered(crank_coupler, follower, ind_cr_coup, ind_follower, R3)

E_man = manipulator.E
J_man = manipulator.J

n_man = len(E_man)
m_man = manipulator.m

ind_cos_man = [5, 6, 0, 3]  #  [0, m_man-3, m_man-2]
ind_u_man = list(set(range(m_man)).difference(set(ind_cos_man)))

G_man = manipulator.B[:, ind_cos_man]
B_man = manipulator.B[:, ind_u_man]

nlmb_ground = len(G_man.T)
Z_lmb = np.zeros((nlmb_ground, nlmb_ground))

E_mech = la.block_diag(E_man, Z_lmb)
J_mech = la.block_diag(J_man, Z_lmb)

J_mech[:n_man, n_man:] = G_man
J_mech[n_man:, :n_man] = -G_man.T

# B_mech = np.concatenate((B_man, np.zeros((nlmb_ground, len(B_man.T)))))
# n_mech = len(E_mech)
# nr_mech = manipulator.n_r
# nlmb_mech = manipulator.n_lmb + nlmb_ground
# np_mech = manipulator.n_p
# nq_mech = manipulator.n_q
# mech = SysPhdaeRig(n_mech, nlmb_mech, nr_mech, np_mech, nq_mech, E=E_mech, J=J_mech, B=B_mech)
# mech_ode = mech.dae_to_ode()
# A_mech = mech_ode.J @ mech_ode.Q
# eigenvalues, eigvectors = la.eig(A_mech)

eigenvalues, eigvectors = la.eig(J_mech, E_mech)

omega_all = np.imag(eigenvalues)
index = omega_all > 0
omega = omega_all[index]
eigvec_omega = eigvectors[:, index]
perm = np.argsort(omega)
eigvec_omega = eigvec_omega[:, perm]
omega.sort()

np_mech = manipulator.n_p
nel_p = int(np_mech/3)  # Three elements compose the mechanism
nel_dof = int(nel_p/3)

nr_mech = manipulator.n_r

eigmech_p = eigvec_omega[nr_mech:nr_mech+np_mech]
i=0
# for i in range(0):
real_eig = np.real(eigmech_p[:, i])
imag_eig = np.imag(eigmech_p[:, i])

if np.linalg.norm(real_eig) > np.linalg.norm(imag_eig):
    eigmech_p_i = real_eig
else:
    eigmech_p_i = imag_eig

eigcrank_p = eigmech_p_i[:nel_p]
eigcoupler_p = eigmech_p_i[nel_p:2*nel_p]
eigfollower_p = eigmech_p_i[2*nel_p:]

eigu_crank = eigcrank_p[:nel_dof]
eigw_crank = eigcrank_p[nel_dof:]

eigu_coupler = eigcoupler_p[:nel_dof]
eigw_coupler = eigcoupler_p[nel_dof:]

eigu_follower = eigfollower_p[:nel_dof]
eigw_follower = eigfollower_p[nel_dof:]

n_draw = 50

x_crank, u_crank, w_crank = draw_deformation(n_draw, eigu_crank, eigw_crank, L_crank)
#
# fig = plt.figure(1); ax = fig.add_subplot(111)
# ax.plot(x_crank, np.zeros_like(x_crank), 'r', label="Undeformed")
# ax.plot(x_crank + u_crank, w_crank, 'b', label="Deformed")
# ax.legend()
#
# x_coupler, u_coupler, w_coupler = draw_deformation(n_draw, eigu_coupler, eigw_coupler, L_coupler)
# fig = plt.figure(2); ax = fig.add_subplot(111)
# plt.plot(x_coupler, np.zeros_like(x_coupler), 'r', label="Undeformed")
# plt.plot(x_coupler + u_coupler, w_coupler, 'b', label="Deformed")
# ax.legend()
#
# x_follower, u_follower, w_follower = draw_deformation(n_draw, eigu_follower, eigw_follower, L_follower)
# fig = plt.figure(3); ax = fig.add_subplot(111)
# plt.plot(x_follower, np.zeros_like(x_follower), 'r', label="Undeformed")
# plt.plot(x_follower + u_follower, w_follower, 'b', label="Deformed")
# ax.legend()
#
# plt.show()

print(omega)
