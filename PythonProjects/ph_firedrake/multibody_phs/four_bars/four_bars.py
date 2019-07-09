import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
fntsize = 15

import numpy as np
import scipy.linalg as la

from modules_phdae.classes_phsystem import SysPhdaeRig
from system_components.beams import FloatFlexBeam, draw_deformation
from multibody_phs.four_bars.fourbars_constants import *
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

    theta_cl = R1[1, 0]
    theta_fl = R1[2, 0]

    return theta_cl, theta_fl


def draw_mechanism(eig_r, eig_p, th1, n_dr):

    nel_r = 3
    nel_p = int(len(eig_p)/3)

    eigcrank_r = eig_r[:nel_r]
    eigcoupler_r = eig_r[nel_r:2 * nel_r]
    eigfollower_r = eig_r[2 * nel_r:]

    eigcrank_p = eig_p[:nel_p]
    eigcoupler_p = eig_p[nel_p:2 * nel_p]
    eigfollower_p = eig_p[2 * nel_p:]

    x_cr, u_cr, w_cr = draw_deformation(n_dr, eigcrank_r, eigcrank_p, L_crank)
    x_cl, u_cl, w_cl = draw_deformation(n_dr, eigcoupler_r, eigcoupler_p, L_coupler)
    x_fl, u_fl, w_fl = draw_deformation(n_dr, eigfollower_r, eigfollower_p, L_follower)

    th2, th3 = configuration(th1, L_ground, L_crank, L_coupler, L_follower)

    R1 = np.array([[np.cos(th1), np.sin(th1)],
                   [-np.sin(th1), np.cos(th1)]])

    R2 = np.array([[np.cos(th2), np.sin(th2)],
                   [-np.sin(th2), np.cos(th2)]])

    R3 = np.array([[np.cos(th3), np.sin(th3)],
                   [-np.sin(th3), np.cos(th3)]])

    n_cr = len(x_cr)
    n_cl = len(x_cl)
    n_fl = len(x_fl)

    xNdef_cr = np.vstack((x_cr, np.zeros_like(x_cr)))
    xNdef_cl = np.vstack((x_cl, np.zeros_like(x_cl)))
    xNdef_fl = np.vstack((x_fl, np.zeros_like(x_fl)))

    xdef_cr = np.vstack((x_cr + u_cr, w_cr))
    xdef_cl = np.vstack((x_cl + u_cl, w_cl))
    xdef_fl = np.vstack((x_fl + u_fl, w_fl))

    xI_cr = np.zeros((2, n_cr))
    xI_cl = np.zeros((2, n_cl))
    xI_fl = np.zeros((2, n_fl))

    xIdef_cr = np.zeros((2, n_cr))
    xIdef_cl = np.zeros((2, n_cl))
    xIdef_fl = np.zeros((2, n_fl))

    xPI_cr = np.zeros((2,))

    for i in range(n_cr):
        xI_cr[:, i] = R1.T @ xNdef_cr[:, i] + xPI_cr
        xIdef_cr[:, i] = R1.T @ xdef_cr[:, i] + xPI_cr

    xPI_cl = xI_cr[:, -1]

    for i in range(n_cl):
        xI_cl[:, i] = R1.T @ R2.T @ xNdef_cl[:, i] + xPI_cl
        xIdef_cl[:, i] = R1.T @ R2.T @ xdef_cl[:, i] + xPI_cl

    xPI_fl = xI_cl[:, -1]

    for i in range(n_fl):
        xI_fl[:, i] = R1.T @ R2.T @ R3.T @ xNdef_fl[:, i] + xPI_fl
        xIdef_fl[:, i] = R1.T @ R2.T @ R3.T @ xdef_fl[:, i] + xPI_fl

    xI_mech = np.hstack((xI_cr, xI_cl, xI_fl))
    xIdef_mech = np.hstack((xIdef_cr, xIdef_cl, xIdef_fl))

    plt.plot(xI_mech[0, :], xI_mech[1, :], 'r', xIdef_mech[0, :], xIdef_mech[1, :], 'b')
    plt.legend(("Undeformed", "Deformed"), shadow=True, fontsize=fntsize)
    plt.xlabel(r'Coordinate $x [m]$', {'fontsize': fntsize})
    plt.ylabel(r'Coordinate $y [m]$', {'fontsize': fntsize})


def construct_mech(n_elem, theta_cr):

    crank = FloatFlexBeam(n_elem, L_crank, rho, A_crank, E, I_crank)
    coupler = FloatFlexBeam(n_elem, L_coupler, rho, A_coupler, E, I_coupler, m_joint=m_link)
    follower = FloatFlexBeam(n_elem, L_follower, rho, A_follower, E, I_follower, m_joint=m_link)

    theta_cl, theta_fl = configuration(theta_cr, L_ground, L_crank, L_coupler, L_follower)

    R1 = np.array([[np.cos(theta_cr), np.sin(theta_cr)],
                   [-np.sin(theta_cr), np.cos(theta_cr)]])

    R2 = np.array([[np.cos(theta_cl), np.sin(theta_cl)],
                   [-np.sin(theta_cl), np.cos(theta_cl)]])

    R3 = np.array([[np.cos(theta_fl), np.sin(theta_fl)],
                   [-np.sin(theta_fl), np.cos(theta_fl)]])

    m_crank = crank.m
    ind_crank = np.array([m_crank - 3, m_crank - 2], dtype=int)
    ind_coupler = np.array([0, 1], dtype=int)

    crank_coupler = SysPhdaeRig.transformer_ordered(crank, coupler, ind_crank, ind_coupler, R2)

    m_int = crank_coupler.m
    ind_cr_coup = np.array([m_int - 3, m_int - 2], dtype=int)
    ind_follower = np.array([0, 1], dtype=int)

    manipulator = SysPhdaeRig.transformer_ordered(crank_coupler, follower, ind_cr_coup, ind_follower, R3)

    E_man = manipulator.E
    J_man = manipulator.J

    n_man = len(E_man)
    m_man = manipulator.m

    ind_cos_man = [0, 1, 2, m_man - 3, m_man - 2]
    ind_u_man = list(set(range(m_man)).difference(set(ind_cos_man)))

    G_man = manipulator.B[:, ind_cos_man]
    B_man = manipulator.B[:, ind_u_man]

    nlmb_ground = len(G_man.T)
    Z_lmb = np.zeros((nlmb_ground, nlmb_ground))

    E_mech = la.block_diag(E_man, Z_lmb)
    J_mech = la.block_diag(J_man, Z_lmb)

    J_mech[:n_man, n_man:] = G_man
    J_mech[n_man:, :n_man] = -G_man.T

    B_mech = np.concatenate((B_man, np.zeros((nlmb_ground, len(B_man.T)))))
    n_mech = len(E_mech)
    nr_mech = manipulator.n_r
    nlmb_mech = manipulator.n_lmb + nlmb_ground
    np_mech = manipulator.n_p
    nq_mech = manipulator.n_q
    fourbars = SysPhdaeRig(n_mech, nlmb_mech, nr_mech, np_mech, nq_mech, E=E_mech, J=J_mech, B=B_mech)

    return fourbars


def compute_eigs(n_om, n_els, theta_cr, draw=False):
    mech = construct_mech(n_els, theta_cr)
    eigenvalues, eigvectors = la.eig(mech.J, mech.E)

    omega_all = np.imag(eigenvalues)
    index = omega_all > 0
    omega = omega_all[index]
    eigvec_omega = eigvectors[:, index]
    perm = np.argsort(omega)
    eigvec_omega = eigvec_omega[:, perm]
    omega.sort()

    np_mech = mech.n_p
    nr_mech = mech.n_r

    eigmech_vel = eigvec_omega[:nr_mech + np_mech]

    if draw:
        abseig = abs(eigmech_vel)

        dup = []
        for i in range(len(abseig)):
            for j in range(n_om):
                dup.append(abseig[i, j])

        max_eig = max(dup)

        for i in range(n_om):
            real_eig = np.real(eigmech_vel[:, i])
            imag_eig = np.imag(eigmech_vel[:, i])

            real_norm = np.linalg.norm(real_eig)
            imag_norm = np.linalg.norm(imag_eig)

            if real_norm > imag_norm:
                eigmech_vel_i = real_eig / (real_norm * max_eig) * 0.1
            else:
                eigmech_vel_i = imag_eig / (imag_norm * max_eig) * 0.1

            eigmech_vel_i = eigmech_vel_i

            eigmech_r = eigmech_vel_i[:nr_mech]
            eigmech_p = eigmech_vel_i[nr_mech:]

            draw_mechanism(eigmech_r, eigmech_p, theta_cr, 50)

            # plt.savefig("Def_theta" + str(int(theta_cr * 180 / pi)) + "_num" + str(i+1) + ".eps")
            plt.show()

    return omega[:n_om], eigmech_vel[:, n_om]


def omegatheta_plot(n_elem, theta_vec, n_om):

    n_ev = len(theta_vec)
    omega_vec = np.zeros((n_ev, n_om))

    for i in range(n_ev):
        om_i, eigs_i = compute_eigs(n_om, n_elem, theta_vec[i])
        omega_vec[i] = om_i

    plt.close("all")
    plt.figure()
    plt.plot(theta_vec*180/pi, omega_vec[:, 0],
             theta_vec*180/pi, omega_vec[:, 1])
    plt.legend(("$\omega_1$", "$\omega_2$"), shadow=True, fontsize=fntsize)
    plt.xlabel(r'Crank angle $\theta$ [deg]', {'fontsize': fntsize})
    plt.ylabel(r'Eigenfrequencies [rad/s]', {'fontsize': fntsize})

    plt.figure()
    plt.plot(theta_vec*180/pi, omega_vec[:, 2], label="$\omega_3$")
    plt.legend(shadow=True, fontsize=fntsize)
    plt.xlabel(r'Crank angle $\theta$ [deg]', {'fontsize': fntsize})
    plt.ylabel(r'Eigenfrequencies [rad/s]', {'fontsize': fntsize})

    plt.show()


n_el = 2
th_vec = np.linspace(0, 2*pi, 50)
n_omega = 6

mech = construct_mech(2, 0)

eigenvalues, eigvectors = la.eig(mech.J, mech.E)

omega_all = np.imag(eigenvalues)
index = omega_all > 0
omega = omega_all[index]
eigvec_omega = eigvectors[:, index]
perm = np.argsort(omega)
eigvec_omega = eigvec_omega[:, perm]
omega.sort()

print(omega)
compute_eigs(n_omega, n_el, 0, draw=True)
# compute_eigs(n_omega, n_el, pi, draw=True)
# omegatheta_plot(n_el, th_vec, n_omega)

