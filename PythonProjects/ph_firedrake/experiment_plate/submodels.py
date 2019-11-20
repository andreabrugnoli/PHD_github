# Convergence test for HHJ

from firedrake import *
import numpy as np
np.set_printoptions(threshold=np.inf)
from math import pi, floor

import scipy.linalg as la

from ufl import algebra as ufl_alg
from modules_phdae.classes_phsystem import SysPhdaeRig

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

plt.close('all')
matplotlib.rcParams['text.usetex'] = True

from scipy.io import savemat

def mirror_model():

    n_mirror = 3

    m_mirror = 24.3 * 1e-3  # kg
    Jx_mirror = 1.968 * 1e-6  # kg/m^2
    Jy_mirror = 1.968 * 1e-6  # kg/m^2

    r_EL = 0.01*np.array([0, 1.44, 0.81])  # m

    M_mirror = np.diag([m_mirror, Jx_mirror, Jy_mirror])

    B_mirror = np.array([[1,        0, 0],
                         [-r_EL[1], 1, 0],
                         [+r_EL[0], 0, 1]])

    mirror = SysPhdaeRig(n_mirror, 0, n_mirror, 0, 0, E=M_mirror, B=B_mirror)

    return mirror


def actuator_model():

    m_mov = 23.5 * 1e-3  # kg
    m_case = 96.5 * 1e-3  # kg

    Jx_case = 114.04 * 1e-6  # kg/m^2
    Jy_case = 114.04 * 1e-6  # kg/m^2

    a = 1.5  # N/V

    k_mov = 26  # N/m
    c_mov = 10  # Ns/s




def plate_model(nx, ny, r):

    E = Constant(69 * 10**9) # Pa
    rho = Constant(2692)  # kg/m^3
    nu = Constant(0.33)
    h = Constant(0.003)

    Lx = 0.04
    Ly = 0.3

    D = Constant(E * h ** 3 / (1 - nu ** 2) / 12)
    fl_rot = Constant(12 / (E * h ** 3))

    # x_P1 = Constant(0.01)  # 1cm
    # y_P1 = Constant(0.25)  # 25cm
    x_P1 = 0.01 # 1cm
    y_P1 = 0.25  # 25cm
    pos_P1 = as_vector([x_P1, y_P1])

    # x_P2 = Constant(0.03)
    # y_P2 = Constant(0.105)
    x_P2 = 0.03
    y_P2 = 0.105
    pos_P2 = as_vector([x_P2, y_P2])

    x_P3 = Constant(0.02)
    y_P3 = Constant(0.28)
    pos_P3 = as_vector([x_P3, y_P3])

    mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=False)

    hx_mesh = Lx/nx
    hy_mesh = Ly/ny
    x, y = SpatialCoordinate(mesh)

    # Operators and functions
    def delta_app(P0):
        x_0, y_0 = P0

        n_h = 2

        tol_x = hx_mesh/n_h
        tol_y = hy_mesh/n_h

        set_x = conditional(le(ufl_alg.Abs(x-x_0), tol_x), 1, 0)
        set_y = conditional(le(ufl_alg.Abs(y-y_0), tol_y), 1, 0)

        area = 4*tol_x*tol_y

        delta = set_x*set_y/area
        # delta = 1./(eps*2*pi)*exp(-0.5*((x - x_0)**2 + (y - y_0)**2)/eps**2)
        # delta = eps/pi/((x - x_0)**2 + (y - y_0)**2 + eps**2)

        return delta


    def bending_curv(momenta):
        kappa = fl_rot * ((1+nu)*momenta - nu * Identity(2) * tr(momenta))
        return kappa

    def j_operator(v_p, v_q, e_p, e_q):

        j_form = - inner(grad(grad(v_p)), e_q) * dx \
        + jump(grad(v_p), n_ver) * dot(dot(e_q('+'), n_ver('+')), n_ver('+')) * dS \
        + dot(grad(v_p), n_ver) * dot(dot(e_q, n_ver), n_ver) * ds \
        + inner(v_q, grad(grad(e_p))) * dx \
        - dot(dot(v_q('+'), n_ver('+')), n_ver('+')) * jump(grad(e_p), n_ver) * dS \
        - dot(dot(v_q, n_ver), n_ver) * dot(grad(e_p), n_ver) * ds

        return j_form


    # Finite element defition

    Vp = FunctionSpace(mesh, 'CG', r)
    Vq = FunctionSpace(mesh, 'HHJ', r-1)
    V = Vp * Vq

    n_p = Vp.dim()
    n_q = Vq.dim()
    n_V = V.dim()

    v = TestFunction(V)
    v_p, v_q = split(v)

    e = TrialFunction(V)
    e_p, e_q = split(e)

    al_p = rho * h * e_p
    al_q = bending_curv(e_q)

    dx = Measure('dx')
    ds = Measure('ds')
    dS = Measure("dS")

    m_form = inner(v_p, al_p) * dx + inner(v_q, al_q) * dx

    n_ver = FacetNormal(mesh)
    s_ver = as_vector([-n_ver[1], n_ver[0]])

    j_form = j_operator(v_p, v_q, e_p, e_q)

    bcs = []

    bcC_d = DirichletBC(V.sub(0), Constant(0.0), 3)

    bcF_l = DirichletBC(V.sub(1), Constant(((0.0, 0.0), (0.0, 0.0))), 1)
    bcF_r = DirichletBC(V.sub(1), Constant(((0.0, 0.0), (0.0, 0.0))), 2)
    bcF_u = DirichletBC(V.sub(1), Constant(((0.0, 0.0), (0.0, 0.0))), 4)

    bcs.append(bcC_d)
    bcs.append(bcF_l)
    bcs.append(bcF_r)
    bcs.append(bcF_u)

    bd_dof_p = []
    bd_dof_q = []

    for node in bcC_d.nodes:
        bd_dof_p.append(node)

    bd_dof_p = sorted(bd_dof_p)
    np_lmb = len(bd_dof_p)

    Gp = np.zeros((n_p, np_lmb))
    for (i, j) in enumerate(bd_dof_p):
        Gp[j, i] = 1

    for node in bcF_l.nodes:
        bd_dof_q.append(node)
    for node in bcF_r.nodes:
        bd_dof_q.append(node)
    for node in bcF_u.nodes:
        bd_dof_q.append(node)

    bd_dof_q = sorted(bd_dof_q)
    nq_lmb = len(bd_dof_q)

    Gq = np.zeros((n_q, nq_lmb))
    for (i, j) in enumerate(bd_dof_q):
        Gq[j, i] = 1

    n_lmb = np_lmb + nq_lmb
    G = la.block_diag(Gp, Gq)
    G_ortho = la.null_space(G.T).T

    J = assemble(j_form, mat_type='aij')
    M = assemble(m_form, mat_type='aij')

    petsc_j = J.M.handle
    petsc_m = M.M.handle

    JJ = np.array(petsc_j.convert("dense").getDenseArray())
    MM = np.array(petsc_m.convert("dense").getDenseArray())

    fz_P1 = v_p*delta_app(pos_P1)*dx
    mx_P1 = v_p.dx(1)*delta_app(pos_P1)*dx
    my_P1 = -v_p.dx(0)*delta_app(pos_P1)*dx

    fz_P2 = v_p * delta_app(pos_P2) * dx
    mx_P2 = v_p.dx(1) * delta_app(pos_P2) * dx
    my_P2 = -v_p.dx(0) * delta_app(pos_P2) * dx

    fz_P3 = v_p * delta_app(pos_P3) * dx
    mx_P3 = v_p.dx(1) * delta_app(pos_P3) * dx
    my_P3 = -v_p.dx(0) * delta_app(pos_P3) * dx

    Fz_P1 = assemble(fz_P1).vector().get_local().reshape((-1, 1))
    Mx_P1 = assemble(mx_P1).vector().get_local().reshape((-1, 1))
    My_P1 = assemble(my_P1).vector().get_local().reshape((-1, 1))

    Fz_P2 = assemble(fz_P2).vector().get_local().reshape((-1, 1))
    Mx_P2 = assemble(mx_P2).vector().get_local().reshape((-1, 1))
    My_P2 = assemble(my_P2).vector().get_local().reshape((-1, 1))

    Fz_P3 = assemble(fz_P3).vector().get_local().reshape((-1, 1))
    Mx_P3 = assemble(mx_P3).vector().get_local().reshape((-1, 1))
    My_P3 = assemble(my_P3).vector().get_local().reshape((-1, 1))

    B = np.hstack((Fz_P1, Mx_P1, My_P1, Fz_P2, Mx_P2, My_P2, Fz_P3, Mx_P3, My_P3))
    # B = Fz_P1

    # tab_coord = mesh.coordinates.dat.data
    # x_cor = tab_coord[:, 0]
    # y_cor = tab_coord[:, 1]
    #
    # ind_x1 = np.isclose(x_cor, x_P1)
    # ind_y1 = np.isclose(y_cor, y_P1)
    #
    # ind_P1, = np.where(np.logical_and(ind_x1, ind_y1))

    # print(max(abs(B)))
    # print(B)
    # print(np.where(B == max(B))[0])
    # print(ind_P1)

    Z_lmb = np.zeros((n_lmb, n_lmb))
    J_aug = np.vstack([np.hstack([JJ, G]),
                       np.hstack([-G.T, Z_lmb])
                       ])

    M_aug = la.block_diag(MM, Z_lmb)

    B_aug = np.concatenate((B, np.zeros((n_lmb, B.shape[1]))))

    plate = SysPhdaeRig(n_V+n_lmb, n_lmb, 0, n_p, n_q, E=M_aug, J=J_aug, B=B_aug)



    return plate, Vp


plate, Vp = plate_model(4, 30, 1)
mirror, Vp = mirror_model()

pl_mir = SysPhdaeRig.transformer_ordered(plate, mirror, [6, 7, 8], [0, 1, 2], np.eye(3))





# eigenvalues, eigvectors = la.eig(J_aug, M_aug)
# omega_all = np.imag(eigenvalues)
#
# tol = 10 ** (-9)
# index = omega_all >= tol
#
# omega = omega_all[index]
# eigvec_omega = eigvectors[:, index]
# perm = np.argsort(omega)
# eigvec_omega = eigvec_omega[:, perm]
#
# omega.sort()
#
# n_om = 5
#
# for i in range(n_om):
#     print(omega[i])
#
# if plot_eig:
#
#     n_fig = n_om
#     fntsize = 16
#
#     for i in range(n_fig):
#         eig_real_w = Function(Vp)
#         eig_imag_w = Function(Vp)
#
#         eig_real_p = np.real(eigvec_omega[:n_p, i])
#         eig_imag_p = np.imag(eigvec_omega[:n_p, i])
#         eig_real_w.vector()[:] = eig_real_p
#         eig_imag_w.vector()[:] = eig_imag_p
#
#         norm_real_eig = np.linalg.norm(eig_real_w.vector().get_local())
#         norm_imag_eig = np.linalg.norm(eig_imag_w.vector().get_local())
#
#         figure = plt.figure(i)
#         ax = figure.add_subplot(111, projection="3d")
#
#         ax.set_xbound(-tol, 1 + tol)
#         ax.set_xlabel('$x [m]$', fontsize=fntsize)
#
#         ax.set_ybound(-tol, 1 + tol)
#         ax.set_ylabel('$y [m]$', fontsize=fntsize)
#
#         ax.set_title('$v_{e_{w}}$', fontsize=fntsize)
#
#         ax.w_zaxis.set_major_locator(LinearLocator(10))
#         ax.w_zaxis.set_major_formatter(FormatStrFormatter('%1.2g'))
#
#         if norm_imag_eig > norm_real_eig:
#             plot(eig_imag_w, axes=ax, plot3d=True)
#         else:
#             plot(eig_real_w, axes=ax, plot3d=True)
#
# plt.show()

# plate_ode, T = plate.dae_to_odeCE(mass=True)[:2]
#
# print(plate_ode.n)
#
# plate_red, V_f = plate_ode.reduce_system(1e-6, 3)
#
# print(plate_red.n)
#
# M_full = plate_ode.E
# J_full = plate_ode.J
# B_full = plate_ode.B
#
# M_red = plate_red.E
# J_red = plate_red.J
# B_red = plate_red.B
#
# pathout = '/home/a.brugnoli/GitProjects/MatlabProjects/PH/ReductionPHDAE/KP_Experiment/'
# M_file = 'M'; Q_file = 'Q'; J_file = 'J'; B_file = 'B'
# savemat(pathout + M_file, mdict={M_file: M_full})
# savemat(pathout + J_file, mdict={J_file: J_full})
# savemat(pathout + B_file, mdict={B_file: B_full})
#
# Mr_file = 'Mr'; Qr_file = 'Qr'; Jr_file = 'Jr'; Br_file = 'Br'
# savemat(pathout + Mr_file, mdict={Mr_file: M_red})
# savemat(pathout + Jr_file, mdict={Jr_file: J_red})
# savemat(pathout + Br_file, mdict={Br_file: B_red})



