# Convergence test for HHJ

from firedrake import *
import numpy as np
np.set_printoptions(threshold=np.inf)
from math import pi, floor

import scipy.linalg as la

from ufl import algebra as ufl_alg
from modules_ph.classes_phsystem import SysPhdaeRig

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from firedrake.plot import _two_dimension_triangle_func_val
from mpl_toolkits.mplot3d import Axes3D

plt.close('all')
matplotlib.rcParams['text.usetex'] = True

from scipy.io import savemat

Lx = 0.04
Ly = 0.3

r_EL = 0.01 * np.array([0, 0, 0.81])  # m
r_PA = 0.01 * np.array([0, 0, 3.08])  # m

# r_EL = 0.01 * np.array([0, 1.44, 0.81])  # m
# r_PA = 0.01 * np.array([0, 0, 1.35])  # m

x_P1 = 0.01  # m
y_P1 = 0.25  # m

x_P2 = 0.03  # m
y_P2 = 0.105  # m

x_E = 0.02  # m
y_E = 0.28  # m

r_P1 = np.array([x_P1, y_P1, 0])
r_P2 = np.array([x_P2, y_P2, 0])
r_E = np.array([x_E, y_E, 0])

r_A1 = np.array([x_P1, y_P1, r_PA[2]])
r_A2 = np.array([x_P2, y_P2, r_PA[2]])

def mirror_model():

    n_mirror = 3

    m_mirror = 23.4 * 1e-3  # kg
    Jx_mirror = 8 * 1e-6  # kg/m^2
    Jy_mirror = 58 * 1e-6  # kg/m^2

    # m_mirror = 24.3 * 1e-3  # kg
    # Jx_mirror = 1.96 * 1e-6  # kg/m^2
    # Jy_mirror = 56.266 * 1e-6  # kg/m^2

    M_mirror = np.diag([m_mirror, Jx_mirror, Jy_mirror])

    s_x = - 0.338 * 1e-3  # kg/m^3
    M_mirror[0, 1] = s_x
    M_mirror[1, 0] = s_x

    J_mirror = np.zeros((n_mirror, n_mirror))

    B_mirror = np.array([[1,        0, 0],
                         [-r_EL[1], 1, 0],
                         [+r_EL[0], 0, 1]])

    mirror = SysPhdaeRig(n_mirror, 0, 0, n_mirror, 0, E=M_mirror, J=J_mirror, B=B_mirror)

    return mirror


def actuator_model():

    n_actuator = 5
    mB_actuator = 4

    m_mov = 23.5 * 1e-3  # kg
    m_case = 182.7 * 1e-3  # kg

    Jx_case = 0.3 * 1e-3  # kg/m^2
    Jy_case = 0.3 * 1e-3  # kg/m^2

    # m_mov = 23.6 * 1e-3  # kg
    # m_case = 96.5 * 1e-3  # kg
    #
    # Jx_case = 114.04 * 1e-6  # kg/m^2
    # Jy_case = 114.04 * 1e-6  # kg/m^2

    a = 30  # N/V

    k_mov = 26  * (2*pi)**2  # N/m
    c_mov = 10  # Ns/s

    M_actuator = np.diag([k_mov, m_mov, m_case, Jx_case, Jy_case])

    J_actuator = np.zeros((n_actuator, n_actuator))
    J_actuator[0, 1:3] = np.array([k_mov, -k_mov])
    J_actuator[1:3, 0] = -np.array([k_mov, -k_mov])

    R_actuator = np.zeros((n_actuator, n_actuator))
    R_actuator[1:3, 1:3] = np.array([[c_mov, -c_mov],
                                     [-c_mov, c_mov]])

    B_actuator = np.zeros((n_actuator, mB_actuator))

    tau_PA = np.array([[1,        0, 0],
                       [-r_PA[1], 1, 0],
                       [+r_PA[0], 0, 1]])

    B_actuator[2:, :3] = tau_PA
    B_actuator[[1, 2], 3] = np.array([-a, a])

    # print(B_actuator)

    actuator = SysPhdaeRig(n_actuator, 0, 0, n_actuator, 0,\
                           E=M_actuator, J=J_actuator, R=R_actuator, B=B_actuator)

    return actuator


def plate_model(nx, ny, r):

    E = Constant(69 * 10**9)  # Pa
    rho = Constant(2692)  # kg/m^3
    nu = Constant(0.33)
    h = Constant(0.003)

    D = Constant(E * h ** 3 / (1 - nu ** 2) / 12)
    fl_rot = Constant(12 / (E * h ** 3))

    # x_P1 = Constant(0.01)  # 1cm
    # y_P1 = Constant(0.25)  # 25cm
    pos_P1 = as_vector([x_P1, y_P1])

    # x_P2 = Constant(0.03)
    # y_P2 = Constant(0.105)
    pos_P2 = as_vector([x_P2, y_P2])
    pos_E = as_vector([x_E, y_E])

    mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=False)

    hx_mesh = Lx/nx
    hy_mesh = Ly/ny
    x, y = SpatialCoordinate(mesh)

    # Operators and functions
    def delta_app(P0):
        x_0, y_0 = P0

        n_h = 1

        tol_x = hx_mesh/n_h
        tol_y = hy_mesh/n_h

        set_x = conditional(le(ufl_alg.Abs(x-x_0), tol_x), 1, 0)
        set_y = conditional(le(ufl_alg.Abs(y-y_0), tol_y), 1, 0)
        area = 4*tol_x*tol_y

        # if y_0 == y_E:
        #     area = area/2

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

    fz_E = v_p * delta_app(pos_E) * dx
    mx_E = v_p.dx(1) * delta_app(pos_E) * dx
    my_E = -v_p.dx(0) * delta_app(pos_E) * dx

    Fz_P1 = assemble(fz_P1).vector().get_local().reshape((-1, 1))
    Mx_P1 = assemble(mx_P1).vector().get_local().reshape((-1, 1))
    My_P1 = assemble(my_P1).vector().get_local().reshape((-1, 1))

    Fz_P2 = assemble(fz_P2).vector().get_local().reshape((-1, 1))
    Mx_P2 = assemble(mx_P2).vector().get_local().reshape((-1, 1))
    My_P2 = assemble(my_P2).vector().get_local().reshape((-1, 1))

    Fz_E = assemble(fz_E).vector().get_local().reshape((-1, 1))
    Mx_E = assemble(mx_E).vector().get_local().reshape((-1, 1))
    My_E = assemble(my_E).vector().get_local().reshape((-1, 1))

    # print('sum Fz_P1: ' + str(sum(Fz_P1[:, 0])))
    # print('sum Fz_P2: ' + str(sum(Fz_P2[:, 0])))
    # print('sum Fz_E: ' + str(sum(Fz_E[:, 0])))

    # tab_coord = mesh.coordinates.dat.data
    # x_cor = tab_coord[:, 0]
    # y_cor = tab_coord[:, 1]
    #
    # ind_xP1 = np.isclose(x_cor, x_P1)
    # ind_yP1 = np.isclose(y_cor, y_P1)
    #
    # ind_P1, = np.where(np.logical_and(ind_xP1, ind_yP1))
    # print(ind_P1, x_cor[ind_P1], y_cor[ind_P1])
    # Fz_P1 = np.zeros((n_V, 1))
    # Fz_P1[ind_P1] = 1
    #
    # ind_xP2 = np.isclose(x_cor, x_P2)
    # ind_yP2 = np.isclose(y_cor, y_P2)
    #
    # ind_P2, = np.where(np.logical_and(ind_xP2, ind_yP2))
    # print(ind_P2, x_cor[ind_P2], y_cor[ind_P2])
    # Fz_P2 = np.zeros((n_V, 1))
    # Fz_P2[ind_P2] = 1
    #
    # ind_xE = np.isclose(x_cor, x_E)
    # ind_yE = np.isclose(y_cor, y_E)
    #
    # ind_E, = np.where(np.logical_and(ind_xE, ind_yE))
    # print(ind_E, x_cor[ind_E], y_cor[ind_E])
    # Fz_E = np.zeros((n_V, 1))
    # Fz_E[ind_E] = 1

    B = np.hstack((Fz_P1, Mx_P1, My_P1, Fz_P2, Mx_P2, My_P2, Fz_E, Mx_E, My_E))
    # B = np.hstack((Fz_P1, Fz_P2, Fz_E))

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


plate, Vp = plate_model(8, 30, 1)
np_plate = Vp.dim()

print(plate.n)
mirror = mirror_model()
actuator = actuator_model()

np_mir = mirror.n_p
np_act = actuator.n_p

pl_mir = SysPhdaeRig.transformer_ordered(plate, mirror, [6, 7, 8], [0, 1, 2], np.eye(3))
plmir_act2 = SysPhdaeRig.transformer_ordered(pl_mir, actuator, [3, 4, 5], [0, 1, 2], np.eye(3))
model_all = SysPhdaeRig.transformer_ordered(plmir_act2, actuator, [0, 1, 2], [0, 1, 2], np.eye(3))

J_sys = model_all.J
M_sys = model_all.E
R_sys = model_all.R

# print(model_all.B[np_plate:model_all.n_p])

# J_act = J_sys[np_plate+np_mir:np_plate+np_mir+np_act, np_plate+np_mir:np_plate+np_mir+np_act]
# R_act = R_sys[np_plate+np_mir:np_plate+np_mir+np_act, np_plate+np_mir:np_plate+np_mir+np_act]
# M_act = M_sys[np_plate+np_mir:np_plate+np_mir+np_act, np_plate+np_mir:np_plate+np_mir+np_act]
#
# eigenvalues, eigvectors = la.eig(J_act-R_act, M_act)
# print(eigenvalues/(2*3.14))

eigenvalues, eigvectors = la.eig(J_sys, M_sys)
omega_all = np.imag(eigenvalues)

tol = 10 ** (-9)
index = omega_all >= tol

omega = omega_all[index]
eigvec_omega = eigvectors[:, index]
perm = np.argsort(omega)
eigvec_omega = eigvec_omega[:, perm]

omega.sort()

n_om = 6

for i in range(n_om):
    print('Omega full lambda ' + str(i+1) + ': ' + str(omega[i]/(2*pi)))


model_ode, T = model_all.dae_to_odeCE(mass=True)[:2]

print(model_ode.n)


M_full = model_ode.E
J_full = model_ode.J
B_full = model_ode.B

eigenvalues, eigvectors = la.eig(J_full, M_full)
omega_all = np.imag(eigenvalues)

tol = 10 ** (-9)
index = omega_all >= tol

omega = omega_all[index]
eigvec_omega = eigvectors[:, index]
perm = np.argsort(omega)
eigvec_omega = eigvec_omega[:, perm]

omega.sort()

n_om = 6

for i in range(n_om):
    print('Omega full ' + str(i+1) + ': ' + str(omega[i]/(2*pi)))

model_red, V_f = model_ode.reduce_system(1, 20)
print(model_red.n)

M_red = model_red.E
J_red = model_red.J
B_red = model_red.B

eigenvalues, eigvectors = la.eig(J_red, M_red)
omega_all = np.imag(eigenvalues)

tol = 10 ** (-9)
index = omega_all >= tol

omega = omega_all[index]
eigvec_omega = eigvectors[:, index]
perm = np.argsort(omega)
eigvec_omega = eigvec_omega[:, perm]

omega.sort()

n_om = 6

for i in range(n_om):
    print('Omega red ' + str(i + 1) + ': ' + str(omega[i] / (2 * pi)))

# pathout = '/home/a.brugnoli/GitProjects/MatlabProjects/PH/ReductionPHDAE/KP_Experiment/'
# # M_file = 'M'; Q_file = 'Q'; J_file = 'J'; B_file = 'B'
# # savemat(pathout + M_file, mdict={M_file: M_full})
# # savemat(pathout + J_file, mdict={J_file: J_full})
# # savemat(pathout + B_file, mdict={B_file: B_full})
#
# Mr_file = 'Mr'; Qr_file = 'Qr'; Jr_file = 'Jr'; Br_file = 'Br'
# savemat(pathout + Mr_file, mdict={Mr_file: M_red})
# savemat(pathout + Jr_file, mdict={Jr_file: J_red})
# savemat(pathout + Br_file, mdict={Br_file: B_red})

plot_eig = False

if plot_eig:

    n_fig = n_om
    fntsize = 16
    linewidth = 2

    for i in range(n_fig):
        eig_real_w = Function(Vp)
        eig_imag_w = Function(Vp)

        eig_real_plate = np.real(eigvec_omega[:np_plate, i])
        eig_imag_plate = np.imag(eigvec_omega[:np_plate, i])
        eig_real_w.vector()[:] = eig_real_plate
        eig_imag_w.vector()[:] = eig_imag_plate

        norm_real_eig = np.linalg.norm(eig_real_w.vector().get_local())
        norm_imag_eig = np.linalg.norm(eig_imag_w.vector().get_local())

        figure = plt.figure(i)
        ax = figure.add_subplot(111, projection="3d")

        ax.set_xlabel('$x [m]$', fontsize=fntsize)
        ax.set_xlim((-0.02, 0.06))
        ax.set_ylabel('$y [m]$', fontsize=fntsize)
        ax.set_ylim((0, 0.3))
        ax.set_title('$v_{e_{w}}$', fontsize=fntsize)

        ax.w_zaxis.set_major_locator(LinearLocator(10))
        ax.w_zaxis.set_major_formatter(FormatStrFormatter('%1.2g'))
        surf_opts = {'cmap': cm.jet}

        if norm_imag_eig > norm_real_eig:
            triangulation, Z = _two_dimension_triangle_func_val(eig_imag_w, 10)
            plot_pl = ax.plot_trisurf(triangulation, Z, label='Plate', cmap=cm.jet)
            # plot_pl = plot(eig_imag_w, axes=ax, plot3d=True, label='Plate', cmap=cm.jet)
            bol_imag = True
        else:
            triangulation, Z = _two_dimension_triangle_func_val(eig_real_w, 10)
            plot_pl = ax.plot_trisurf(triangulation, Z, label='Plate', cmap=cm.jet)
            # plot_pl = plot(eig_real_w, axes=ax, plot3d=True, label='Plate', cmap=cm.jet)
            bol_imag = False


        r_L = r_E + r_EL
        x_L, y_L, z_L = r_L
        # z_L=0

        ind_v_mir = np_plate

        eig_real_mir = np.real(eigvec_omega[ind_v_mir, i])
        eig_imag_mir = np.imag(eigvec_omega[ind_v_mir, i])

        norm_real_eig_mir = abs(eig_real_mir)
        norm_imag_eig_mir = abs(eig_imag_mir)

        # if norm_imag_eig_mir > norm_real_eig_mir:
        if bol_imag:
            eig_mir = eig_imag_mir
        else:
            eig_mir = eig_real_mir

        ax.plot([x_L, x_L], [y_L, y_L], [z_L, z_L + eig_mir], \
                linewidth=linewidth, linestyle='--', label='Mirror', color='blue')

        x_A2, y_A2, z_A2 = r_A2
        # z_A2 = 0
        ind_v_act2 = np_plate + np_mir + 2

        eig_real_act2 = np.real(eigvec_omega[ind_v_act2, i])
        eig_imag_act2 = np.imag(eigvec_omega[ind_v_act2, i])

        norm_real_eig_act2 = abs(eig_real_act2)
        norm_imag_eig_act2 = abs(eig_imag_act2)

        # if norm_imag_eig_act2 > norm_real_eig_act2:
        if bol_imag:
            eig_act2 = eig_imag_act2
        else:
            eig_act2 = eig_real_act2

        ax.plot([x_A2, x_A2], [y_A2, y_A2], [z_A2, z_A2+eig_act2], \
                linewidth=linewidth, linestyle='--', label='Act 2', color='black')

        x_A1, y_A1, z_A1 = r_A1
        # z_A1 = 0
        ind_v_act1 = np_plate + np_mir + np_act + 2


        eig_real_act1 = np.real(eigvec_omega[ind_v_act1, i])
        eig_imag_act1 = np.imag(eigvec_omega[ind_v_act1, i])

        norm_real_eig_act1 = abs(eig_real_act1)
        norm_imag_eig_act1 = abs(eig_imag_act1)

        # if norm_imag_eig_act1 > norm_real_eig_act1:
        if bol_imag:
            eig_act1 = eig_imag_act1
        else:
            eig_act1 = eig_real_act1

        ax.plot([x_A1, x_A1], [y_A1, y_A1], [z_A1, z_A1 + eig_act1], \
                linewidth=linewidth, linestyle='--', label='Act 1', color='black')

        off_mov = 0
        z_mov2 = z_A2 + off_mov
        ind_v_mov2 = np_plate + np_mir + 1

        eig_real_mov2 = np.real(eigvec_omega[ind_v_mov2, i])
        eig_imag_mov2 = np.imag(eigvec_omega[ind_v_mov2, i])

        norm_real_eig_mov2 = abs(eig_real_mov2)
        norm_imag_eig_mov2 = abs(eig_imag_mov2)

        # if norm_imag_eig_mov2 > norm_real_eig_mov2:
        if bol_imag:
            eig_mov2 = eig_imag_mov2
        else:
            eig_mov2 = eig_real_mov2

        ax.plot([x_A2, x_A2], [y_A2, y_A2], [z_mov2, z_mov2 + eig_mov2], \
                linewidth=linewidth, linestyle='-.', label='Mov 2', color='red')

        z_mov1 = z_A1 + off_mov
        ind_v_mov1 = np_plate + np_mir + np_act + 1

        eig_real_mov1 = np.real(eigvec_omega[ind_v_mov1, i])
        eig_imag_mov1 = np.imag(eigvec_omega[ind_v_mov1, i])

        norm_real_eig_mov1 = abs(eig_real_mov1)
        norm_imag_eig_mov1 = abs(eig_imag_mov1)

        # if norm_imag_eig_mov1 > norm_real_eig_mov1:
        if bol_imag:
            eig_mov1 = eig_imag_mov1
        else:
            eig_mov1 = eig_real_mov1

        ax.plot([x_A1, x_A1], [y_A1, y_A1], [z_mov1, z_mov1 + eig_mov1], \
                linewidth=linewidth, linestyle='-.', label='Mov 1', color='red')

        # print('Eig mir: ' + str(eig_mir))
        # print('Eig act2: ' + str(eig_act2))
        # print('Eig act1: ' + str(eig_act1))
        # print('Eig mov2: ' + str(eig_mov2))
        # print('Eig mov1: ' + str(eig_mov1))

        # ind_diffz2 = np_plate + np_mir
        #
        # eig_real_diffz2 = np.real(eigvec_omega[ind_diffz2, i])
        # eig_imag_diffz2 = np.imag(eigvec_omega[ind_diffz2, i])
        #
        # norm_real_eig_diffz2 = abs(eig_real_diffz2)
        # norm_imag_eig_diffz2 = abs(eig_imag_diffz2)
        #
        # # if norm_imag_eig_diffz2 > norm_real_eig_diffz2:
        # if bol_imag:
        #     eig_diffz2 = eig_imag_diffz2
        # else:
        #     eig_diffz2 = eig_real_diffz2
        #
        # print('Eig diffz2: ' + str(eig_diffz2))
        #
        # ax.plot([x_A2, x_A2], [y_A2, y_A2], [z_A2, z_A2 + eig_diffz2], \
        #         linewidth=linewidth, linestyle='-', label='Diffz 2', color='green')
        #
        # ind_diffz1 = np_plate + np_mir + np_act2
        #
        # eig_real_diffz1 = np.real(eigvec_omega[ind_diffz1, i])
        # eig_imag_diffz1 = np.imag(eigvec_omega[ind_diffz1, i])
        #
        # norm_real_eig_diffz1 = abs(eig_real_diffz1)
        # norm_imag_eig_diffz1 = abs(eig_imag_diffz1)
        #
        # # if norm_imag_eig_diffz1 > norm_real_eig_diffz1:
        # if bol_imag:
        #     eig_diffz1 = eig_imag_diffz1
        # else:
        #     eig_diffz1 = eig_real_diffz1
        #
        # print('Eig diffz1: ' + str(eig_mov1))
        #
        # ax.plot([x_A1, x_A1], [y_A1, y_A1], [z_A1, z_A1 + eig_diffz1], \
        #         linewidth=linewidth, linestyle='-', label='Diffz 1', color='green')

        plot_pl._facecolors2d = plot_pl._facecolors3d
        plot_pl._edgecolors2d = plot_pl._edgecolors3d

        plt.legend()

plt.show()

