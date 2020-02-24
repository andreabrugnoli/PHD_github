from firedrake import *
import numpy as np
import scipy.linalg as la
from modules_ph.classes_phsystem import SysPhdae, check_positive_matrix
import warnings
np.set_printoptions(threshold=np.inf)
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from firedrake.plot import _two_dimension_triangle_func_val
from mpl_toolkits.mplot3d import Axes3D
from math import pi
plt.rc('text', usetex=True)

# wave = NeumannWave(1, 1, 1, 1, 10, 10, modes=True)
# wave = DirichletWave(1, 1, 1, 1, 10, 10, modes=True)
tol = 1e-2

L = 1
# E = Constant(1)
# nu = Constant(0.3)
#
# rho = Constant(1)
# k = Constant(5 / 6)
# h = Constant(0.1)
E = 1
nu = 0.3

rho = 1
k = 0.8601
h = 0.1

D = E * h ** 3 / (1 - nu ** 2) / 12
fl_rot = 12 / (E * h ** 3)

G = E / 2 / (1 + nu)
F = G * h * k

def gradSym(u):
    return 0.5 * (nabla_grad(u) + nabla_grad(u).T)
    # return sym(nabla_grad(u))


def bending_mom(kappa):
    momenta = D * ((1 - nu) * kappa + nu * Identity(2) * tr(kappa))
    return momenta


def bending_curv(momenta):
    kappa = fl_rot * ((1 + nu) * momenta - nu * Identity(2) * tr(momenta))
    return kappa


def create_sys1(mesh1, r):
    x1, y1 = SpatialCoordinate(mesh1)

    tab_coord1 = mesh1.coordinates.dat.data
    x1_cor = tab_coord1[:, 0]
    y1_cor = tab_coord1[:, 1]

    # Operators and functions

    def m_operator1(v_pw, al_pw, v_pth, al_pth, v_qth, al_qth, e_qth, v_qw, al_qw, v_skw, e_skw):
        m_form = v_pw * al_pw * dx1 \
                 + dot(v_pth, al_pth) * dx1 \
                 + inner(v_qth, al_qth) * dx1 + inner(v_qth, e_skw) * dx1 \
                 + dot(v_qw, al_qw) * dx1 \
                 + inner(v_skw, e_qth) * dx1

        return m_form

    def j_operator1(v_pw, e_pw, v_pth, e_pth, v_qth, e_qth, v_qw, e_qw):
        j_div = v_pw * div(e_qw) * dx1
        j_divIP = -div(v_qw) * e_pw * dx1

        j_divSym = dot(v_pth, div(e_qth)) * dx1
        j_divSymIP = -dot(div(v_qth), e_pth) * dx1

        j_Id = dot(v_pth, e_qw) * dx1
        j_IdIP = -dot(v_qw, e_pth) * dx1

        j_form = j_div + j_divIP + j_divSym + j_divSymIP + j_Id + j_IdIP

        return j_form

    # The unit square mesh is divided in :math:`N\times N` quadrilaterals::


    # plot(mesh);
    # plt.show()

    # Finite element defition

    Vpw_1 = FunctionSpace(mesh1, "DG", r1-1)
    Vpth_1 = VectorFunctionSpace(mesh1, "DG", r1 - 1)
    Vskw_1 = FunctionSpace(mesh1, "DG", r1 - 1)

    Vqth1_1 = FunctionSpace(mesh1, "BDM", r1)
    Vqth2_1 = FunctionSpace(mesh1, "BDM", r1)
    # Vqth1_1 = FunctionSpace(mesh1, "RT", r)
    # Vqth2_1 = FunctionSpace(mesh1, "RT", r)
    Vqw_1 = FunctionSpace(mesh1, "RT", r1)

    V1 = MixedFunctionSpace([Vpw_1, Vpth_1, Vqth1_1, Vqth2_1, Vqw_1, Vskw_1])
    n_1 = V1.dim()
    print(n_1)

    np_1 = Vpw_1.dim() + Vpth_1.dim()
    nq_1 = Vqth1_1.dim() + Vqth2_1.dim() + Vqw_1.dim()

    v1 = TestFunction(V1)
    vpw_1, vpth_1, vqth1_1, vqth2_1, vqw_1, vskw_1 = split(v1)

    e1 = TrialFunction(V1)
    epw_1, epth_1, eqth1_1, eqth2_1, eqw_1, eskw_1 = split(e1)

    vqth_1 = as_tensor([[vqth1_1[0], vqth1_1[1]],
                       [vqth2_1[0], vqth2_1[1]]
                       ])

    eqth_1 = as_tensor([[eqth1_1[0], eqth1_1[1]],
                       [eqth2_1[0], eqth2_1[1]]
                       ])

    alpw_1 = rho * h * epw_1
    alpth_1 = (rho * h ** 3) / 12. * epth_1
    alqth_1 = bending_curv(eqth_1)
    alqw_1 = 1. / F * eqw_1

    vskw_1 = as_tensor([[0, vskw_1],
                       [-vskw_1, 0]])
    eskw_1 = as_tensor([[0, eskw_1],
                       [-eskw_1, 0]])

    dx1 = Measure('dx', domain=mesh1)
    ds1 = Measure('ds', domain=mesh1)

    j_form1 = j_operator1(vpw_1, epw_1, vpth_1, epth_1, vqth_1, eqth_1, vqw_1, eqw_1)
    m_form1 = m_operator1(vpw_1, alpw_1, vpth_1, alpth_1, vqth_1, alqth_1, eqth_1, vqw_1, alqw_1, vskw_1, eskw_1)

    petsc_j1 = assemble(j_form1, mat_type='aij').M.handle
    petsc_m1 = assemble(m_form1, mat_type='aij').M.handle

    J1 = np.array(petsc_j1.convert("dense").getDenseArray())
    M1 = np.array(petsc_m1.convert("dense").getDenseArray())

    n_ver1 = FacetNormal(mesh1)
    s_ver1 = as_vector([-n_ver1[1], n_ver1[0]])

    vqn_1 = dot(vqw_1, n_ver1)
    vMnn_1 = inner(vqth_1, outer(n_ver1, n_ver1))
    vMns_1 = inner(vqth_1, outer(n_ver1, s_ver1))

    is_int1 = conditional(And(gt(x1, L/2 - tol), And(lt(y1, x1+tol), gt(y1, L - x1-tol))), 1, 0)

    Vu = FunctionSpace(mesh1, "CG", 1)
    n_u = Vu.dim()

    f_D = TrialFunction(Vu)
    v_D = TestFunction(Vu)

    m_delta = v_D * f_D * is_int1 * ds1

    petsc_md = assemble(m_delta, mat_type='aij').M.handle
    Mdelta = np.array(petsc_md.convert("dense").getDenseArray())

    wt_1 = TrialFunction(Vu)
    omn_1 = TrialFunction(Vu)
    oms_1 = TrialFunction(Vu)


    b_D1 = vqn_1 * wt_1 * is_int1 * ds1
    b_D2 = vMnn_1 * omn_1 * is_int1 * ds1
    b_D3 = vMns_1 * oms_1 * is_int1 * ds1

    petsc_bD1 = assemble(b_D1, mat_type='aij').M.handle
    B_D1 = np.array(petsc_bD1.convert("dense").getDenseArray())

    petsc_bD2 = assemble(b_D2, mat_type='aij').M.handle
    B_D2 = np.array(petsc_bD2.convert("dense").getDenseArray())

    petsc_bD3 = assemble(b_D3, mat_type='aij').M.handle
    B_D3 = np.array(petsc_bD3.convert("dense").getDenseArray())

    perm_y1 = np.argsort(y1_cor)

    B_D1 = B_D1[:, perm_y1]
    B_D2 = B_D2[:, perm_y1]
    B_D3 = B_D3[:, perm_y1]

    Mdelta = Mdelta[:, perm_y1]
    Mdelta = Mdelta[perm_y1, :]

    int_dofs1_1 = np.where(B_D1.any(axis=0))[0]
    int_dofs1_2 = np.where(B_D2.any(axis=0))[0]
    int_dofs1_3 = np.where(B_D3.any(axis=0))[0]

    Mdelta = Mdelta[:, int_dofs1_1]
    Mdelta = Mdelta[int_dofs1_1, :]

    B_D1 = B_D1[:, int_dofs1_1]
    B_D2 = B_D2[:, int_dofs1_2]
    B_D3 = B_D3[:, int_dofs1_3]

    B_D = np.concatenate((B_D1, B_D2, B_D3), axis=1)

    y1_cor.sort()
    x1_cor = x1_cor[perm_y1]

    plt.plot(x1_cor[int_dofs1_1], y1_cor[int_dofs1_1], 'r*')
    # print(len(int_dofs1_1))
    # print(y1_cor[int_dofs1_1])

    sys1 = SysPhdae(n_1, 0, np_1, nq_1, E=M1, J=J1, B=B_D)

    Mdelta = la.block_diag(Mdelta, Mdelta, Mdelta)

    return sys1, Mdelta, Vpw_1


def create_sys2(mesh2, r2):

    x2, y2 = SpatialCoordinate(mesh2)

    tab_coord2 = mesh2.coordinates.dat.data
    x2_cor = tab_coord2[:, 0]
    y2_cor = tab_coord2[:, 1]

    # Operators and functions

    def m_operator2(v_pw, al_pw, v_pth, al_pth, v_qth, al_qth, v_qw, al_qw):
        m_form = v_pw * al_pw * dx2 \
                 + dot(v_pth, al_pth) * dx2 \
                 + inner(v_qth, al_qth) * dx2 \
                 + dot(v_qw, al_qw) * dx2

        return m_form

    def j_operator2(v_pw, e_pw, v_pth, e_pth, v_qth, e_qth, v_qw, e_qw):
        j_grad = dot(v_qw, grad(e_pw)) * dx2
        j_gradIP = -dot(grad(v_pw), e_qw) * dx2

        j_gradSym = inner(v_qth, gradSym(e_pth)) * dx2
        j_gradSymIP = -inner(gradSym(v_pth), e_qth) * dx2

        j_Id = dot(v_pth, e_qw) * dx2
        j_IdIP = -dot(v_qw, e_pth) * dx2

        j_allgrad = j_grad + j_gradIP + j_gradSym + j_gradSymIP + j_Id + j_IdIP

        j_form = j_allgrad

        return j_form

    Vpw_2 = FunctionSpace(mesh2, "CG", r2)
    Vpth_2 = VectorFunctionSpace(mesh2, "CG", r2)
    VqthD_2 = VectorFunctionSpace(mesh2, "DG", r2 - 1)
    Vqth12_2 = FunctionSpace(mesh2, "DG", r2 - 1)
    Vqw_2 = VectorFunctionSpace(mesh2, "DG", r2 - 1)

    V2 = MixedFunctionSpace([Vpw_2, Vpth_2, VqthD_2, Vqth12_2, Vqw_2])

    n_2 = V2.dim()
    np_2 = Vpw_2.dim() + Vpth_2.dim()
    nq_2 = Vqth12_2.dim() + VqthD_2.dim() + Vqw_2.dim()
    print(n_2)

    v2 = TestFunction(V2)
    vpw_2, vpth_2, vqthD_2, vqth12_2, vqw_2 = split(v2)

    e2 = TrialFunction(V2)
    epw_2, epth_2, eqthD_2, eqth12_2, eqw_2 = split(e2)

    vqth_2 = as_tensor([[vqthD_2[0], vqth12_2],
                       [vqth12_2, vqthD_2[1]]
                       ])

    eqth_2 = as_tensor([[eqthD_2[0], eqth12_2],
                       [eqth12_2, eqthD_2[1]]
                       ])

    alpw_2 = rho * h * epw_2
    alpth_2 = (rho * h ** 3) / 12. * epth_2
    alqth_2 = bending_curv(eqth_2)
    alqw_2 = 1. / F * eqw_2

    dx2 = Measure('dx', domain=mesh2)
    ds2 = Measure('ds', domain=mesh2)

    j_form2 = j_operator2(vpw_2, epw_2, vpth_2, epth_2, vqth_2, eqth_2, vqw_2, eqw_2)
    m_form2 = m_operator2(vpw_2, alpw_2, vpth_2, alpth_2, vqth_2, alqth_2, vqw_2, alqw_2)

    petsc_j2 = assemble(j_form2, mat_type='aij').M.handle
    petsc_m2 = assemble(m_form2, mat_type='aij').M.handle
    J2 = np.array(petsc_j2.convert("dense").getDenseArray())
    M2 = np.array(petsc_m2.convert("dense").getDenseArray())

    n_ver2 = FacetNormal(mesh2)
    s_ver2 = as_vector([-n_ver2[1], n_ver2[0]])

    Vu = FunctionSpace(mesh2, 'CG', 1)
    q_n = TrialFunction(Vu)
    M_nn = TrialFunction(Vu)
    M_ns = TrialFunction(Vu)

    v_omn = dot(vpth_2, n_ver2)
    v_oms = dot(vpth_2, s_ver2)

    is_int2 = conditional(Or(gt(y2, x2 - tol), lt(y2, L - x2 + tol)), 1, 0)

    b_N1 = vpw_2 * q_n * is_int2 * ds2
    b_N2 = v_omn * M_nn * is_int2 * ds2
    b_N3 = v_oms * M_ns * is_int2 * ds2

    petsc_bN1 = assemble(b_N1, mat_type='aij').M.handle
    B_N1 = np.array(petsc_bN1.convert("dense").getDenseArray())

    petsc_bN2 = assemble(b_N2, mat_type='aij').M.handle
    B_N2 = np.array(petsc_bN2.convert("dense").getDenseArray())

    petsc_bN3 = assemble(b_N3, mat_type='aij').M.handle
    B_N3 = np.array(petsc_bN3.convert("dense").getDenseArray())

    perm_y2 = np.argsort(y2_cor)

    B_N1 = B_N1[:, perm_y2]
    B_N2 = B_N2[:, perm_y2]
    B_N3 = B_N3[:, perm_y2]

    int_dofs2_1 = np.where(B_N1.any(axis=0))[0]
    int_dofs2_2 = np.where(B_N2.any(axis=0))[0]
    int_dofs2_3 = np.where(B_N3.any(axis=0))[0]

    B_N1 = B_N1[:, int_dofs2_1]
    B_N2 = B_N2[:, int_dofs2_2]
    B_N3 = B_N3[:, int_dofs2_3]

    B_N = np.concatenate((B_N1, B_N2, B_N3), axis=1)

    y2_cor.sort()
    x2_cor = x2_cor[perm_y2]

    plt.plot(x2_cor[int_dofs2_1], y2_cor[int_dofs2_1], 'go')

    sys2 = SysPhdae(n_2, 0, np_2, nq_2, E=M2, J=J2, B=B_N)

    return sys2, Vpw_2


def print_modes(sys1, sys2, Mdelta, Vp1, Vp2, n_modes):

    m1 = sys1.m
    m2 = sys2.m

    sys_CF = SysPhdae.gyrator(sys1, sys2, list(range(m1)), list(range(m2)), la.inv(Mdelta))

    eigenvalues, eigvectors = la.eig(sys_CF.J, sys_CF.E)
    omega_all = np.imag(eigenvalues)

    index = omega_all > 0

    omega = omega_all[index]
    eigvec_omega = eigvectors[:, index]
    perm = np.argsort(omega)
    eigvec_omega = eigvec_omega[:, perm]

    omega.sort()

    omega_tilde = omega*L*((2*(1+nu)*rho)/E)**0.5

    fntsize = 15

    n_Vp1 = Vp1.dim()
    n_Vp2 = Vp2.dim()
    n1 = sys1.n

    for i in range(int(n_modes)):
        print("Eigenvalue num " + str(i+1) + ":" + str(omega_tilde[i]))
        eig_real_p1 = Function(Vp1)
        eig_imag_p1 = Function(Vp1)

        eig_real_p1.vector()[:] = np.real(eigvec_omega[:n_Vp1, i])
        eig_imag_p1.vector()[:] = np.imag(eigvec_omega[:n_Vp1, i])

        eig_real_p2 = Function(Vp2)
        eig_imag_p2 = Function(Vp2)

        eig_real_p2.vector()[:] = np.real(eigvec_omega[n1:n1 + n_Vp2, i])
        eig_imag_p2.vector()[:] = np.imag(eigvec_omega[n1:n1 + n_Vp2, i])

        p_index = list(set.union(set(range(n_Vp1)), set(range(n1,n1+n_Vp2))))

        norm_real_eig = np.linalg.norm(np.real(eigvec_omega[p_index, i]))
        norm_imag_eig = np.linalg.norm(np.imag(eigvec_omega[p_index, i]))

        figure = plt.figure()
        ax = figure.add_subplot(111, projection="3d")

        if norm_imag_eig > norm_real_eig:
            plot(eig_imag_p1, axes=ax, plot3d=True)
            plot(eig_imag_p2, axes=ax, plot3d=True)
            # triangulation1, z1_goodeig = _two_dimension_triangle_func_val(eig_imag_p1, 10)
            # triangulation2, z2_goodeig = _two_dimension_triangle_func_val(eig_imag_p2, 10)
        else:
            plot(eig_real_p1, axes=ax, plot3d=True)
            plot(eig_real_p2, axes=ax, plot3d=True)
            # triangulation1, z1_goodeig = _two_dimension_triangle_func_val(eig_real_p1, 10)
            # triangulation2, z2_goodeig = _two_dimension_triangle_func_val(eig_real_p2, 10)

        ax.set_xlabel('$x [m]$', fontsize=fntsize)
        ax.set_ylabel('$y [m]$', fontsize=fntsize)
        ax.set_title('Eigenvector num ' + str(i + 1), fontsize=fntsize)

        ax.w_zaxis.set_major_locator(LinearLocator(10))
        ax.w_zaxis.set_major_formatter(FormatStrFormatter('%1.2g'))

        # plot_eig1 = ax.plot_trisurf(triangulation1, z1_goodeig, cmap=cm.jet)
        # plot_eig1._facecolors2d = plot_eig1._facecolors3d
        # plot_eig1._edgecolors2d = plot_eig1._edgecolors3d
        #
        # plot_eig2 = ax.plot_trisurf(triangulation2, z2_goodeig, cmap=cm.jet)
        # plot_eig2._facecolors2d = plot_eig2._facecolors3d
        # plot_eig2._edgecolors2d = plot_eig2._edgecolors3d
        #
        # ax.legend(("mesh 1", "mesh2"))

        # path_figs = "/home/a.brugnoli/Plots_Videos/Python/Plots/Waves/"
        # plt.savefig(path_figs + "Eig_n" + str(i) + ".eps")


r1 = 1
r2 = 1

path_mesh = "/home/a.brugnoli/GitProjects/PythonProjects/ph_firedrake/thick_plate/mixed_bc/"
mesh1 = Mesh(path_mesh + "dom_1.msh")
mesh2 = Mesh(path_mesh + "dom_2.msh")
# mesh1 = Mesh(path_mesh + "circle1.msh")
# mesh2 = Mesh(path_mesh + "circle2.msh")

figure = plt.figure()
ax = figure.add_subplot(111)
plot(mesh1, axes=ax)
plot(mesh2, axes=ax)
plt.xlim((-0.1, L+0.1))
plt.ylim((-0.1, L+0.1))


sys1, Mdelta, Vp1 = create_sys1(mesh1, r1)
sys2, Vp2 = create_sys2(mesh2, r2)

print_modes(sys1, sys2, Mdelta, Vp1, Vp2, 3)

plt.show()
