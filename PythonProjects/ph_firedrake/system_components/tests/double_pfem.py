from firedrake import *
import numpy as np
import scipy.linalg as la
from modules_phdae.classes_phsystem import SysPhdaeRig, check_positive_matrix
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


def create_sys1(mesh1, deg_p1, deg_q1):
    x1, y1 = SpatialCoordinate(mesh1)

    tab_coord1 = mesh1.coordinates.dat.data
    x1_cor = tab_coord1[:, 0]
    y1_cor = tab_coord1[:, 1]

    Vp1 = FunctionSpace(mesh1, "CG", deg_p1)
    Vq1 = FunctionSpace(mesh1, "RT", deg_q1)

    V1 = Vp1 * Vq1

    n_1 = V1.dim()
    np_1 = Vp1.dim()
    nq_1 = Vq1.dim()

    v1 = TestFunction(V1)
    v_p1, v_q1 = split(v1)

    e1 = TrialFunction(V1)
    e_p1, e_q1 = split(e1)

    al_p1 = rho * e_p1
    al_q1 = 1./T * e_q1

    dx1 = Measure('dx', domain=mesh1)
    ds1 = Measure('ds', domain=mesh1)

    m_p1 = v_p1 * al_p1 * dx1
    m_q1 = dot(v_q1, al_q1) * dx1

    m_form1 = m_p1 + m_q1

    j_div = dot(v_p1, div(e_q1)) * dx1
    j_divIP = -dot(div(v_q1), e_p1) * dx1

    j_form1 = j_div + j_divIP
    petsc_j1 = assemble(j_form1, mat_type='aij').M.handle
    petsc_m1 = assemble(m_form1, mat_type='aij').M.handle

    J1 = np.array(petsc_j1.convert("dense").getDenseArray())
    M1 = np.array(petsc_m1.convert("dense").getDenseArray())

    Vf1 = FunctionSpace(mesh1, "CG", 1)
    f_D = TrialFunction(Vf1)
    v_D = TestFunction(Vf1)

    m_delta = v_D * f_D * ds1

    petsc_md = assemble(m_delta, mat_type='aij').M.handle
    Mdelta = np.array(petsc_md.convert("dense").getDenseArray())

    n = FacetNormal(mesh1)
    tolx1 = 1e-15
    is_int1 = conditional(gt(x1, tolx1), 1, 0)
    b_D = dot(v_q1, n) * f_D * is_int1 * ds1

    petsc_bD = assemble(b_D, mat_type='aij').M.handle
    B_D1 = np.array(petsc_bD.convert("dense").getDenseArray())

    perm_y1 = np.argsort(y1_cor)
    B_D1 = B_D1[:, perm_y1]

    Mdelta = Mdelta[:, perm_y1]
    Mdelta = Mdelta[perm_y1, :]

    int_dofs1 = np.where(B_D1.any(axis=0))[0]

    Mdelta = Mdelta[:, int_dofs1]
    Mdelta = Mdelta[int_dofs1, :]

    B_D1 = B_D1[:, int_dofs1]

    # y1_cor.sort()
    # x1_cor = x1_cor[perm_y1]
    #
    # plt.plot(x1_cor[int_dofs1], y1_cor[int_dofs1], 'r*'); plt.show()
    # print(len(int_dofs1))
    # print(y1_cor[int_dofs1])

    sys1 = SysPhdaeRig(n_1, 0, 0, np_1, nq_1, E=M1, J=J1, B=B_D1)

    return sys1, Mdelta, Vp1


def create_sys2(mesh2, deg_p2, deg_q2):
    x2, y2 = SpatialCoordinate(mesh2)

    tab_coord2 = mesh2.coordinates.dat.data
    x2_cor = tab_coord2[:, 0]
    y2_cor = tab_coord2[:, 1]

    Vp2 = FunctionSpace(mesh2, "CG", deg_p2)
    Vq2 = FunctionSpace(mesh2, "RT", deg_q2)

    V2 = Vp2 * Vq2

    n_2 = V2.dim()
    np_2 = Vp2.dim()
    nq_2 = Vq2.dim()

    v2 = TestFunction(V2)
    v_p2, v_q2 = split(v2)

    e2 = TrialFunction(V2)
    e_p2, e_q2 = split(e2)

    al_p2 = rho * e_p2
    al_q2 = 1. / T * e_q2

    dx2 = Measure('dx', domain=mesh2)
    ds2 = Measure('ds', domain=mesh2)

    m_p2 = v_p2 * al_p2 * dx2
    m_q2 = dot(v_q2, al_q2) * dx2
    m_form2 = m_p2 + m_q2

    j_grad = dot(v_q2, grad(e_p2)) * dx2
    j_gradIP = -dot(grad(v_p2), e_q2) * dx2
    j_form2 = j_grad + j_gradIP

    petsc_j2 = assemble(j_form2, mat_type='aij').M.handle
    petsc_m2 = assemble(m_form2, mat_type='aij').M.handle
    J2 = np.array(petsc_j2.convert("dense").getDenseArray())
    M2 = np.array(petsc_m2.convert("dense").getDenseArray())

    Vf2 = FunctionSpace(mesh2, "CG", 1)
    f_N = TrialFunction(Vf2)

    tolx1 = 1e-15
    tolx2 = 0.25

    toly1 = tolx1
    toly2 = 1 - toly1

    xis_int2 = conditional(And(gt(x2, tolx1), lt(x2, tolx2)), 1, 0)
    yis_int2 = conditional(And(gt(y2, toly1), lt(y2, toly2)), 1, 0)

    b_N = v_p2 * f_N * xis_int2 * yis_int2 * ds2

    petsc_bN = assemble(b_N, mat_type='aij').M.handle
    B_N2 = np.array(petsc_bN.convert("dense").getDenseArray())

    perm_y2 = np.argsort(y2_cor)
    B_N2 = B_N2[:, perm_y2]

    int_dofs2 = np.where(B_N2.any(axis=0))[0]

    B_N2 = B_N2[:, int_dofs2]

    # y2_cor.sort()
    # x2_cor = x2_cor[perm_y2]
    #
    # plt.plot(x2_cor[int_dofs2], y2_cor[int_dofs2], 'b*')
    # plt.show()
    # print(len(int_dofs2))
    # print(y2_cor[int_dofs2])

    sys2 = SysPhdaeRig(n_2, 0, 0, np_2, nq_2, E=M2, J=J2, B=B_N2)

    return sys2, Vp2


def print_modes(sys1, sys2, Mdelta, Vp1, Vp2, n_modes):

    m1 = sys1.m
    m2 = sys2.m

    sys_CF = SysPhdaeRig.gyrator_ordered(sys1, sys2, list(range(m1)), list(range(m2)), la.inv(Mdelta))

    eigenvalues, eigvectors = la.eig(sys_CF.J_f, sys_CF.M_f)
    omega_all = np.imag(eigenvalues)

    index = omega_all > 0

    omega = omega_all[index]
    eigvec_omega = eigvectors[:, index]
    perm = np.argsort(omega)
    eigvec_omega = eigvec_omega[:, perm]

    omega.sort()

    fntsize = 15

    n_Vp1 = Vp1.dim()
    n_Vp2 = Vp2.dim()

    for i in range(int(n_modes)):
        print("Eigenvalue num " + str(i+1) + ":" + str(omega[i]))
        eig_real_p1 = Function(Vp1)
        eig_imag_p1 = Function(Vp1)

        eig_real_p1.vector()[:] = np.real(eigvec_omega[:n_Vp1, i])
        eig_imag_p1.vector()[:] = np.imag(eigvec_omega[:n_Vp1, i])

        eig_real_p2 = Function(Vp2)
        eig_imag_p2 = Function(Vp2)

        eig_real_p2.vector()[:] = np.real(eigvec_omega[n_Vp1:n_Vp1 + n_Vp2, i])
        eig_imag_p2.vector()[:] = np.imag(eigvec_omega[n_Vp1:n_Vp1 + n_Vp2, i])

        norm_real_eig = np.linalg.norm(np.real(eigvec_omega[:n_Vp1+n_Vp2, i]))
        norm_imag_eig = np.linalg.norm(np.imag(eigvec_omega[:n_Vp1+n_Vp2, i]))

        figure = plt.figure()
        ax = figure.add_subplot(111, projection="3d")

        if norm_imag_eig > norm_real_eig:
            # plot(eig_imag_p1, axes=ax, plot3d=True)
            # plot(eig_imag_p2, axes=ax, plot3d=True)
            triangulation1, z1_goodeig = _two_dimension_triangle_func_val(eig_imag_p1, 10)
            triangulation2, z2_goodeig = _two_dimension_triangle_func_val(eig_imag_p2, 10)
        else:
            # plot(eig_real_p1, axes=ax, plot3d=True)
            # plot(eig_real_p2, axes=ax, plot3d=True)
            triangulation1, z1_goodeig = _two_dimension_triangle_func_val(eig_real_p1, 10)
            triangulation2, z2_goodeig = _two_dimension_triangle_func_val(eig_real_p2, 10)

        ax.set_xlabel('$x [m]$', fontsize=fntsize)
        ax.set_ylabel('$y [m]$', fontsize=fntsize)
        ax.set_title('Eigenvector num ' + str(i + 1), fontsize=fntsize)

        ax.w_zaxis.set_major_locator(LinearLocator(10))
        ax.w_zaxis.set_major_formatter(FormatStrFormatter('%1.2g'))

        plot_eig1 = ax.plot_trisurf(triangulation1, z1_goodeig, cmap=cm.jet)
        plot_eig1._facecolors2d = plot_eig1._facecolors3d
        plot_eig1._edgecolors2d = plot_eig1._edgecolors3d

        plot_eig2 = ax.plot_trisurf(triangulation2, z2_goodeig, cmap=cm.jet)
        plot_eig2._facecolors2d = plot_eig2._facecolors3d
        plot_eig2._edgecolors2d = plot_eig2._edgecolors3d

        ax.legend(("mesh 1", "mesh2"))

        path_figs = "/home/a.brugnoli/Plots_Videos/Python/Plots/Waves/"
        # plt.savefig(path_figs + "Eig_n" + str(i) + ".eps")

    plt.show()


rho = 1
T = 1

degp = 1
degq = 2

path_mesh = "/home/a.brugnoli/GitProjects/PythonProjects/ph_firedrake/system_components/tests/meshes/"
mesh1 = Mesh(path_mesh + "dom1.msh")
mesh2 = Mesh(path_mesh + "dom2.msh")

plot(mesh1)
plot(mesh2)
plt.show()

sys1, Mdelta, Vp1 = create_sys1(mesh1, degp, degq)
sys2, Vp2 = create_sys2(mesh2, degp, degq)

print_modes(sys1, sys2, Mdelta, Vp1, Vp2, 10)

