# EB beam written with the port Hamiltonian approach

from firedrake import *
import numpy as np
import scipy.linalg as la
from modules_phdae.classes_phsystem import SysPhdaeRig
from firedrake.plot import _two_dimension_triangle_func_val
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

matplotlib.rcParams['text.usetex'] = True


class FloatingPlanarEB(SysPhdaeRig):

    def __init__(self, n_el, rho, EI, L, m_joint=0.0, J_joint=0.0):

        mesh = IntervalMesh(n_el, L)
        x = SpatialCoordinate(mesh)

        # Finite element defition
        deg = 3
        Vp = FunctionSpace(mesh, "Hermite", deg)
        Vq = FunctionSpace(mesh, "Hermite", deg)

        V = Vp * Vq
        n_Vp = Vp.dim()
        n_Vq = Vq.dim()
        n_V = V.dim()

        v = TestFunction(V)
        v_p, v_q = split(v)

        e = TrialFunction(V)
        e_p, e_q = split(e)

        al_p = rho * e_p
        al_q = 1./EI * e_q

        dx = Measure('dx')
        ds = Measure('ds')

        m_form = v_p * al_p * dx + v_q * al_q * dx

        petsc_m = assemble(m_form, mat_type='aij').M.handle
        M_FEM = np.array(petsc_m.convert("dense").getDenseArray())

        n_rig = 3  # Planar motion
        n_fl = n_V - 2
        n_tot = n_rig + n_fl
        M_f = M_FEM[-n_fl:, -n_fl:]

        M_fr = np.zeros((n_fl, n_rig))
        M_fr[:, 1] = assemble(v_p * rho * dx).vector().get_local()[2:]
        M_fr[:, 2] = assemble(v_p * rho * x[0] * dx).vector().get_local()[2:]

        M_r = np.zeros((n_rig, n_rig))
        m_beam = rho * L
        J_beam = 1/3 * m_beam * L**2
        M_r[0][0] = m_beam + m_joint
        M_r[1][1] = m_beam + m_joint
        M_r[2][2] = J_beam + J_joint
        M_r[1][2] = m_beam * L/2
        M_r[2][1] = m_beam * L/2

        M = np.zeros((n_tot, n_tot))
        M[:n_rig, :n_rig] = M_r
        M[n_rig:, :n_rig] = M_fr
        M[:n_rig, n_rig:] = M_fr.T
        M[n_rig:, n_rig:] = M_f

        Q = la.inv(M)

        # isnot_P = conditional(ne(x[0], 0.0), 1., 0.)

        # j_divDiv = -v_p * e_q.dx(0).dx(0) * dx
        # j_divDivIP = v_q.dx(0).dx(0) * e_p * dx

        j_gradgrad = v_q * e_p.dx(0).dx(0) * dx
        j_gradgradIP = -v_p.dx(0).dx(0) * e_q * dx

        j_form = j_gradgrad + j_gradgradIP

        petcs_j = assemble(j_form, mat_type='aij').M.handle
        J_FEM = np.array(petcs_j.convert("dense").getDenseArray())

        J = np.zeros((n_tot, n_tot))
        J[n_rig:, n_rig:] = J_FEM[-n_fl:, -n_fl:]

        tau_CP = np.array([[1, 0, 0], [0, 1, L], [0, 0, 1]])
        b_F = v_p * ds(2)
        b_M = v_p.dx(0) * ds(2)

        B_Ffl = assemble(b_F).vector().get_local()
        B_Mfl = assemble(b_M).vector().get_local()

        B = np.zeros((n_tot, 6))
        B[1:, 4] = B_Ffl
        B[1:, 5] = B_Mfl
        B[:n_rig, :n_rig] = np.eye(n_rig)
        B[:n_rig, n_rig:] = tau_CP.T

        n_p = n_Vp - 2
        n_q = n_Vq

        SysPhdaeRig.__init__(self, n_tot, 0, n_rig, n_p, n_q, E=M, J=J, B=B)


class FloatingKP(SysPhdaeRig):

    def __init__(self, Lx, Ly, h, rho, E, nu, nx, ny, coord_P, coord_C=np.empty((0, 2)), modes=False):

        assert len(coord_P) == 2
        assert coord_C.shape[1] == 2
        fl_rot = 12. / (E * h ** 3)

        C_b_vec = as_tensor([
            [fl_rot, -nu * fl_rot, 0],
            [-nu * fl_rot, fl_rot, 0],
            [0, 0, fl_rot * 2 * (1 + nu)]
        ])


        # Vectorial Formulation possible only
        def bending_curv_vec(MM):
            return dot(C_b_vec, MM)

        def tensor_divDiv_vec(MM):
            return MM[0].dx(0).dx(0) + MM[1].dx(1).dx(1) + 2 * MM[2].dx(0).dx(1)

        def Gradgrad_vec(u):
            return as_vector([ u.dx(0).dx(0), u.dx(1).dx(1), 2 * u.dx(0).dx(1) ])

        mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=False)
        x, y = SpatialCoordinate(mesh)

        name_FEp = 'Bell'
        name_FEq = 'Bell'
        deg_p = 5
        deg_q = 5



        Vp = FunctionSpace(mesh, name_FEp, deg_p)
        Vq = VectorFunctionSpace(mesh, name_FEq, deg_q, dim=3)

        V = Vp * Vq
        n_Vp = Vp.dim()
        n_Vq = Vq.dim()
        n_V = n_Vp + n_Vq


        v = TestFunction(V)
        v_p, v_q = split(v)

        e_v = TrialFunction(V)
        e_p, e_q = split(e_v)

        al_p = rho * h * e_p
        al_q = bending_curv_vec(e_q)

        dx = Measure('dx')
        ds = Measure('ds')
        m_p = dot(v_p, al_p) * dx
        m_q = inner(v_q, al_q) * dx
        m_form = m_p + m_q

        j_gradgrad = inner(v_q, Gradgrad_vec(e_p)) * dx
        j_gradgradIP = -inner(Gradgrad_vec(v_p), e_q) * dx

        j_grad = j_gradgrad + j_gradgradIP
        j_form = j_grad
        petsc_j = assemble(j_form, mat_type='aij').M.handle
        petsc_m = assemble(m_form, mat_type='aij').M.handle

        J_FEM = np.array(petsc_j.convert("dense").getDenseArray())
        M_FEM = np.array(petsc_m.convert("dense").getDenseArray())

        tab_coord = mesh.coordinates.dat.data
        i_P = find_point(tab_coord, coord_P)[0]

        ndof_Vp = 6
        n_p = n_Vp - ndof_Vp
        n_q = n_Vq
        dof_P = i_P*ndof_Vp

        dofs2dump = list(range(dof_P, dof_P+ndof_Vp))
        dofs2keep = list(set(range(n_V)).difference(set(dofs2dump)))

        J_f = J_FEM
        M_f = M_FEM

        J_f = J_f[dofs2keep, :]
        J_f = J_f[:, dofs2keep]

        M_f = M_f[dofs2keep, :]
        M_f = M_f[:, dofs2keep]

        n_rig = 6  # 3D Spatial motion
        n_fl = n_V - ndof_Vp
        n_tot = n_rig + n_fl

        x_G = Lx/2
        y_G = Ly/2
        x_P, y_P = tab_coord[i_P]

        assert x_P == 0 or x_P == Lx or x_P == 0 or y_P == Ly

        Jxx = assemble(rho*h*(y-y_P)**2*dx)
        Jyy = assemble(rho*h*(x-x_P)**2*dx)

        M_r = np.zeros((n_rig, n_rig))
        m_plate = rho * h * Lx * Ly
        M_r[:3, :3] = m_plate*np.eye(3)
        M_r[2, 3] = m_plate*(y_G - y_P)
        M_r[3, 2] = M_r[2, 3]
        M_r[2, 4] = -m_plate * (x_G - x_P)
        M_r[4, 2] = M_r[2, 4]

        M_r[3][3] = Jxx
        M_r[4][4] = Jyy
        M_r[5][5] = Jxx + Jyy

        M_fr = np.zeros((n_fl, n_rig))
        M_fr[:, 2] = assemble(v_p * rho * h * dx).vector().get_local()[dofs2keep]
        M_fr[:, 3] = assemble(v_p * rho * h * (y - y_P) * dx).vector().get_local()[dofs2keep]
        M_fr[:, 4] = assemble(- v_p * rho * h * (x - x_P) * dx).vector().get_local()[dofs2keep]

        M = np.zeros((n_tot, n_tot))
        M[:n_rig, :n_rig] = M_r
        M[n_rig:, :n_rig] = M_fr
        M[:n_rig, n_rig:] = M_fr.T
        M[n_rig:, n_rig:] = M_f
        assert np.linalg.matrix_rank(M_r) == n_rig
        assert np.linalg.matrix_rank(M_f[:n_p, :n_p]) == n_p
        assert np.linalg.matrix_rank(M_f[n_p:, n_p:]) == n_q
        assert np.linalg.matrix_rank(M[:n_Vp, :n_Vp]) == n_Vp
        print(np.linalg.matrix_rank(M), n_tot)
        assert np.linalg.matrix_rank(M) == n_tot

        J = np.zeros((n_tot, n_tot))
        J[n_rig:, n_rig:] = J_f

        # Dirichlet Boundary Conditions and related constraints
        # The boundary edges in this mesh are numbered as follows:

        # 1: plane x == 0
        # 2: plane x == 1
        # 3: plane y == 0
        # 4: plane y == 1

        n = FacetNormal(mesh)
        # s = as_vector([-n[1], n[0]])

        V_qn = FunctionSpace(mesh, 'Lagrange', 1)
        V_Mnn = FunctionSpace(mesh, 'Lagrange', 1)

        Vu = V_qn * V_Mnn
        n_qn = V_qn.dim()

        q_n, M_nn = TrialFunction(Vu)
        v_omn = dot(grad(v_p), n)

        b_form = v_p * q_n * ds + v_omn * M_nn * ds
        petsc_b = assemble(b_form, mat_type='aij').M.handle
        B_FEM = np.array(petsc_b.convert("dense").getDenseArray())

        n_C = len(coord_C)
        n_u = n_rig*(n_C + 1)
        B = np.zeros((n_tot, n_u))
        B[:n_rig, :n_rig] = np.eye(n_rig)

        Br_C = np.eye(n_rig)
        Bf_C = np.zeros((n_fl, n_rig))

        for i in range(n_C):
            i_C = find_point(tab_coord, coord_C[i])[0]
            x_C, y_C = tab_coord[i_C]

            assert x_C == 0 or x_C == Lx or y_C == 0 or y_C == Ly

            tau_CP = np.eye(3)
            tau_CP[0, 1] = y_C - y_P
            tau_CP[0, 2] = -(x_C - x_P)
            Br_C[2:5, 2:5] = tau_CP.T

            # Force contribution to flexibility
            Bf_C[:, 2] = B_FEM[dofs2keep, i_C]
            # Momentum contribution to flexibility
            if x_C == 0 or x_C == Lx:
                Bf_C[:, 4] = B_FEM[dofs2keep, n_qn + i_C]
            else:
                Bf_C[:, 3] = B_FEM[dofs2keep, n_qn + i_C]

            B[:, (i+1)*n_rig:(i+2)*n_rig] = np.concatenate((Br_C, Bf_C))

        SysPhdaeRig.__init__(self, n_tot, 0, n_rig, n_p, n_q, E=M, J=J, B=B)

        if modes:
            M_FEM[dofs2dump, :] = 0
            M_FEM[:, dofs2dump] = 0
            J_FEM[dofs2dump, :] = 0
            J_FEM[dofs2dump, dofs2dump] = 1

            n_modes = input('Number modes to be visualized:')

            print_modes(M_FEM, J_FEM, mesh, Vp, n_modes)




def find_point(coords, point):

    n_cor = len(coords)
    i_min = 0
    dist_min = np.linalg.norm(coords[i_min] - point)
    for i in range(1, n_cor):
        dist_i = np.linalg.norm(coords[i] - point)
        if dist_i < dist_min:
            dist_min = dist_i
            i_min = i

    return i_min, dist_min


def print_modes(M, J, mesh, Vp, n_modes):
    eigenvalues, eigvectors = la.eig(J, M)
    omega_all = np.imag(eigenvalues)

    index = omega_all > 1e-9

    omega = omega_all[index]
    eigvec_omega = eigvectors[:, index]
    perm = np.argsort(omega)
    eigvec_omega = eigvec_omega[:, perm]

    omega.sort()

    # NonDimensional China Paper

    fntsize = 15

    n_Vp = Vp.dim()
    for i in range(int(n_modes)):
        print("Eigenvalue num " + str(i) + ":" + str(omega[i]))
        eig_real_w = Function(Vp)
        eig_imag_w = Function(Vp)

        eig_real_p = np.real(eigvec_omega[:n_Vp, i])
        eig_imag_p = np.imag(eigvec_omega[:n_Vp, i])
        eig_real_w.vector()[:] = eig_real_p
        eig_imag_w.vector()[:] = eig_imag_p

        Vp_CG = FunctionSpace(mesh, 'Lagrange', 3)
        eig_real_wCG = project(eig_real_w, Vp_CG)
        eig_imag_wCG = project(eig_imag_w, Vp_CG)

        norm_real_eig = np.linalg.norm(eig_real_wCG.vector().get_local())
        norm_imag_eig = np.linalg.norm(eig_imag_wCG.vector().get_local())

        if norm_imag_eig > norm_real_eig:
            triangulation, z_goodeig = _two_dimension_triangle_func_val(eig_imag_wCG, 10)
        else:
            triangulation, z_goodeig = _two_dimension_triangle_func_val(eig_real_wCG, 10)

        figure = plt.figure()
        ax = figure.add_subplot(111, projection="3d")

        ax.plot_trisurf(triangulation, z_goodeig, cmap=cm.jet)

        ax.set_xlabel('$x [m]$', fontsize=fntsize)
        ax.set_ylabel('$y [m]$', fontsize=fntsize)
        ax.set_title('Eigenvector num '+str(i+1), fontsize=fntsize)

        ax.w_zaxis.set_major_locator(LinearLocator(10))
        ax.w_zaxis.set_major_formatter(FormatStrFormatter('%1.2g'))

    plt.show()
