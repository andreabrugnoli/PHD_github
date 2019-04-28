from firedrake import *
import numpy as np
import scipy.linalg as la
from modules_phdae.classes_phsystem import SysPhdaeRig, check_positive_matrix
import warnings
np.set_printoptions(threshold=np.inf)


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

        def gradgrad_vec(u):
            return as_vector([u.dx(0).dx(0), u.dx(1).dx(1), 2 * u.dx(0).dx(1)])

        mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=False)
        x, y = SpatialCoordinate(mesh)

        name_FEp = "Bell"
        deg_p = 5
        name_FEq = "Bell"
        deg_q = 5

        Vp = FunctionSpace(mesh, name_FEp, deg_p)
        Vq = VectorFunctionSpace(mesh, name_FEq, deg_q, dim=3)

        V = Vp * Vq

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

        j_gradgrad = inner(v_q, gradgrad_vec(e_p)) * dx
        j_gradgradIP = -inner(gradgrad_vec(v_p), e_q) * dx

        j_form = j_gradgrad + j_gradgradIP
        petsc_j = assemble(j_form, mat_type='aij').M.handle
        petsc_m = assemble(m_form, mat_type='aij').M.handle

        J_f = np.array(petsc_j.convert("dense").getDenseArray())
        M_f = np.array(petsc_m.convert("dense").getDenseArray())

        tab_coord = mesh.coordinates.dat.data
        i_P = find_point(tab_coord, coord_P)[0]

        n_p = Vp.dim()
        n_q = Vq.dim()

        n_rig = 3  # Displacement about z, rotations about x and y
        n_fl = n_p + n_q
        n_e = n_rig + n_fl

        x_P, y_P = tab_coord[i_P]

        Jxx = assemble(rho * h * (y - y_P) ** 2 * dx)
        Jyy = assemble(rho * h * (x - x_P) ** 2 * dx)

        M_r = np.zeros((n_rig, n_rig))
        m_plate = rho * h * Lx * Ly
        M_r[0, 0] = m_plate
        M_r[1, 1] = Jxx
        M_r[2, 2] = Jyy
        M_r[0, 1] = assemble(rho * h * (y - y_P) * dx)  # m_plate*(y_G - y_P)  #
        M_r[1, 0] = M_r[0, 1]
        M_r[0, 2] = assemble(- rho * h * (x - x_P) * dx)  # -m_plate * (x_G - x_P)  #
        M_r[2, 0] = M_r[0, 2]
        M_r[1, 2] = assemble(- rho * h * (x - x_P) * (y - y_P) * dx)
        M_r[2, 1] = M_r[1, 2]

        M_fr = np.zeros((n_fl, n_rig))
        M_fr[:, 0] = assemble(v_p * rho * h * dx).vector().get_local()
        M_fr[:, 1] = assemble(v_p * rho * h * (y - y_P) * dx).vector().get_local()
        M_fr[:, 2] = assemble(- v_p * rho * h * (x - x_P) * dx).vector().get_local()

        # B matrices based on Lagrange
        Vf = FunctionSpace(mesh, 'CG', 1)
        Fz = TrialFunction(Vf)
        Mx = TrialFunction(Vf)
        My = TrialFunction(Vf)

        v_gradx = v_p.dx(0)
        v_grady = v_p.dx(1)

        b_Fz = v_p * Fz * ds
        b_Mx = v_grady * Mx * ds
        b_My = -v_gradx * My * ds

        petsc_bFz = assemble(b_Fz, mat_type='aij').M.handle
        B_Fz = np.array(petsc_bFz.convert("dense").getDenseArray())
        petsc_bMx = assemble(b_Mx, mat_type='aij').M.handle
        B_Mx = np.array(petsc_bMx.convert("dense").getDenseArray())
        petsc_bMy = assemble(b_My, mat_type='aij').M.handle
        B_My = np.array(petsc_bMy.convert("dense").getDenseArray())

        n_C = coord_C.shape[0]
        n_u = n_rig * (n_C + 1)
        B = np.zeros((n_e, n_u))

        Br_P = np.eye(n_rig)
        B[:n_rig, :n_rig] = Br_P

        for i in range(n_C):
            i_C = find_point(tab_coord, coord_C[i])[0]
            x_C, y_C = tab_coord[i_C]

            assert x_C == 0 or x_C == Lx or y_C == 0 or y_C == Ly

            tau_CP = np.eye(3)
            tau_CP[0, 1] = y_C - y_P
            tau_CP[0, 2] = -(x_C - x_P)
            Br_C = tau_CP.T

            # Force contribution to flexibility

            Bf_C = np.zeros((n_fl, n_rig))
            Bf_C[:, 0] = B_Fz[:, i_C]
            Bf_C[:, 1] = B_Mx[:, i_C]
            Bf_C[:, 2] = B_My[:, i_C]

            B[:, (i + 1) * n_rig:(i + 2) * n_rig] = np.concatenate((Br_C, Bf_C), axis=0)

        Gf_P = np.zeros((n_fl, n_rig))

        # Momentum contribution to flexibility
        if x_P == 0 or x_P == Lx or x_P == 0 or y_P == Ly:
                Gf_P[:, 0] = B_Fz[:, i_P]
                Gf_P[:, 1] = B_Mx[:, i_P]
                Gf_P[:, 2] = B_My[:, i_P]
        else:

            g_Fz = v_p * Fz * dx
            g_Mx = v_p.dx(1) * Mx * dx
            g_My = -v_p.dx(0) * My * dx
            petsc_gFz = assemble(g_Fz, mat_type='aij').M.handle
            G_Fz = np.array(petsc_gFz.convert("dense").getDenseArray())
            petsc_gMx = assemble(g_Mx, mat_type='aij').M.handle
            G_Mx = np.array(petsc_gMx.convert("dense").getDenseArray())
            petsc_gMy = assemble(g_My, mat_type='aij').M.handle
            G_My = np.array(petsc_gMy.convert("dense").getDenseArray())

            Gf_P[:, 0] = G_Fz[:, i_P]
            Gf_P[:, 1] = G_Mx[:, i_P]
            Gf_P[:, 2] = G_My[:, i_P]

        Z_lmb = np.zeros((n_u, n_u))
        Ef_aug = la.block_diag(M_f, Z_lmb)
        Jf_aug = la.block_diag(J_f, Z_lmb)
        Jf_aug[:n_fl, n_fl:] = np.concatenate((Gf_P, B[n_rig:, n_rig:]), axis=1)
        Jf_aug[n_fl:, :n_fl] = -np.concatenate((Gf_P, B[n_rig:, n_rig:]), axis=1).T #-Gf_P.T

        if modes:
            n_modes = input('Number modes to be visualized:')
            print_modes(Ef_aug, Jf_aug, mesh, Vp, n_modes)

        T = la.null_space(Gf_P.T).T

        M_f = T @ M_f @ T.T
        M_fr = T @ M_fr
        J_f = T @ J_f @ T.T

        B_r = B[:n_rig]
        B_f = T @ B[n_rig:]

        M = la.block_diag(M_r, M_f)
        M[n_rig:, :n_rig] = M_fr
        M[:n_rig, n_rig:] = M_fr.T

        J = la.block_diag(Z_lmb, J_f)
        B = np.concatenate((B_r, B_f))

        if not np.linalg.matrix_rank(M) == len(M):
            warnings.warn("Singular mass matrix")
        assert check_positive_matrix(M)

        n_lmb = 3 # n_rig
        n_tot = n_e - n_lmb
        n_p = n_p - n_lmb

        SysPhdaeRig.__init__(self, n_tot, 0, n_rig, n_p, n_q, E=M, J=J, B=B)


class FloatingBellKP(SysPhdaeRig):

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

        # def tensor_divDiv_vec(MM):
        #     return MM[0].dx(0).dx(0) + MM[1].dx(1).dx(1) + 2 * MM[2].dx(0).dx(1)

        def gradgrad_vec(u):
            return as_vector([ u.dx(0).dx(0), u.dx(1).dx(1), 2 * u.dx(0).dx(1) ])

        mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=False)
        x, y = SpatialCoordinate(mesh)

        name_FEp = "Bell"
        deg_p = 5
        name_FEq = "Bell"
        deg_q = 5
        ndof_Vp = 6

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

        j_gradgrad = inner(v_q, gradgrad_vec(e_p)) * dx
        j_gradgradIP = -inner(gradgrad_vec(v_p), e_q) * dx

        j_form = j_gradgrad + j_gradgradIP
        petsc_j = assemble(j_form, mat_type='aij').M.handle
        petsc_m = assemble(m_form, mat_type='aij').M.handle

        J_FEM = np.array(petsc_j.convert("dense").getDenseArray())
        M_FEM = np.array(petsc_m.convert("dense").getDenseArray())

        tab_coord = mesh.coordinates.dat.data
        i_P = find_point(tab_coord, coord_P)[0]

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

        x_P, y_P = tab_coord[i_P]

        print(x_P, y_P)

        # assert x_P == 0 or x_P == Lx or x_P == 0 or y_P == Ly

        Jxx = assemble(rho*h*(y-y_P)**2*dx)
        Jyy = assemble(rho*h*(x-x_P)**2*dx)

        n_rig = 3  # Displacement about z, rotations about x and y
        n_fl = n_p + n_q
        n_tot = n_rig + n_fl

        M_r = np.zeros((n_rig, n_rig))
        m_plate = rho * h * Lx * Ly
        M_r[0, 0] = m_plate
        M_r[1, 1] = Jxx
        M_r[2, 2] = Jyy
        M_r[0, 1] = assemble(+rho * h * (y - y_P) * dx)  # m_plate*(y_G - y_P)  #
        M_r[1, 0] = M_r[0, 1]
        M_r[0, 2] = assemble(-rho * h * (x - x_P) * dx)  # -m_plate * (x_G - x_P)  #
        M_r[2, 0] = M_r[0, 2]
        M_r[1, 2] = assemble(-rho * h * (x - x_P) * (y - y_P) * dx)
        M_r[2, 1] = M_r[1, 2]

        M_fr = np.zeros((n_p + n_q, n_rig))
        M_fr[:, 0] = assemble(+v_p * rho * h * dx).vector().get_local()[dofs2keep]
        M_fr[:, 1] = assemble(+v_p * rho * h * (y - y_P) * dx).vector().get_local()[dofs2keep]
        M_fr[:, 2] = assemble(-v_p * rho * h * (x - x_P) * dx).vector().get_local()[dofs2keep]

        M = la.block_diag(M_r, M_f)
        M[n_rig:, :n_rig] = M_fr
        M[:n_rig, n_rig:] = M_fr.T
        if not np.linalg.matrix_rank(M) == n_tot:
            warnings.warn("Singular mass matrix")

        J = np.zeros((n_tot, n_tot))
        J[n_rig:, n_rig:] = J_f

        # B MATRIX
        Vf = FunctionSpace(mesh, 'CG', 1)
        Fz = TrialFunction(Vf)
        Mx = TrialFunction(Vf)
        My = TrialFunction(Vf)

        v_gradx = v_p.dx(0)
        v_grady = v_p.dx(1)

        b_Fz = v_p * Fz * ds
        b_Mx = v_grady * Mx * ds
        b_My = -v_gradx * My * ds

        petsc_bFz = assemble(b_Fz, mat_type='aij').M.handle
        B_Fz = np.array(petsc_bFz.convert("dense").getDenseArray())
        petsc_bMx = assemble(b_Mx, mat_type='aij').M.handle
        B_Mx = np.array(petsc_bMx.convert("dense").getDenseArray())
        petsc_bMy = assemble(b_My, mat_type='aij').M.handle
        B_My = np.array(petsc_bMy.convert("dense").getDenseArray())

        n_C = coord_C.shape[0]
        n_u = n_rig*(n_C + 1)
        B = np.zeros((n_tot, n_u))
        B[:n_rig, :n_rig] = np.eye(n_rig)

        for i in range(n_C):
            i_C = find_point(tab_coord, coord_C[i])[0]
            x_C, y_C = tab_coord[i_C]

            assert x_C == 0 or x_C == Lx or y_C == 0 or y_C == Ly

            Bf_C = np.zeros((n_fl, n_rig))
            tau_CP = np.eye(3)
            tau_CP[0, 1] = y_C - y_P
            tau_CP[0, 2] = -(x_C - x_P)
            Br_C = tau_CP.T

            # Force contribution to flexibility
            Bf_C[:, 0] = B_Fz[dofs2keep, i_C]
            Bf_C[:, 1] = B_Mx[dofs2keep, i_C]
            Bf_C[:, 2] = B_My[dofs2keep, i_C]

            B[:, (i+1)*n_rig:(i+2)*n_rig] = np.concatenate((Br_C, Bf_C), axis=0)

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


import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from firedrake.plot import _two_dimension_triangle_func_val
from mpl_toolkits.mplot3d import Axes3D
matplotlib.rcParams['text.usetex'] = True


def print_modes(M, J, grid, Vp, n_modes):
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
        print("Eigenvalue num " + str(i+1) + ":" + str(omega[i]))
        eig_real_w = Function(Vp)
        eig_imag_w = Function(Vp)

        eig_real_p = np.real(eigvec_omega[:n_Vp, i])
        eig_imag_p = np.imag(eigvec_omega[:n_Vp, i])
        eig_real_w.vector()[:] = eig_real_p
        eig_imag_w.vector()[:] = eig_imag_p

        Vp_CG = FunctionSpace(grid, 'Lagrange', 3)
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


