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


class NeumannWave(SysPhdaeRig):

    def __init__(self, Lx, Ly, rho, T, nx, ny, modes=False):

        mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=False)

        # plot(mesh); plt.show()

        x, y = SpatialCoordinate(mesh)

        deg_p = 2
        deg_q = 1
        Vp = FunctionSpace(mesh, "CG", deg_p)
        Vq = FunctionSpace(mesh, "RT", deg_q)
        # Vq = VectorFunctionSpace(mesh, "DG", deg_q)

        V = Vp * Vq

        v = TestFunction(V)
        v_p, v_q = split(v)

        e_v = TrialFunction(V)
        e_p, e_q = split(e_v)

        al_p = rho * e_p
        al_q = 1./T * e_q

        dx = Measure('dx')
        ds = Measure('ds')
        m_p = v_p * al_p * dx
        m_q = dot(v_q, al_q) * dx
        m_form = m_p + m_q

        j_grad = dot(v_q, grad(e_p)) * dx
        j_gradIP = -dot(grad(v_p), e_q) * dx

        j_form = j_grad + j_gradIP
        petsc_j = assemble(j_form, mat_type='aij').M.handle
        petsc_m = assemble(m_form, mat_type='aij').M.handle

        JJ = np.array(petsc_j.convert("dense").getDenseArray())
        MM = np.array(petsc_m.convert("dense").getDenseArray())

        n_p = Vp.dim()
        n_q = Vq.dim()

        n_e = V.dim()

        # B matrices based on Lagrange
        Vf = FunctionSpace(mesh, 'CG', deg_p)
        f_N = TrialFunction(Vf)

        # is_clamped = conditional(And(le(x, 0.1), le(y, 0.1)), 1, 0)
        b_N = v_p * f_N * ds(1)

        petsc_bN = assemble(b_N, mat_type='aij').M.handle
        B_N = np.array(petsc_bN.convert("dense").getDenseArray())

        boundary_dofs = np.where(B_N.any(axis=0))[0]
        B_N = B_N[:, boundary_dofs]

        n_u = B_N.shape[1]
        print(n_u)

        Z_u = np.zeros((n_u, n_u))
        Ef_aug = la.block_diag(MM, Z_u)
        Jf_aug = la.block_diag(JJ, Z_u)

        Jf_aug[:n_e, n_e:] = B_N
        Jf_aug[n_e:, :n_e] = -B_N.T

        x, y = SpatialCoordinate(mesh)
        con_dom1 = And(And(gt(x, Lx / 4), lt(x, 3 * Lx / 4)), And(gt(y, Ly / 4), lt(y, 3 * Ly / 4)))
        Dom_f = conditional(con_dom1, 1., 0.)
        B_f = assemble(v_p * Dom_f * dx).vector().get_local()
        B_aug = np.concatenate((B_f, np.zeros(n_u, )), axis=0).reshape((-1, 1))

        if modes:
            n_modes = input('Number modes to be visualized:')
            # printmodes(MM, JJ, Vp, n_modes)
            printmodes(Ef_aug, Jf_aug, Vp, n_modes)

        SysPhdaeRig.__init__(self, n_e+n_u, n_u, 0, n_p, n_q, E=Ef_aug, J=Jf_aug, B=B_aug)



class DirichletWave(SysPhdaeRig):

    def __init__(self, Lx, Ly, rho, T, nx, ny, modes=False):

        mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=False)

        # plot(mesh); plt.show()

        x, y = SpatialCoordinate(mesh)

        deg_p = 0
        deg_q = 1
        Vp = FunctionSpace(mesh, "DG", deg_p)
        Vq = FunctionSpace(mesh, "RT", deg_q)
        # Vq = VectorFunctionSpace(mesh, "Lagrange", deg_q)

        V = Vp * Vq

        v = TestFunction(V)
        v_p, v_q = split(v)

        e_v = TrialFunction(V)
        e_p, e_q = split(e_v)

        al_p = rho * e_p
        al_q = 1./T * e_q

        dx = Measure('dx')
        ds = Measure('ds')
        m_p = v_p * al_p * dx
        m_q = dot(v_q, al_q) * dx
        m_form = m_p + m_q

        j_div = dot(v_p, div(e_q)) * dx
        j_divIP = -dot(div(v_q), e_p) * dx

        j_form = j_div + j_divIP
        petsc_j = assemble(j_form, mat_type='aij').M.handle
        petsc_m = assemble(m_form, mat_type='aij').M.handle

        JJ = np.array(petsc_j.convert("dense").getDenseArray())
        MM = np.array(petsc_m.convert("dense").getDenseArray())

        n_p = Vp.dim()
        n_q = Vq.dim()

        n_e = V.dim()

        # B matrices based on Lagrange
        Vf = FunctionSpace(mesh, 'DG', 0)
        f_D = TrialFunction(Vf)

        n = FacetNormal(mesh)
        # is_clamped = conditional(And(le(x, 0.1), le(y, 0.1)), 1, 0)
        b_D = dot(v_q, n) * f_D * ds(2) + dot(v_q, n) * f_D * ds(3) + dot(v_q, n) * f_D * ds(4)

        petsc_bD = assemble(b_D, mat_type='aij').M.handle
        B_D = np.array(petsc_bD.convert("dense").getDenseArray())

        boundary_dofs = np.where(B_D.any(axis=0))[0]
        B_D = B_D[:, boundary_dofs]

        n_u = B_D.shape[1]

        Z_u = np.zeros((n_u, n_u))
        Ef_aug = la.block_diag(MM, Z_u)
        Jf_aug = la.block_diag(JJ, Z_u)

        Jf_aug[:n_e, n_e:] = B_D
        Jf_aug[n_e:, :n_e] = -B_D.T

        x, y = SpatialCoordinate(mesh)
        con_dom1 = And(And(gt(x, Lx / 4), lt(x, 3 * Lx / 4)), And(gt(y, Ly / 4), lt(y, 3 * Ly / 4)))
        Dom_f = conditional(con_dom1, 1., 0.)
        B_f = assemble(v_p * Dom_f * dx).vector().get_local()
        B_aug = np.concatenate((B_f, np.zeros(n_u, )), axis=0).reshape((-1, 1))

        if modes:
            n_modes = input('Number modes to be visualized:')
            # printmodes(MM, JJ, Vp, n_modes)
            printmodes(Ef_aug, Jf_aug, Vp, n_modes)

        SysPhdaeRig.__init__(self, n_e+n_u, n_u, 0, n_p, n_q, E=Ef_aug, J=Jf_aug, B=B_aug)


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



def printmodes(M, J, Vp, n_modes):
    eigenvalues, eigvectors = la.eig(J, M)
    omega_all = np.imag(eigenvalues)

    index = omega_all > 0

    omega = omega_all[index]
    eigvec_omega = eigvectors[:, index]
    perm = np.argsort(omega)
    eigvec_omega = eigvec_omega[:, perm]

    omega.sort()

    fntsize = 15

    n_Vp = Vp.dim()
    for i in range(int(n_modes)):
        print("Eigenvalue num " + str(i+1) + ":" + str(omega[i]))
        eig_real_p = Function(Vp)
        eig_imag_p = Function(Vp)

        eig_real_p.vector()[:] = np.real(eigvec_omega[:n_Vp, i])
        eig_imag_p.vector()[:] = np.imag(eigvec_omega[:n_Vp, i])

        norm_real_eig = np.linalg.norm(eig_real_p.vector().get_local())
        norm_imag_eig = np.linalg.norm(eig_imag_p.vector().get_local())

        if norm_imag_eig > norm_real_eig:
            triangulation, z_goodeig = _two_dimension_triangle_func_val(eig_imag_p, 10)
        else:
            triangulation, z_goodeig = _two_dimension_triangle_func_val(eig_real_p, 10)

        figure = plt.figure()
        ax = figure.add_subplot(111, projection="3d")

        ax.plot_trisurf(triangulation, z_goodeig, cmap=cm.jet)

        ax.set_xlabel('$x [m]$', fontsize=fntsize)
        ax.set_ylabel('$y [m]$', fontsize=fntsize)
        ax.set_title('Eigenvector num '+str(i+1), fontsize=fntsize)

        ax.w_zaxis.set_major_locator(LinearLocator(10))
        ax.w_zaxis.set_major_formatter(FormatStrFormatter('%1.2g'))

    plt.show()
