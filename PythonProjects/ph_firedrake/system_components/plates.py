from firedrake import *
import numpy as np
import scipy.linalg as la
from modules_ph.classes_phsystem import SysPhdaeRig, check_positive_matrix
import warnings
np.set_printoptions(threshold=np.inf)
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from firedrake.plot import _two_dimension_triangle_func_val
from mpl_toolkits.mplot3d import Axes3D
from math import pi
plt.rc('text', usetex=True)

def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])




class FloatingKP6dofs_Bell(SysPhdaeRig):

    def __init__(self, Lx, Ly, h, rho, E, nu, nx, ny, coord_P, coord_C=np.empty((0, 2)), modes=False):

        assert len(coord_P) == 2
        assert coord_C.shape[1] == 2
        fl_rot = 12. / (E * h ** 3)
        fl_mem = 1./ (E*h)


        C_b_vec = as_tensor([
            [fl_rot, -nu * fl_rot, 0],
            [-nu * fl_rot, fl_rot, 0],
            [0, 0, fl_rot * 2 * (1 + nu)]
        ])

        C_m_vec = as_tensor([
            [fl_mem, -nu * fl_mem, 0],
            [-nu * fl_mem, fl_mem, 0],
            [0, 0, fl_mem * 2 * (1 + nu)]
        ])

        # Vectorial Formulation possible only
        def membrane_def_vec(S):
            return dot(C_m_vec, S)


        def bending_curv_vec(MM):
            return dot(C_b_vec, MM)

        def gradgrad_vec(w):
            return as_vector([w.dx(0).dx(0), w.dx(1).dx(1), 2 * w.dx(0).dx(1)])

        def Grad_vec(u):
            return as_vector([u[0].dx(0), u[1].dx(1), u[0].dx(1) + u[1].dx(0)])

        mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=False)
        # mesh = Mesh("plate_hole_ref2.msh")

        # plot(mesh); plt.show()

        x, y = SpatialCoordinate(mesh)

        Vp_m = VectorFunctionSpace(mesh, 'CG', 1)
        Vq_m = VectorFunctionSpace(mesh, 'DG', 0, dim=3)

        Vp_b = FunctionSpace(mesh, "Bell", 5)
        Vq_b = VectorFunctionSpace(mesh, "DG", 3, dim=3)

        V = Vp_m * Vp_b * Vq_m * Vq_b
        n_V = V.dim()

        v = TestFunction(V)
        vp_m, vp_b, vq_m, vq_b = split(v)

        e_v = TrialFunction(V)
        ep_m, ep_b, eq_m, eq_b = split(e_v)

        alp_m = rho * h * ep_m

        alp_b = rho * h * ep_b

        alq_m = membrane_def_vec(eq_m)

        alq_b = bending_curv_vec(eq_b)

        dx = Measure('dx')
        ds = Measure('ds')
        mp_m = dot(vp_m, alp_m) * dx
        mp_b = dot(vp_b, alp_b) * dx

        mq_m = inner(vq_m, alq_m) * dx
        mq_b = inner(vq_b, alq_b) * dx


        m_form = mp_m + mp_b + mq_m + mq_b

        j_Grad = inner(vq_m, Grad_vec(ep_m)) * dx
        j_GradIP = -inner(Grad_vec(vp_m), eq_m) * dx

        j_gradgrad = inner(vq_b, gradgrad_vec(ep_b)) * dx
        j_gradgradIP = -inner(gradgrad_vec(vp_b), eq_b) * dx

        j_form = j_gradgrad + j_gradgradIP + j_Grad + j_GradIP

        petsc_j = assemble(j_form, mat_type='aij').M.handle
        petsc_m = assemble(m_form, mat_type='aij').M.handle

        J_FEM = np.array(petsc_j.convert("dense").getDenseArray())
        M_FEM = np.array(petsc_m.convert("dense").getDenseArray())

        tab_coord = mesh.coordinates.dat.data
        i_P = find_point(tab_coord, coord_P)[0]

        ndofVp_b = 3
        ndofVp_m = 2

        np_m1 = Vp_m.sub(0).dim()

        np_m = Vp_m.dim()
        np_b = Vp_b.dim()

        nq_m = Vq_m.dim()
        nq_b = Vq_b.dim()

        n_p = np_m + np_b - ndofVp_b - ndofVp_m
        n_q = nq_m + nq_b


        dofP_m1 = list([i_P])
        dofP_m2 = list([np_m1 + i_P])

        dofP_b = i_P*6 + np_m

        dofs2dump = dofP_m1 + dofP_m2 + list(range(dofP_b, dofP_b + ndofVp_b))
        dofs2keep = list(set(range(n_V)).difference(set(dofs2dump)))

        J_f = J_FEM
        M_f = M_FEM

        J_f = J_f[dofs2keep, :]
        J_f = J_f[:, dofs2keep]

        M_f = M_f[dofs2keep, :]
        M_f = M_f[:, dofs2keep]

        x_P, y_P = tab_coord[i_P]

        # print(x_P, y_P)
        # print(i_P)
        # print(np_m1, dofs2dump)

        n_rig = 6
        n_fl = n_p + n_q
        n_tot = n_rig + n_fl

        Jxx = assemble(rho * h * (y - y_P) ** 2 * dx)
        Jyy = assemble(rho * h * (x - x_P) ** 2 * dx)
        Jzz = assemble(rho * h * ((x - x_P) ** 2 + (y - y_P) ** 2) * dx)
        Jxy = -assemble(rho * h * (x - x_P)*(y - y_P) * dx)
        Jxz = 0
        Jyz = 0

        J_in = np.array([[Jxx, Jxy, 0],
                         [Jxy, Jyy, 0],
                         [0,   0, Jzz]])

        S_x = assemble(rho * h * (x - x_P) * dx)
        S_y = assemble(rho * h * (y - y_P) * dx)

        S_vec = np.array([S_x, S_y, 0])
        M_r = np.zeros((n_rig, n_rig))
        m_plate = rho * h * Lx * Ly


        M_r[:3, :3] = m_plate*np.eye(3)
        M_r[3:, 3:] = J_in
        M_r[3:, :3] = skew(S_vec)
        M_r[:3, 3:] = skew(S_vec).T

        M_fr = np.zeros((n_fl, n_rig))
        M_fr[:, 0] = assemble(+vp_m[0] * rho * h * dx).vector().get_local()[dofs2keep]
        M_fr[:, 1] = assemble(+vp_m[1] * rho * h * dx).vector().get_local()[dofs2keep]
        M_fr[:, 2] = assemble(+vp_b * rho * h * dx).vector().get_local()[dofs2keep]

        M_fr[:, 5] = assemble(-vp_m[0] * rho * h * (y - y_P) * dx).vector().get_local()[dofs2keep]
        M_fr[:, 5] = assemble(+vp_m[1] * rho * h * (x - x_P) * dx).vector().get_local()[dofs2keep]

        M_fr[:, 3] = assemble(+vp_b * rho * h * (y - y_P) * dx).vector().get_local()[dofs2keep]
        M_fr[:, 4] = assemble(-vp_b * rho * h * (x - x_P) * dx).vector().get_local()[dofs2keep]

        M = la.block_diag(M_r, M_f)
        M[n_rig:, :n_rig] = M_fr
        M[:n_rig, n_rig:] = M_fr.T
        if not np.linalg.matrix_rank(M) == n_tot:
            warnings.warn("Singular mass matrix")

        J = np.zeros((n_tot, n_tot))
        J[n_rig:, n_rig:] = J_f

        # B MATRIX

        n_C = coord_C.shape[0]
        n_u = n_rig * (n_C + 1)
        B = np.zeros((n_tot, n_u))
        B[:n_rig, :n_rig] = np.eye(n_rig)


        if modes:
            M_FEM[dofs2dump, :] = 0
            M_FEM[:, dofs2dump] = 0
            J_FEM[dofs2dump, :] = 0
            J_FEM[dofs2dump, dofs2dump] = 1

            n_modes = input('Number modes to be visualized:')

            printmodes_kir(M_FEM, J_FEM, mesh, Vp_b, n_modes)

        SysPhdaeRig.__init__(self, n_tot, 0, n_rig, n_p, n_q, E=M, J=J, B=B)

        FloatingKP6dofs_Bell.vp_m = vp_m
        FloatingKP6dofs_Bell.vp_b = vp_b
        FloatingKP6dofs_Bell.dofs2keep = dofs2keep
        FloatingKP6dofs_Bell.coords = tab_coord
        FloatingKP6dofs_Bell.np_m1 = np_m1

        FloatingKP6dofs_Bell.rho = rho
        FloatingKP6dofs_Bell.h = h

        FloatingKP6dofs_Bell.Vp_x = FunctionSpace(mesh, 'CG', 1)
        FloatingKP6dofs_Bell.Vp_y = FunctionSpace(mesh, 'CG', 1)
        FloatingKP6dofs_Bell.Vp_z = FunctionSpace(mesh, 'Bell', 5)
        FloatingKP6dofs_Bell.x = x
        FloatingKP6dofs_Bell.y = y


    def vec_shapefun(self):

        col1 = assemble(self.vp_m[0]*dx).vector().get_local()[self.dofs2keep]
        col2 = assemble(self.vp_m[1]*dx).vector().get_local()[self.dofs2keep]
        col3 = assemble(self.vp_b*dx).vector().get_local()[self.dofs2keep]

        return np.concatenate((col1.reshape((-1,1)), col2.reshape((-1,1)), col3.reshape((-1,1))), axis=1)

    def matrices_j3d(self):


        ## x_P = (0, 0)
        # Finite element defition
        Vp_x = self.Vp_x
        Vp_y = self.Vp_y
        Vp_z = self.Vp_z

        h = self.h
        rho = self.rho

        x = FloatingKP6dofs_Bell.x
        y = FloatingKP6dofs_Bell.y

        np_x = Vp_x.dim()
        np_y = Vp_y.dim()
        np_z = Vp_z.dim()

        vp_x = TestFunction(Vp_x)
        vp_y = TestFunction(Vp_y)
        vp_z = TestFunction(Vp_z)

        ep_x = TrialFunction(Vp_x)
        ep_y = TrialFunction(Vp_y)
        ep_z = TrialFunction(Vp_z)

        dofs2dump_x = list([0])
        dofs2keep_x = list(set(range(np_x)).difference(set(dofs2dump_x)))

        dofs2dump_y = list([0])
        dofs2dump_z = list([0, 1, 2])

        dofs2keep_y = list(set(range(np_y)).difference(set(dofs2dump_y)))
        dofs2keep_z = list(set(range(np_z)).difference(set(dofs2dump_z)))

        npx_keep = len(dofs2keep_x)
        npy_keep = len(dofs2keep_y)
        npz_keep = len(dofs2keep_z)

        np_keep = npx_keep + npy_keep + npz_keep

        dx = Measure('dx')

        Jf_01tz = assemble(vp_x * rho * h * dx).vector().get_local()[dofs2keep_x]
        Jf_01ry = assemble(- vp_x * rho * h * x * dx).vector().get_local()[dofs2keep_x]

        jf_01fz = vp_x * rho * h * ep_z * dx
        petsc_jf_01fz = assemble(jf_01fz, mat_type='aij').M.handle
        Jf_01fz = np.array(petsc_jf_01fz.convert("dense").getDenseArray())

        Jf_01fz = Jf_01fz[dofs2keep_x, :]
        Jf_01fz = Jf_01fz[:, dofs2keep_z]

        Jf_10tz = assemble(vp_y * rho * h * dx).vector().get_local()[dofs2keep_y]
        Jf_10ry = assemble(-vp_y * rho * h * x * dx).vector().get_local()[dofs2keep_y]
        Jf_12rz = assemble(vp_y * rho * h * y * dx).vector().get_local()[dofs2keep_y]
        Jf_10rx = assemble(vp_y * rho * h * y * dx).vector().get_local()[dofs2keep_y]

        jf_10fz = vp_y * rho * h * ep_z * dx

        petsc_jf_10fz = assemble(jf_10fz, mat_type='aij').M.handle

        Jf_10fz = np.array(petsc_jf_10fz.convert("dense").getDenseArray())

        Jf_10fz = Jf_10fz[dofs2keep_y, :]
        Jf_10fz = Jf_10fz[:, dofs2keep_z]

        Jf_tz = np.zeros((np_keep, 3))
        Jf_tz[:npx_keep, 1] = -Jf_01tz
        Jf_tz[npx_keep:npx_keep + npy_keep, 0] = Jf_10tz

        Jf_ry = np.zeros((np_keep, 3))
        Jf_ry[:npx_keep, 1] = -Jf_01ry
        Jf_ry[npx_keep:npx_keep + npy_keep, 0] = Jf_10ry

        Jf_fz = np.zeros((np_keep, 3, npz_keep))
        Jf_fz[:npx_keep, 1, :] = -Jf_01fz
        Jf_fz[npx_keep:npx_keep + npy_keep, 0, :] = Jf_10fz

        Jf_02ty = assemble(vp_x * rho * h * dx).vector().get_local()[dofs2keep_x]
        Jf_02rz = assemble(vp_x * rho * h * x * dx).vector().get_local()[dofs2keep_x]
        Jf_01rx = assemble(vp_x * rho * h * y * dx).vector().get_local()[dofs2keep_x]

        jf_02fy = vp_x * rho * h * ep_y * dx

        petsc_jf_02fy = assemble(jf_02fy, mat_type='aij').M.handle

        Jf_02fy = np.array(petsc_jf_02fy.convert("dense").getDenseArray())

        Jf_02fy = Jf_02fy[dofs2keep_x, :]
        Jf_02fy = Jf_02fy[:, dofs2keep_y]

        Jf_20ty = assemble(vp_z * rho * h * dx).vector().get_local()[dofs2keep_z]
        Jf_20rz = assemble(vp_z * rho * h * x * dx).vector().get_local()[dofs2keep_z]
        Jf_21rz = assemble(vp_z * rho * h * y * dx).vector().get_local()[dofs2keep_z]


        jf_20fy = vp_z * rho * h * ep_y * dx

        petsc_jf_20fy = assemble(jf_20fy, mat_type='aij').M.handle

        Jf_20fy = np.array(petsc_jf_20fy.convert("dense").getDenseArray())

        Jf_20fy = Jf_20fy[dofs2keep_z, :]
        Jf_20fy = Jf_20fy[:, dofs2keep_y]

        Jf_ty = np.zeros((np_keep, 3))
        Jf_ty[:npx_keep, 2] = Jf_02ty
        Jf_ty[npx_keep + npy_keep:, 0] = -Jf_20ty

        Jf_rx = np.zeros((np_keep, 3))
        Jf_rx[:npx_keep, 1] = -Jf_01rx
        Jf_rx[npx_keep:npx_keep + npy_keep, 0] = Jf_10rx

        Jf_rz = np.zeros((np_keep, 3))
        Jf_rz[:npx_keep, 2] = Jf_02rz
        Jf_rz[npx_keep + npy_keep:, 0] = -Jf_20rz

        Jf_rz[npx_keep:npx_keep + npy_keep, 2] = -Jf_12rz
        Jf_rz[npx_keep + npy_keep:, 1] = Jf_21rz

        Jf_fy = np.zeros((np_keep, 3, npy_keep))
        Jf_fy[:npx_keep, 2, :] = Jf_02fy
        Jf_fy[npx_keep + npy_keep:, 0, :] = -Jf_20fy

        Jf_12tx = assemble(vp_y * rho * h * dx).vector().get_local()[dofs2keep_y]

        jf_12fx = vp_y * rho * h * ep_x * dx

        petsc_jf_12fx = assemble(jf_12fx, mat_type='aij').M.handle

        Jf_12fx = np.array(petsc_jf_12fx.convert("dense").getDenseArray())

        Jf_12fx = Jf_12fx[dofs2keep_y, :]
        Jf_12fx = Jf_12fx[:, dofs2keep_x]

        Jf_21tx = assemble(vp_z * rho * h * dx).vector().get_local()[dofs2keep_z]

        jf_21fx = vp_z * rho * h * ep_x * dx

        petsc_jf_21fx = assemble(jf_21fx, mat_type='aij').M.handle

        Jf_21fx = np.array(petsc_jf_21fx.convert("dense").getDenseArray())

        Jf_21fx = Jf_21fx[dofs2keep_z, :]
        Jf_21fx = Jf_21fx[:, dofs2keep_x]

        Jf_tx = np.zeros((np_keep, 3))
        Jf_tx[npx_keep:npx_keep + npy_keep, 2] = - Jf_12tx
        Jf_tx[npx_keep + npy_keep:, 1] = Jf_21tx

        Jf_fx = np.zeros((np_keep, 3, npx_keep))
        Jf_fx[npx_keep:npx_keep + npy_keep, 2, :] = - Jf_12fx
        Jf_fx[npx_keep + npy_keep:, 1, :] = Jf_21fx

        return Jf_tx, Jf_ty, Jf_tz, Jf_rx, Jf_ry, Jf_rz, Jf_fx, Jf_fy, Jf_fz


class FloatingKP3dofs_Bell(SysPhdaeRig):

    def __init__(self, Lx, Ly, h, rho, E, nu, nx, ny, coord_P, coord_C=np.empty((0, 2)), modes=False):

        assert len(coord_P) == 2
        assert coord_C.shape[1] == 2
        fl_rot = 12. / (E * h ** 3)
        fl_mem = 1./ (E*h)


        C_b_vec = as_tensor([
            [fl_rot, -nu * fl_rot, 0],
            [-nu * fl_rot, fl_rot, 0],
            [0, 0, fl_rot * 2 * (1 + nu)]
        ])

        def bending_curv_vec(MM):
            return dot(C_b_vec, MM)

        def gradgrad_vec(w):
            return as_vector([w.dx(0).dx(0), w.dx(1).dx(1), 2 * w.dx(0).dx(1)])

        mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=False)
        # mesh = Mesh("plate_hole_ref2.msh")

        # plot(mesh); plt.show()

        x, y = SpatialCoordinate(mesh)

        Vp_b = FunctionSpace(mesh, "Bell", 5)
        Vq_b = VectorFunctionSpace(mesh, "DG", 3, dim=3)

        V = Vp_b * Vq_b
        n_V = V.dim()

        v = TestFunction(V)
        vp_b, vq_b = split(v)

        e_v = TrialFunction(V)
        ep_b, eq_b = split(e_v)

        alp_b = rho * h * ep_b


        alq_b = bending_curv_vec(eq_b)

        dx = Measure('dx')
        ds = Measure('ds')
        mp_b = dot(vp_b, alp_b) * dx

        mq_b = inner(vq_b, alq_b) * dx


        m_form = mp_b +  mq_b

        j_gradgrad = inner(vq_b, gradgrad_vec(ep_b)) * dx
        j_gradgradIP = -inner(gradgrad_vec(vp_b), eq_b) * dx

        j_form = j_gradgrad + j_gradgradIP

        petsc_j = assemble(j_form, mat_type='aij').M.handle
        petsc_m = assemble(m_form, mat_type='aij').M.handle

        J_FEM = np.array(petsc_j.convert("dense").getDenseArray())
        M_FEM = np.array(petsc_m.convert("dense").getDenseArray())

        tab_coord = mesh.coordinates.dat.data
        i_P = find_point(tab_coord, coord_P)[0]

        ndofVp_b = 3

        np_b = Vp_b.dim()

        nq_b = Vq_b.dim()

        n_p = np_b - ndofVp_b
        n_q =  nq_b


        dofP_b = i_P*6

        dofs2dump = list(range(dofP_b, dofP_b + ndofVp_b))
        dofs2keep = list(set(range(n_V)).difference(set(dofs2dump)))

        J_f = J_FEM
        M_f = M_FEM

        J_f = J_f[dofs2keep, :]
        J_f = J_f[:, dofs2keep]

        M_f = M_f[dofs2keep, :]
        M_f = M_f[:, dofs2keep]

        x_P, y_P = tab_coord[i_P]

        # print(x_P, y_P)
        # print(i_P)
        # print(np_m1, dofs2dump)

        n_rig = 6
        n_fl = n_p + n_q
        n_tot = n_rig + n_fl

        Jxx = assemble(rho * h * (y - y_P) ** 2 * dx)
        Jyy = assemble(rho * h * (x - x_P) ** 2 * dx)
        Jzz = assemble(rho * h * ((x - x_P) ** 2 + (y - y_P) ** 2) * dx)
        Jxy = -assemble(rho * h * (x - x_P)*(y - y_P) * dx)
        Jxz = 0
        Jyz = 0

        J_in = np.array([[Jxx, Jxy, 0],
                         [Jxy, Jyy, 0],
                         [0,   0, Jzz]])

        S_x = assemble(rho * h * (x - x_P) * dx)
        S_y = assemble(rho * h * (y - y_P) * dx)

        S_vec = np.array([S_x, S_y, 0])
        M_r = np.zeros((n_rig, n_rig))
        m_plate = rho * h * Lx * Ly


        M_r[:3, :3] = m_plate*np.eye(3)
        M_r[3:, 3:] = J_in
        M_r[3:, :3] = skew(S_vec)
        M_r[:3, 3:] = skew(S_vec).T

        M_fr = np.zeros((n_fl, n_rig))
        M_fr[:, 2] = assemble(+vp_b * rho * h * dx).vector().get_local()[dofs2keep]

        M_fr[:, 3] = assemble(+vp_b * rho * h * (y - y_P) * dx).vector().get_local()[dofs2keep]
        M_fr[:, 4] = assemble(-vp_b * rho * h * (x - x_P) * dx).vector().get_local()[dofs2keep]

        M = la.block_diag(M_r, M_f)
        M[n_rig:, :n_rig] = M_fr
        M[:n_rig, n_rig:] = M_fr.T
        if not np.linalg.matrix_rank(M) == n_tot:
            warnings.warn("Singular mass matrix")

        J = np.zeros((n_tot, n_tot))
        J[n_rig:, n_rig:] = J_f

        # B MATRIX

        n_C = coord_C.shape[0]
        n_u = n_rig * (n_C + 1)
        B = np.zeros((n_tot, n_u))
        B[:n_rig, :n_rig] = np.eye(n_rig)


        if modes:
            M_FEM[dofs2dump, :] = 0
            M_FEM[:, dofs2dump] = 0
            J_FEM[dofs2dump, :] = 0
            J_FEM[dofs2dump, dofs2dump] = 1

            n_modes = input('Number modes to be visualized:')

            printmodes_kir(M_FEM, J_FEM, mesh, Vp_b, n_modes)

        SysPhdaeRig.__init__(self, n_tot, 0, n_rig, n_p, n_q, E=M, J=J, B=B)

        FloatingKP3dofs_Bell.vp_b = vp_b
        FloatingKP3dofs_Bell.dofs2keep = dofs2keep
        FloatingKP3dofs_Bell.coords = tab_coord


    def vec_shapefun(self):

        col3 = assemble(self.vp_b*dx).vector().get_local()[self.dofs2keep]

        return col3


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

        # mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=False)
        mesh = Mesh("plate_hole_ref2.msh")

        plot(mesh); plt.show()

        x, y = SpatialCoordinate(mesh)

        name_FEp = "Bell"
        deg_p = 5
        name_FEq = "Hermite"
        deg_q = 3

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
        print(x_P, y_P)

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

        is_clamped = conditional(And(le(x, 0.1), le(y, 0.1)), 1, 0)
        b_Fz = v_p * Fz * is_clamped * ds
        b_Mx = v_grady * Mx * is_clamped * ds
        b_My = -v_gradx * My * is_clamped * ds

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

            print(x_C, y_C)
            # assert x_C == 0 or x_C == Lx or y_C == 0 or y_C == Ly

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

        Gf_P[:, 0] = B_Fz[:, i_P]
        Gf_P[:, 1] = B_Mx[:, i_P]
        Gf_P[:, 2] = B_My[:, i_P]

        # Momentum contribution to flexibility
        # if x_P == 0 or x_P == Lx or x_P == 0 or y_P == Ly:
        #     Gf_P[:, 0] = B_Fz[:, i_P]
        #     Gf_P[:, 1] = B_Mx[:, i_P]
        #     Gf_P[:, 2] = B_My[:, i_P]
        # else:
        #
        #     g_Fz = v_p * Fz * dx
        #     g_Mx = v_p.dx(1) * Mx * dx
        #     g_My = -v_p.dx(0) * My * dx
        #     petsc_gFz = assemble(g_Fz, mat_type='aij').M.handle
        #     G_Fz = np.array(petsc_gFz.convert("dense").getDenseArray())
        #     petsc_gMx = assemble(g_Mx, mat_type='aij').M.handle
        #     G_Mx = np.array(petsc_gMx.convert("dense").getDenseArray())
        #     petsc_gMy = assemble(g_My, mat_type='aij').M.handle
        #     G_My = np.array(petsc_gMy.convert("dense").getDenseArray())
        #
        #     Gf_P[:, 0] = G_Fz[:, i_P]
        #     Gf_P[:, 1] = G_Mx[:, i_P]
        #     Gf_P[:, 2] = G_My[:, i_P]


        Z_u = np.zeros((n_u, n_u))
        Ef_aug = la.block_diag(M_f, Z_u)
        Jf_aug = la.block_diag(J_f, Z_u)

        Gf_all = np.concatenate((Gf_P, B[n_rig:, n_rig:]), axis=1)
        Jf_aug[:n_fl, n_fl:] = Gf_all
        Jf_aug[n_fl:, :n_fl] = -Gf_all.T #-Gf_P.T

        if modes:
            n_modes = input('Number modes to be visualized:')
            printmodes_kir(Ef_aug, Jf_aug, mesh, Vp, n_modes)

        T = la.null_space(Gf_P.T).T

        M_f = T @ M_f @ T.T
        M_fr = T @ M_fr
        J_f = T @ J_f @ T.T

        B_r = B[:n_rig]
        B_f = T @ B[n_rig:]
        B = np.concatenate((B_r, B_f))

        M = la.block_diag(M_r, M_f)
        M[n_rig:, :n_rig] = M_fr
        M[:n_rig, n_rig:] = M_fr.T

        Z_rig = np.zeros((n_rig, n_rig))
        J = la.block_diag(Z_rig, J_f)

        n_lmb = 3  # n_rig
        n_tot = n_e - n_lmb
        n_p = n_p - n_lmb

        if np.linalg.det(M_f) == 0:
            warnings.warn("Singular flexible mass matrix")
        assert check_positive_matrix(M)

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
        name_FEq = "DG"
        deg_q = 3
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

        if modes:
            M_FEM[dofs2dump, :] = 0
            M_FEM[:, dofs2dump] = 0
            J_FEM[dofs2dump, :] = 0
            J_FEM[dofs2dump, dofs2dump] = 1

            n_modes = input('Number modes to be visualized:')

            printmodes_kir(M_FEM, J_FEM, mesh, Vp, n_modes)

        SysPhdaeRig.__init__(self, n_tot, 0, n_rig, n_p, n_q, E=M, J=J, B=B)


class FloatingMP(SysPhdaeRig):

    def __init__(self, Lx, Ly, h, rho, E, nu, nx, ny, coord_P, coord_C=np.empty((0, 2)), modes=False):

        # Mindlin plate written with the port Hamiltonian approach
        k_sh = 5 / 6


        D = E * h ** 3 / (1 - nu ** 2) / 12.
        G = E / 2 / (1 + nu)
        F = G * h * k_sh

        # Useful Matrices

        fl_rot = 12. / (E * h ** 3)

        C_b = as_tensor([
            [fl_rot, -nu * fl_rot, 0],
            [-nu * fl_rot, fl_rot, 0],
            [0, 0, fl_rot * 2 * (1 + nu)]
        ])

        # Operators for vectorial formulation

        def bending_curv_vec(MM):
            return dot(C_b, MM)

        def gradSym_vec(u):
            return as_vector([u[0].dx(0), u[1].dx(1), u[0].dx(1) + u[1].dx(0)])

        # The unit square mesh is divided in :math:`N\times N` quadrilaterals::
        # mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=False)
        mesh = Mesh("plate_hole_ref2.msh")
        x, y = SpatialCoordinate(mesh)

        plot(mesh)
        plt.show()

        # Finite element defition

        deg = 2
        Vp_w = FunctionSpace(mesh, "CG", deg)
        Vp_th = VectorFunctionSpace(mesh, "CG", deg)
        Vq_th = VectorFunctionSpace(mesh, "CG", deg, dim=3)
        Vq_w = VectorFunctionSpace(mesh, "CG", deg)

        V = Vp_w * Vp_th * Vq_th * Vq_w

        v = TestFunction(V)
        vp_w, vp_th, vq_th, vq_w = split(v)

        e = TrialFunction(V)
        ep_w, ep_th, eq_th, eq_w = split(e)

        alp_w = rho * h * ep_w
        alp_th = (rho * h ** 3) / 12. * ep_th
        alq_th = bending_curv_vec(eq_th)
        alq_w = 1. / F * eq_w

        dx = Measure("dx")
        ds = Measure("ds")

        m_form = inner(vp_w, alp_w) * dx + inner(vp_th, alp_th) * dx +\
                 inner(vq_th, alq_th) * dx + inner(vq_w, alq_w) * dx

        # For vector notation

        j_grad = dot(vq_w, grad(ep_w)) * dx
        j_gradIP = -dot(grad(vp_w), eq_w) * dx

        j_gradSym = inner(vq_th, gradSym_vec(ep_th)) * dx
        j_gradSymIP = -inner(gradSym_vec(vp_th), eq_th) * dx

        j_Id = dot(vp_th, eq_w) * dx
        j_IdIP = -dot(vq_w, ep_th) * dx

        j_form = j_grad + j_gradIP + j_gradSym + j_gradSymIP + j_Id + j_IdIP

        # Assemble the stiffness matrix and the mass matrix.

        petsc_j = assemble(j_form, mat_type="aij").M.handle
        petsc_m = assemble(m_form, mat_type="aij").M.handle

        J_f = np.array(petsc_j.convert("dense").getDenseArray())
        M_f = np.array(petsc_m.convert("dense").getDenseArray())

        tab_coord = mesh.coordinates.dat.data
        i_P = find_point(tab_coord, coord_P)[0]

        n_p = Vp_w.dim() + Vp_th.dim()
        n_q = Vq_th.dim() + Vq_w.dim()

        n_rig = 3  # Displacement about z, rotations about x and y
        n_fl = n_p + n_q
        n_e = n_rig + n_fl

        x_P, y_P = tab_coord[i_P]
        print(x_P, y_P)

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
        M_fr[:, 0] = assemble(vp_w * rho * h * dx).vector().get_local()
        M_fr[:, 1] = assemble(vp_w * rho * h * (y - y_P) * dx).vector().get_local()
        M_fr[:, 2] = assemble(- vp_w * rho * h * (x - x_P) * dx).vector().get_local()

        # B matrices based on Lagrange

        Vf = FunctionSpace(mesh, 'CG', 1)
        Fz = TrialFunction(Vf)
        Mx = TrialFunction(Vf)
        My = TrialFunction(Vf)

        v_phix, v_phiy = vp_th

        is_clamped = conditional(And(le(x, 0.1), le(y, 0.1)), 1, 0)
        b_Fz = vp_w * Fz * is_clamped * ds
        b_Mx = v_phiy * Mx * is_clamped * ds
        b_My = -v_phix * My * is_clamped * ds

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
            print(x_C, y_C)
            # assert x_C == 0 or x_C == Lx or y_C == 0 or y_C == Ly

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

        Gf_P[:, 0] = B_Fz[:, i_P]
        Gf_P[:, 1] = B_Mx[:, i_P]
        Gf_P[:, 2] = B_My[:, i_P]

        # # Momentum contribution to flexibility
        # if x_P == 0 or x_P == Lx or x_P == 0 or y_P == Ly:
        #     Gf_P[:, 0] = B_Fz[:, i_P]
        #     Gf_P[:, 1] = B_Mx[:, i_P]
        #     Gf_P[:, 2] = B_My[:, i_P]
        # else:
        #
        #     g_Fz = vp_w * Fz * dx
        #     g_Mx = v_phiy * Mx * dx
        #     g_My = -v_phix * My * dx
        #     petsc_gFz = assemble(g_Fz, mat_type='aij').M.handle
        #     G_Fz = np.array(petsc_gFz.convert("dense").getDenseArray())
        #     petsc_gMx = assemble(g_Mx, mat_type='aij').M.handle
        #     G_Mx = np.array(petsc_gMx.convert("dense").getDenseArray())
        #     petsc_gMy = assemble(g_My, mat_type='aij').M.handle
        #     G_My = np.array(petsc_gMy.convert("dense").getDenseArray())
        #
        #     Gf_P[:, 0] = G_Fz[:, i_P]
        #     Gf_P[:, 1] = G_Mx[:, i_P]
        #     Gf_P[:, 2] = G_My[:, i_P]

        Z_u = np.zeros((n_u, n_u))
        Ef_aug = la.block_diag(M_f, Z_u)
        Jf_aug = la.block_diag(J_f, Z_u)
        Gf = np.concatenate((Gf_P, B[n_rig:, n_rig:]), axis=1)
        Jf_aug[:n_fl, n_fl:] = Gf
        Jf_aug[n_fl:, :n_fl] = -Gf.T



        if modes:
            n_modes = input('Number modes to be visualized:')
            printmodes_min(Ef_aug, Jf_aug, Vp_w, n_modes)

        T = la.null_space(Gf_P.T).T

        M_f = T @ M_f @ T.T
        M_fr = T @ M_fr
        J_f = T @ J_f @ T.T

        B_r = B[:n_rig]
        B_f = T @ B[n_rig:]
        B = np.concatenate((B_r, B_f))

        M = la.block_diag(M_r, M_f)
        M[n_rig:, :n_rig] = M_fr
        M[:n_rig, n_rig:] = M_fr.T

        Z_rig = np.zeros((n_rig, n_rig))
        J = la.block_diag(Z_rig, J_f)

        n_lmb = 3  # n_rig
        n_tot = n_e - n_lmb
        n_p = n_p - n_lmb

        if np.linalg.det(M_f) == 0:
            warnings.warn("Singular mass matrix")
        assert check_positive_matrix(M)

        SysPhdaeRig.__init__(self, n_tot, 0, n_rig, n_p, n_q, E=M, J=J, B=B)


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


def printmodes_kir(M, J, grid, Vp, n_modes):
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
        print("Eigenvalue num " + str(i+1) + ":" + str(omega[i]/(2*pi)))
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


def printmodes_min(M, J, Vp_w, n_modes):
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

    n_Vpw = Vp_w.dim()
    for i in range(int(n_modes)):
        print("Eigenvalue num " + str(i+1) + ":" + str(omega[i]/(2*pi)))
        eig_real_w = Function(Vp_w)
        eig_imag_w = Function(Vp_w)

        eig_real_pw = np.real(eigvec_omega[:n_Vpw, i])
        eig_imag_pw = np.imag(eigvec_omega[:n_Vpw, i])
        eig_real_w.vector()[:] = eig_real_pw
        eig_imag_w.vector()[:] = eig_imag_pw

        norm_real_eig = np.linalg.norm(eig_real_w.vector().get_local())
        norm_imag_eig = np.linalg.norm(eig_imag_w.vector().get_local())

        if norm_imag_eig > norm_real_eig:
            triangulation, z_goodeig = _two_dimension_triangle_func_val(eig_imag_w, 10)
        else:
            triangulation, z_goodeig = _two_dimension_triangle_func_val(eig_real_w, 10)

        figure = plt.figure()
        ax = figure.add_subplot(111, projection="3d")

        ax.plot_trisurf(triangulation, z_goodeig, cmap=cm.jet)

        ax.set_xlabel('$x [m]$', fontsize=fntsize)
        ax.set_ylabel('$y [m]$', fontsize=fntsize)
        ax.set_title('Eigenvector num '+str(i+1), fontsize=fntsize)

        ax.w_zaxis.set_major_locator(LinearLocator(10))
        ax.w_zaxis.set_major_formatter(FormatStrFormatter('%1.2g'))

    plt.show()
