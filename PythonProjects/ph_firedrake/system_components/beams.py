# EB beam written with the port Hamiltonian approach

from firedrake import *
import numpy as np
import scipy.linalg as la
from modules_phdae.classes_phsystem import SysPhdaeRig, check_positive_matrix,\
    check_skew_symmetry, permute_rows_columns, permute_rows
import warnings
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)


class FloatingPlanarEB(SysPhdaeRig):

    def __init__(self, n_el, L, rho, A,  E, I, m_joint=0.0, J_joint=0.0):

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

        al_p = rho * A * e_p
        al_q = 1./(E*I) * e_q

        dx = Measure('dx')
        ds = Measure('ds')

        m_form = v_p * al_p * dx + v_q * al_q * dx

        petsc_m = assemble(m_form, mat_type='aij').M.handle
        M_FEM = np.array(petsc_m.convert("dense").getDenseArray())

        j_gradgrad = v_q * e_p.dx(0).dx(0) * dx
        j_gradgradIP = -v_p.dx(0).dx(0) * e_q * dx

        j_form = j_gradgrad + j_gradgradIP

        petcs_j = assemble(j_form, mat_type='aij').M.handle
        J_FEM = np.array(petcs_j.convert("dense").getDenseArray())

        dofs2dump = list([0, 1])
        dofs2keep = list(set(range(n_V)).difference(set(dofs2dump)))

        M_f = M_FEM
        M_f = M_f[:, dofs2keep]
        M_f = M_f[dofs2keep, :]

        J_f = J_FEM
        J_f = J_f[:, dofs2keep]
        J_f = J_f[dofs2keep, :]

        n_rig = 3  # Planar motion
        n_fl = n_V - len(dofs2dump)
        n_tot = n_rig + n_fl
        n_p = n_Vp - 2
        n_q = n_Vq

        M_fr = np.zeros((n_fl, n_rig))
        M_fr[:, 1] = assemble(v_p * rho * A * dx).vector().get_local()[dofs2keep]
        M_fr[:, 2] = assemble(v_p * rho * A * x[0] * dx).vector().get_local()[dofs2keep]

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

        J = np.zeros((n_tot, n_tot))
        J[n_rig:, n_rig:] = J_f

        assert check_positive_matrix(M)

        tau_CP = np.array([[1, 0, 0], [0, 1, L], [0, 0, 1]])
        b_Fy = v_p * ds(2)
        b_Mz = v_p.dx(0) * ds(2)

        B_Fy = assemble(b_Fy).vector().get_local().reshape((-1, 1))[dofs2keep]
        B_Mz = assemble(b_Mz).vector().get_local().reshape((-1, 1))[dofs2keep]

        B = np.zeros((n_tot, 6))
        B[:n_rig, :n_rig] = np.eye(n_rig)
        B[:n_rig, n_rig:] = tau_CP.T
        B[n_rig:, 4:] = np.hstack((B_Fy, B_Mz))

        SysPhdaeRig.__init__(self, n_tot, 0, n_rig, n_p, n_q, E=M, J=J, B=B)


class FloatFlexBeam(SysPhdaeRig):

    def __init__(self, n_el, L, rho, A, E, I, m_joint=0.0, J_joint=0.0):

        mesh = IntervalMesh(n_el, L)
        x = SpatialCoordinate(mesh)

        # Finite element defition
        Vp_x = FunctionSpace(mesh, "Lagrange", 1)
        Vq_x = FunctionSpace(mesh, "Lagrange", 1)
        # Vq_x = FunctionSpace(mesh, "DG", 1)

        Vp_y = FunctionSpace(mesh, "Hermite", 3)
        Vq_y = FunctionSpace(mesh, "Hermite", 3)
        # Vq_y = FunctionSpace(mesh, "DG", 1)

        V = Vp_x * Vp_y * Vq_x * Vq_y

        n_V = V.dim()
        np_x = Vp_x.dim()
        np_y = Vp_y.dim()
        nq_x = Vq_x.dim()
        nq_y = Vq_y.dim()

        n_p = np_x + np_y
        n_q = nq_x + nq_y

        v = TestFunction(V)
        vp_x, vp_y, vq_x, vq_y = split(v)

        e = TrialFunction(V)
        ep_x, ep_y, eq_x, eq_y = split(e)

        alp_x = rho * A * ep_x
        alq_x = 1./(E*A) * eq_x

        alp_y = rho * A * ep_y
        alq_y = 1./(E*I) * eq_y

        dx = Measure('dx')
        ds = Measure('ds')

        mp_form = vp_x * alp_x * dx + vp_y * alp_y * dx
        mq_form = vq_x * alq_x * dx + vq_y * alq_y * dx
        m_form = mp_form + mq_form

        petsc_m = assemble(m_form, mat_type='aij').M.handle
        M_FEM = np.array(petsc_m.convert("dense").getDenseArray())

        dofs2dump = list([0, np_x, np_x + 1])
        dofs2keep = list(set(range(n_V)).difference(set(dofs2dump)))

        n_rig = 3  # Planar motion
        n_fl = n_V - len(dofs2dump)
        n_p = n_p - n_rig
        n_tot = n_rig + n_fl

        M_f = M_FEM
        M_f = M_f[:, dofs2keep]
        M_f = M_f[dofs2keep, :]

        assert len(M_f) == n_fl

        M_fr = np.zeros((n_V, n_rig))
        M_fr[:, 0] = assemble(vp_x * rho * A * dx).vector().get_local()
        M_fr[:, 1] = assemble(vp_y * rho * A * dx).vector().get_local()
        M_fr[:, 2] = assemble(vp_y * rho * A * x[0] * dx).vector().get_local()

        M_fr = M_fr[dofs2keep, :]

        M_r = np.zeros((n_rig, n_rig))
        m_beam = rho * L * A
        J_beam = 1/3 * m_beam * L**2
        M_r[0][0] = m_beam + m_joint
        M_r[1][1] = m_beam + m_joint
        M_r[2][2] = J_beam + J_joint
        M_r[1][2] = m_beam * L/2
        M_r[2][1] = m_beam * L/2

        M = la.block_diag(M_r, M_f)
        M[n_rig:, :n_rig] = M_fr
        M[:n_rig, n_rig:] = M_fr.T

        assert check_positive_matrix(M_r)
        assert check_positive_matrix(M_f)
        assert np.linalg.det(M) != 0
        assert check_positive_matrix(M)

        j_grad_x = vq_x * ep_x.dx(0) * dx
        j_gradIP_x = -vp_x.dx(0) * eq_x * dx

        j_gradgrad_y = vq_y * ep_y.dx(0).dx(0) * dx
        j_gradgradIP_y = -vp_y.dx(0).dx(0) * eq_y * dx

        j_form = j_grad_x + j_gradIP_x + j_gradgrad_y + j_gradgradIP_y

        petcs_j = assemble(j_form, mat_type='aij').M.handle
        J_FEM = np.array(petcs_j.convert("dense").getDenseArray())

        J_f = J_FEM
        J_f = J_f[:, dofs2keep]
        J_f = J_f[dofs2keep, :]

        J = np.zeros((n_tot, n_tot))
        J[n_rig:, n_rig:] = J_f

        tau_CP = np.array([[1, 0, 0], [0, 1, L], [0, 0, 1]])

        b_Fx = vp_x * ds(2)
        b_Fy = vp_y * ds(2)
        b_Mz = vp_y.dx(0) * ds(2)

        B_Fx = assemble(b_Fx).vector().get_local().reshape((-1, 1))[dofs2keep]
        B_Fy = assemble(b_Fy).vector().get_local().reshape((-1, 1))[dofs2keep]
        B_Mz = assemble(b_Mz).vector().get_local().reshape((-1, 1))[dofs2keep]

        B = np.zeros((n_tot, 6))
        B[:n_rig, :n_rig] = np.eye(n_rig)
        B[:n_rig, n_rig:] = tau_CP.T
        B[n_rig:, n_rig:] = np.hstack((B_Fx, B_Fy, B_Mz))

        SysPhdaeRig.__init__(self, n_tot, 0, n_rig, n_p, n_q, E=M, J=J, B=B)


class SpatialBeam(SysPhdaeRig):

    def __init__(self, n_el, L, rho, A, E, I, Jxx, m_joint=0.0, J_joint=np.zeros((3, 3))):

        mesh = IntervalMesh(n_el, L)
        x = SpatialCoordinate(mesh)

        # Finite element defition
        Vp_x = FunctionSpace(mesh, "CG", 1)
        Vq_x = FunctionSpace(mesh, "CG", 1)

        Vp_y = FunctionSpace(mesh, "Hermite", 3)
        Vq_y = FunctionSpace(mesh, "Hermite", 3)

        Vp_z = FunctionSpace(mesh, "Hermite", 3)
        Vq_z = FunctionSpace(mesh, "Hermite", 3)

        V = Vp_x * Vp_y * Vp_z * Vq_x * Vq_y * Vq_z

        n_V = V.dim()
        np_x = Vp_x.dim()
        np_y = Vp_y.dim()
        np_z = Vp_z.dim()

        nq_x = Vq_x.dim()
        nq_y = Vq_y.dim()
        nq_z = Vq_z.dim()

        n_p = np_x + np_y + np_z
        n_q = nq_x + nq_y + nq_z

        assert n_V == n_p + n_q

        v = TestFunction(V)
        vp_x, vp_y, vp_z, vq_x, vq_y, vq_z = split(v)

        e = TrialFunction(V)
        ep_x, ep_y, ep_z, eq_x, eq_y, eq_z = split(e)

        alp_x = rho * A * ep_x
        alq_x = 1./(E*A) * eq_x

        alp_y = rho * A * ep_y
        alq_y = 1./(E*I) * eq_y

        alp_z = rho * A * ep_z
        alq_z = 1. / (E * I) * eq_z

        dx = Measure('dx')
        ds = Measure('ds')

        mp_form = vp_x * alp_x * dx + vp_y * alp_y * dx + vp_z * alp_z * dx

        mq_form = vq_x * alq_x * dx + vq_y * alq_y * dx + vq_z * alq_z * dx

        m_form = mp_form + mq_form

        petsc_m = assemble(m_form, mat_type='aij').M.handle
        M_FEM = np.array(petsc_m.convert("dense").getDenseArray())

        dofs2dump = list([0, np_x, np_x + 1, np_x + np_y, np_x + np_y + 1])
        dofs2keep = list(set(range(n_V)).difference(set(dofs2dump)))
        # print(dofs2keep)

        n_rig = 6  # Spatial motion
        n_fl = n_V - len(dofs2dump)
        n_p = n_p - len(dofs2dump)
        n_tot = n_rig + n_fl

        M_f = M_FEM
        M_f = M_f[:, dofs2keep]
        M_f = M_f[dofs2keep, :]

        assert len(M_f) == n_fl

        M_fr = np.zeros((n_fl, n_rig))
        M_fr[:, 0] = assemble(vp_x * rho * A * dx).vector().get_local()[dofs2keep]
        M_fr[:, 1] = assemble(vp_y * rho * A * dx).vector().get_local()[dofs2keep]
        M_fr[:, 2] = assemble(vp_z * rho * A * dx).vector().get_local()[dofs2keep]

        M_fr[:, 4] = assemble(- vp_z * rho * A * x[0] * dx).vector().get_local()[dofs2keep]
        M_fr[:, 5] = assemble(vp_y * rho * A * x[0] * dx).vector().get_local()[dofs2keep]

        M_r = np.zeros((n_rig, n_rig))

        m_beam = rho * L * A

        Jyy = 1 / 3 * m_beam * L ** 2
        Jzz = 1 / 3 * m_beam * L ** 2
        J_beam = np.diag([Jxx, Jyy, Jzz])
        M_r[:3, :3] = (m_beam + m_joint)* np.eye(3)
        M_r[3:6, 3:6] = J_beam + J_joint

        s = m_beam * L/2 * np.array([[0, 0, 0],
                                    [0,  0, 1],
                                    [0, -1, 0]])

        M_r[:3, 3:6] = s
        M_r[3:6, :3] = s.T

        M = la.block_diag(M_r, M_f)
        M[n_rig:, :n_rig] = M_fr
        M[:n_rig, n_rig:] = M_fr.T

        assert check_positive_matrix(M_r)
        assert check_positive_matrix(M_f)
        assert check_positive_matrix(M)
        assert np.linalg.det(M) != 0

        j_grad_x = vq_x * ep_x.dx(0) * dx
        j_gradIP_x = -vp_x.dx(0) * eq_x * dx

        j_gradgrad_y = vq_y * ep_y.dx(0).dx(0) * dx
        j_gradgradIP_y = -vp_y.dx(0).dx(0) * eq_y * dx

        j_gradgrad_z = vq_z * ep_z.dx(0).dx(0) * dx
        j_gradgradIP_z = -vp_z.dx(0).dx(0) * eq_z * dx

        j_form = j_grad_x + j_gradIP_x + j_gradgrad_y + j_gradgradIP_y + j_gradgrad_z + j_gradgradIP_z

        petcs_j = assemble(j_form, mat_type='aij').M.handle
        J_FEM = np.array(petcs_j.convert("dense").getDenseArray())

        J_f = J_FEM
        J_f = J_f[:, dofs2keep]
        J_f = J_f[dofs2keep, :]

        J = np.zeros((n_tot, n_tot))
        J[n_rig:, n_rig:] = J_f

        tau_CP = np.eye(n_rig)
        tau_CP[1, 5] = L
        tau_CP[2, 4] = - L

        b_Fx = vp_x * ds(2)
        b_Fy = vp_y * ds(2)
        b_Fz = vp_z * ds(2)

        b_My = - vp_z.dx(0) * ds(2)
        b_Mz = vp_y.dx(0) * ds(2)

        B_Fx = assemble(b_Fx).vector().get_local().reshape((-1, 1))[dofs2keep]
        B_Fy = assemble(b_Fy).vector().get_local().reshape((-1, 1))[dofs2keep]
        B_Fz = assemble(b_Fz).vector().get_local().reshape((-1, 1))[dofs2keep]

        B_My = assemble(b_My).vector().get_local().reshape((-1, 1))[dofs2keep]
        B_Mz = assemble(b_Mz).vector().get_local().reshape((-1, 1))[dofs2keep]

        B = np.zeros((n_tot, n_rig*2))
        B[:n_rig, :n_rig] = np.eye(n_rig)
        B[:n_rig, n_rig:] = tau_CP.T
        B[n_rig:, n_rig:n_rig+3] = np.concatenate((B_Fx, B_Fy, B_Fz), axis=1)
        B[n_rig:, n_rig+4:n_rig + 6] = np.concatenate((B_My, B_Mz), axis=1)

        SysPhdaeRig.__init__(self, n_tot, 0, n_rig, n_p, n_q, E=M, J=J, B=B)


class SpatialBeamNoZ(SysPhdaeRig):

    def __init__(self, n_el, L, rho, A, E, I, Jxx, m_joint=0.0, J_joint=np.zeros((3, 3))):

        mesh = IntervalMesh(n_el, L)
        x = SpatialCoordinate(mesh)

        # Finite element defition
        Vp_x = FunctionSpace(mesh, "CG", 1)
        Vq_x = FunctionSpace(mesh, "CG", 1)

        Vp_y = FunctionSpace(mesh, "Hermite", 3)
        Vq_y = FunctionSpace(mesh, "Hermite", 3)

        V = Vp_x * Vp_y * Vq_x * Vq_y

        n_V = V.dim()
        np_x = Vp_x.dim()
        np_y = Vp_y.dim()

        nq_x = Vq_x.dim()
        nq_y = Vq_y.dim()

        n_p = np_x + np_y
        n_q = nq_x + nq_y

        assert n_V == n_p + n_q

        v = TestFunction(V)
        vp_x, vp_y, vq_x, vq_y = split(v)

        e = TrialFunction(V)
        ep_x, ep_y, eq_x, eq_y = split(e)

        alp_x = rho * A * ep_x
        alq_x = 1./(E*A) * eq_x

        alp_y = rho * A * ep_y
        alq_y = 1./(E*I) * eq_y

        dx = Measure('dx')
        ds = Measure('ds')

        mp_form = vp_x * alp_x * dx + vp_y * alp_y * dx

        mq_form = vq_x * alq_x * dx + vq_y * alq_y * dx

        m_form = mp_form + mq_form

        petsc_m = assemble(m_form, mat_type='aij').M.handle
        M_FEM = np.array(petsc_m.convert("dense").getDenseArray())

        dofs2dump = list([0, np_x, np_x + 1])
        dofs2keep = list(set(range(n_V)).difference(set(dofs2dump)))
        # print(dofs2keep)

        n_rig = 6  # Spatial motion
        n_fl = n_V - len(dofs2dump)
        n_p = n_p - len(dofs2dump)
        n_tot = n_rig + n_fl

        M_f = M_FEM
        M_f = M_f[:, dofs2keep]
        M_f = M_f[dofs2keep, :]

        assert len(M_f) == n_fl

        M_fr = np.zeros((n_fl, n_rig))
        M_fr[:, 0] = assemble(vp_x * rho * A * dx).vector().get_local()[dofs2keep]
        M_fr[:, 1] = assemble(vp_y * rho * A * dx).vector().get_local()[dofs2keep]

        M_fr[:, 5] = assemble(vp_y * rho * A * x[0] * dx).vector().get_local()[dofs2keep]

        M_r = np.zeros((n_rig, n_rig))

        m_beam = rho * L * A

        Jyy = 1 / 3 * m_beam * L ** 2
        Jzz = 1 / 3 * m_beam * L ** 2
        J_beam = np.diag([Jxx, Jyy, Jzz])
        M_r[:3, :3] = (m_beam + m_joint)* np.eye(3)
        M_r[3:6, 3:6] = J_beam + J_joint

        s = m_beam * L/2 * np.array([[0, 0, 0],
                                    [0,  0, 1],
                                    [0, -1, 0]])

        M_r[:3, 3:6] = s
        M_r[3:6, :3] = s.T

        M = la.block_diag(M_r, M_f)
        M[n_rig:, :n_rig] = M_fr
        M[:n_rig, n_rig:] = M_fr.T

        assert check_positive_matrix(M_r)
        assert check_positive_matrix(M_f)
        assert check_positive_matrix(M)
        assert np.linalg.det(M) != 0

        j_grad_x = vq_x * ep_x.dx(0) * dx
        j_gradIP_x = -vp_x.dx(0) * eq_x * dx

        j_gradgrad_y = vq_y * ep_y.dx(0).dx(0) * dx
        j_gradgradIP_y = -vp_y.dx(0).dx(0) * eq_y * dx

        j_form = j_grad_x + j_gradIP_x + j_gradgrad_y + j_gradgradIP_y

        petcs_j = assemble(j_form, mat_type='aij').M.handle
        J_FEM = np.array(petcs_j.convert("dense").getDenseArray())

        J_f = J_FEM
        J_f = J_f[:, dofs2keep]
        J_f = J_f[dofs2keep, :]

        J = np.zeros((n_tot, n_tot))
        J[n_rig:, n_rig:] = J_f

        tau_CP = np.eye(n_rig)
        tau_CP[1, 5] = L
        tau_CP[2, 4] = - L

        b_Fx = vp_x * ds(2)
        b_Fy = vp_y * ds(2)

        b_Mz = vp_y.dx(0) * ds(2)

        B_Fx = assemble(b_Fx).vector().get_local().reshape((-1, 1))[dofs2keep]
        B_Fy = assemble(b_Fy).vector().get_local().reshape((-1, 1))[dofs2keep]

        B_Mz = assemble(b_Mz).vector().get_local()[dofs2keep]

        B = np.zeros((n_tot, n_rig*2))
        B[:n_rig, :n_rig] = np.eye(n_rig)
        B[:n_rig, n_rig:] = tau_CP.T
        B[n_rig:, n_rig:n_rig+2] = np.concatenate((B_Fx, B_Fy), axis=1)
        B[n_rig:, n_rig+5] = B_Mz

        SysPhdaeRig.__init__(self, n_tot, 0, n_rig, n_p, n_q, E=M, J=J, B=B)



class SpatialBeamNoTraction(SysPhdaeRig):

    def __init__(self, n_el, L, rho, A, E, I, Jxx, m_joint=0.0, J_joint=np.zeros((3, 3))):

        mesh = IntervalMesh(n_el, L)
        x = SpatialCoordinate(mesh)

        # Finite element defition
        Vp_y = FunctionSpace(mesh, "Hermite", 3)
        Vq_y = FunctionSpace(mesh, "Hermite", 3)

        Vp_z = FunctionSpace(mesh, "Hermite", 3)
        Vq_z = FunctionSpace(mesh, "Hermite", 3)

        V = Vp_y * Vp_z * Vq_y * Vq_z

        n_V = V.dim()
        np_y = Vp_y.dim()
        np_z = Vp_z.dim()

        nq_y = Vq_y.dim()
        nq_z = Vq_z.dim()

        n_p = np_y + np_z
        n_q = nq_y + nq_z

        assert n_V == n_p + n_q

        v = TestFunction(V)
        vp_y, vp_z, vq_y, vq_z = split(v)

        e = TrialFunction(V)
        ep_y, ep_z, eq_y, eq_z = split(e)

        alp_y = rho * A * ep_y
        alq_y = 1./(E*I) * eq_y

        alp_z = rho * A * ep_z
        alq_z = 1. / (E * I) * eq_z

        dx = Measure('dx')
        ds = Measure('ds')

        mp_form = vp_y * alp_y * dx + vp_z * alp_z * dx

        mq_form = vq_y * alq_y * dx + vq_z * alq_z * dx

        m_form = mp_form + mq_form

        petsc_m = assemble(m_form, mat_type='aij').M.handle
        M_FEM = np.array(petsc_m.convert("dense").getDenseArray())

        j_gradgrad_y = vq_y * ep_y.dx(0).dx(0) * dx
        j_gradgradIP_y = -vp_y.dx(0).dx(0) * eq_y * dx

        j_gradgrad_z = vq_z * ep_z.dx(0).dx(0) * dx
        j_gradgradIP_z = -vp_z.dx(0).dx(0) * eq_z * dx

        j_form = j_gradgrad_y + j_gradgradIP_y + \
                 j_gradgrad_z + j_gradgradIP_z

        petcs_j = assemble(j_form, mat_type='aij').M.handle
        J_FEM = np.array(petcs_j.convert("dense").getDenseArray())

        b_Fy = vp_y * ds(2)
        b_Fz = vp_z * ds(2)

        b_My = - vp_z.dx(0) * ds(2)
        b_Mz = vp_y.dx(0) * ds(2)

        B_Fx = np.zeros((n_V, ))
        B_Fy = assemble(b_Fy).vector().get_local()
        B_Fz = assemble(b_Fz).vector().get_local()

        B_Mx = np.zeros((n_V, ))
        B_My = assemble(b_My).vector().get_local()
        B_Mz = assemble(b_Mz).vector().get_local()

        B_FEM = np.column_stack((B_Fx, B_Fy, B_Fz, B_Mx, B_My, B_Mz))

        n_rig = 6  # Spatial motion
        M_fr = np.zeros((n_V, n_rig))

        M_fr[:, 1] = assemble(vp_y * rho * A * dx).vector().get_local()
        M_fr[:, 2] = assemble(vp_z * rho * A * dx).vector().get_local()

        M_fr[:, 4] = assemble(- vp_z * rho * A * x[0] * dx).vector().get_local()
        M_fr[:, 5] = assemble(vp_y * rho * A * x[0] * dx).vector().get_local()

        dofs2dump = list([0, 1, np_y, np_y + 1])
        dofs2keep = list(set(range(n_V)).difference(set(dofs2dump)))
        # print(dofs2keep)

        n_fl = n_V - len(dofs2dump)
        n_p = n_p - len(dofs2dump)
        n_tot = n_rig + n_fl

        M_f = M_FEM[:, dofs2keep]
        M_f = M_f[dofs2keep, :]

        J_f = J_FEM[:, dofs2keep]
        J_f = J_f[dofs2keep, :]

        M_fr = M_fr[dofs2keep]
        B_f = B_FEM[dofs2keep]

        assert len(M_f) == n_fl

        M_r = np.zeros((n_rig, n_rig))

        m_beam = rho * L * A

        Jyy = 1 / 3 * m_beam * L ** 2
        Jzz = 1 / 3 * m_beam * L ** 2
        J_beam = np.diag([Jxx, Jyy, Jzz])
        M_r[:3, :3] = (m_beam + m_joint)* np.eye(3)
        M_r[3:6, 3:6] = J_beam + J_joint

        s = m_beam * L/2 * np.array([[0, 0, 0],
                                    [0,  0, 1],
                                    [0, -1, 0]])

        M_r[:3, 3:6] = s
        M_r[3:6, :3] = s.T

        M = la.block_diag(M_r, M_f)
        M[n_rig:, :n_rig] = M_fr
        M[:n_rig, n_rig:] = M_fr.T

        assert check_positive_matrix(M_r)
        assert check_positive_matrix(M_f)
        assert check_positive_matrix(M)
        assert np.linalg.det(M) != 0

        J = np.zeros((n_tot, n_tot))
        J[n_rig:, n_rig:] = J_f

        tau_CP = np.eye(n_rig)
        tau_CP[1, 5] = L
        tau_CP[2, 4] = - L

        B = np.zeros((n_tot, n_rig*2))
        B[:n_rig, :n_rig] = np.eye(n_rig)
        B[:n_rig, n_rig:] = tau_CP.T
        B[n_rig:, n_rig:2*n_rig] = B_f

        SysPhdaeRig.__init__(self, n_tot, 0, n_rig, n_p, n_q, E=M, J=J, B=B)


class FloatTruss(SysPhdaeRig):

    def __init__(self, n_el, L, rho, A, E):

        mesh = IntervalMesh(n_el, L)
        x = SpatialCoordinate(mesh)

        # Finite element defition
        Vp = FunctionSpace(mesh, "Lagrange", 1)
        Vq = FunctionSpace(mesh, "Lagrange", 1)

        V = Vp * Vq

        n_V = V.dim()
        n_p = Vp.dim()
        n_q = Vq.dim()

        v = TestFunction(V)
        vp, vq = split(v)

        e = TrialFunction(V)
        ep, eq = split(e)

        alp = rho * A * ep
        alq = 1./(E*A) * eq

        dx = Measure('dx')
        ds = Measure('ds')

        mp_form = vp * alp * dx
        mq_form = vq * alq * dx
        m_form = mp_form + mq_form

        petsc_m = assemble(m_form, mat_type='aij').M.handle
        M_FEM = np.array(petsc_m.convert("dense").getDenseArray())

        dofs2dump = list([0])
        dofs2keep = list(set(range(n_V)).difference(set(dofs2dump)))

        n_rig = 3  # Planar motion
        n_fl = n_V - 1
        n_p = n_p - 1
        n_tot = n_rig + n_fl

        M_f = M_FEM
        M_f = M_f[:, dofs2keep]
        M_f = M_f[dofs2keep, :]

        M_fr = np.zeros((n_fl, n_rig))
        M_fr[:, 0] = assemble(vp * rho * dx).vector().get_local()[dofs2keep]

        M_r = np.zeros((n_rig, n_rig))
        m_beam = rho * L * A
        J_beam = 1/3 * m_beam * L**2
        M_r[0][0] = m_beam
        M_r[1][1] = m_beam
        M_r[2][2] = J_beam
        M_r[1][2] = m_beam * L/2
        M_r[2][1] = m_beam * L/2

        M = np.zeros((n_tot, n_tot))
        M[:n_rig, :n_rig] = M_r
        M[n_rig:, :n_rig] = M_fr
        M[:n_rig, n_rig:] = M_fr.T
        M[n_rig:, n_rig:] = M_f

        j_grad = vq * ep.dx(0) * dx
        j_gradIP = -vp.dx(0) * eq * dx

        j_form = j_grad + j_gradIP

        petcs_j = assemble(j_form, mat_type='aij').M.handle
        J_FEM = np.array(petcs_j.convert("dense").getDenseArray())

        J_f = J_FEM
        J_f = J_f[:, dofs2keep]
        J_f = J_f[dofs2keep, :]

        J = np.zeros((n_tot, n_tot))
        J[n_rig:, n_rig:] = J_f

        tau_CP = np.array([[1, 0, 0], [0, 1, L], [0, 0, 1]])
        b_Fx = vp * ds(2)

        B_Fx = assemble(b_Fx).vector().get_local()[dofs2keep]

        B = np.zeros((n_tot, 6))
        B[:n_rig, :n_rig] = np.eye(n_rig)
        B[:n_rig, n_rig:] = tau_CP.T
        B[n_rig:, 3] = B_Fx

        SysPhdaeRig.__init__(self, n_tot, 0, n_rig, n_p, n_q, E=M, J=J, B=B)


class FreeEB(SysPhdaeRig):

    def __init__(self, n_el, L, rho, A, E, I):
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

        al_p = rho * A * e_p
        al_q = 1. / (E * I) * e_q

        dx = Measure('dx')
        ds = Measure('ds')

        m_form = v_p * al_p * dx + v_q * al_q * dx

        petsc_m = assemble(m_form, mat_type='aij').M.handle
        M = np.array(petsc_m.convert("dense").getDenseArray())

        assert check_positive_matrix(M)

        j_gradgrad = v_q * e_p.dx(0).dx(0) * dx
        j_gradgradIP = -v_p.dx(0).dx(0) * e_q * dx

        j_form = j_gradgrad + j_gradgradIP

        petcs_j = assemble(j_form, mat_type='aij').M.handle
        J = np.array(petcs_j.convert("dense").getDenseArray())

        assert check_skew_symmetry(J)

        b0_Fy = v_p * ds(1)
        b0_Mz = v_p.dx(0) * ds(1)
        bL_Fy = v_p * ds(2)
        bL_Mz = v_p.dx(0) * ds(2)

        B0_Fy = assemble(b0_Fy).vector().get_local().reshape((-1, 1))
        B0_Mz = assemble(b0_Mz).vector().get_local().reshape((-1, 1))
        BL_Fy = assemble(bL_Fy).vector().get_local().reshape((-1, 1))
        BL_Mz = assemble(bL_Mz).vector().get_local().reshape((-1, 1))

        B = np.hstack((B0_Fy, B0_Mz, BL_Fy, BL_Mz))

        SysPhdaeRig.__init__(self, n_V, 0, 0, n_Vp, n_Vq, E=M, J=J, B=B)


class ClampedEB(SysPhdaeRig):

    def __init__(self, n_el, L, rho, A, E, I):
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

        al_p = rho * A * e_p
        al_q = 1. / (E * I) * e_q

        dx = Measure('dx')
        ds = Measure('ds')

        m_form = v_p * al_p * dx + v_q * al_q * dx

        petsc_m = assemble(m_form, mat_type='aij').M.handle
        M = np.array(petsc_m.convert("dense").getDenseArray())

        assert check_positive_matrix(M)

        j_divDiv = -v_p * e_q.dx(0).dx(0) * dx
        j_divDivIP = v_q.dx(0).dx(0) * e_p * dx

        j_form = j_divDiv + j_divDivIP

        petcs_j = assemble(j_form, mat_type='aij').M.handle
        J = np.array(petcs_j.convert("dense").getDenseArray())

        assert check_skew_symmetry(J)

        b0_wy = v_q.dx(0) * ds(1)
        b0_phiz = - v_q * ds(1)
        bL_wy = - v_q.dx(0) * ds(2)
        bL_phiz = v_q * ds(2)

        B0_wy = assemble(b0_wy).vector().get_local().reshape((-1, 1))
        B0_phiz = assemble(b0_phiz).vector().get_local().reshape((-1, 1))
        BL_wy = assemble(bL_wy).vector().get_local().reshape((-1, 1))
        BL_phiz = assemble(bL_phiz).vector().get_local().reshape((-1, 1))

        B = np.hstack((B0_wy, B0_phiz, BL_wy, BL_phiz))

        SysPhdaeRig.__init__(self, n_V, 0, 0, n_Vp, n_Vq, E=M, J=J, B=B)


class FreeTruss(SysPhdaeRig):

    def __init__(self, n_el, L, rho, A, E):
        mesh = IntervalMesh(n_el, L)
        x = SpatialCoordinate(mesh)

        # Finite element defition
        deg = 1
        Vp = FunctionSpace(mesh, "Lagrange", deg)
        Vq = FunctionSpace(mesh, "Lagrange", deg)

        V = Vp * Vq
        n_Vp = Vp.dim()
        n_Vq = Vq.dim()
        n_V = V.dim()

        v = TestFunction(V)
        v_p, v_q = split(v)

        e = TrialFunction(V)
        e_p, e_q = split(e)

        al_p = rho * A * e_p
        al_q = 1. / (E * A) * e_q

        dx = Measure('dx')
        ds = Measure('ds')

        m_form = v_p * al_p * dx + v_q * al_q * dx

        petsc_m = assemble(m_form, mat_type='aij').M.handle
        M = np.array(petsc_m.convert("dense").getDenseArray())

        assert check_positive_matrix(M)

        j_grad = v_q * e_p.dx(0) * dx
        j_gradIP = -v_p.dx(0) * e_q * dx

        j_form = j_grad + j_gradIP

        petcs_j = assemble(j_form, mat_type='aij').M.handle
        J = np.array(petcs_j.convert("dense").getDenseArray())

        assert check_skew_symmetry(J)

        b0_N = v_p * ds(1)
        bL_N = v_p * ds(2)

        B0_Fy = assemble(b0_N).vector().get_local().reshape((-1, 1))
        BL_Fy = assemble(bL_N).vector().get_local().reshape((-1, 1))

        B = np.hstack((B0_Fy, BL_Fy))

        SysPhdaeRig.__init__(self, n_V, 0, 0, n_Vp, n_Vq, E=M, J=J, B=B)


class ClampedTruss(SysPhdaeRig):

    def __init__(self, n_el, L, rho, A, E):
        mesh = IntervalMesh(n_el, L)
        x = SpatialCoordinate(mesh)

        # Finite element defition
        deg = 1
        Vp = FunctionSpace(mesh, "Lagrange", deg)
        Vq = FunctionSpace(mesh, "Lagrange", deg)

        V = Vp * Vq
        n_Vp = Vp.dim()
        n_Vq = Vq.dim()
        n_V = V.dim()

        v = TestFunction(V)
        v_p, v_q = split(v)

        e = TrialFunction(V)
        e_p, e_q = split(e)

        al_p = rho * A * e_p
        al_q = 1. / (E * A) * e_q

        dx = Measure('dx')
        ds = Measure('ds')

        m_form = v_p * al_p * dx + v_q * al_q * dx

        petsc_m = assemble(m_form, mat_type='aij').M.handle
        M = np.array(petsc_m.convert("dense").getDenseArray())

        assert check_positive_matrix(M)

        j_div = v_p * e_q.dx(0) * dx
        j_divIP = -v_q.dx(0) * e_p * dx

        j_form = j_div + j_divIP

        petcs_j = assemble(j_form, mat_type='aij').M.handle
        J = np.array(petcs_j.convert("dense").getDenseArray())

        assert check_skew_symmetry(J)

        b0_D = -v_q * ds(1)
        bL_D = v_q * ds(2)

        B0_D = assemble(b0_D).vector().get_local().reshape((-1, 1))
        BL_D = assemble(bL_D).vector().get_local().reshape((-1, 1))

        B = np.hstack((B0_D, BL_D))

        SysPhdaeRig.__init__(self, n_V, 0, 0, n_Vp, n_Vq, E=M, J=J, B=B)


def draw_deformation(n_draw, v_rig, v_fl, L):
    # Suppose no displacement in zero
    assert len(v_rig) == 3
    assert len(v_fl) % 3 == 0
    n_el = int(len(v_fl)/3)

    ufl_dofs = v_fl[:n_el]
    wfl_dofs = v_fl[n_el:]

    dx_el = L/n_el

    u_P = v_rig[0]
    w_P = v_rig[1]
    th_P = v_rig[2]

    x_coord = np.linspace(0, L, num=n_draw)

    u_r = u_P*np.ones_like(x_coord)
    w_r = w_P*np.ones_like(x_coord) + x_coord*th_P

    u_fl = np.zeros_like(x_coord)
    w_fl = np.zeros_like(x_coord)

    i_el = 0
    xin_elem = i_el * dx_el

    for i in range(n_draw):

        x_til = (x_coord[i] - xin_elem) / dx_el

        if x_til > 1:
            i_el = i_el + 1
            if i_el == n_el:
                i_el = i_el - 1

            xin_elem = i_el * dx_el
            x_til = (x_coord[i] - xin_elem) / dx_el

        phi1_u = (1 - x_til)
        phi2_u = x_til

        phi1_w = 1 - 3 * x_til ** 2 + 2 * x_til ** 3
        phi2_w = x_til - 2 * x_til ** 2 + x_til ** 3
        phi3_w = + 3 * x_til ** 2 - 2 * x_til ** 3
        phi4_w = + x_til ** 3 - x_til ** 2

        if i_el == 0:
            u_fl[i] = phi2_u * ufl_dofs[i_el]
            w_fl[i] = phi3_w * wfl_dofs[2 * i_el] + phi4_w * wfl_dofs[2 * i_el + 1]
        else:
            u_fl[i] = phi1_u * ufl_dofs[i_el-1] + phi2_u * ufl_dofs[i_el]
            w_fl[i] = phi1_w * wfl_dofs[2 * (i_el - 1)] + phi2_w * wfl_dofs[2 * i_el - 1] + \
                        phi3_w * wfl_dofs[2 * i_el] + phi4_w * wfl_dofs[2 * i_el + 1]

    u_tot = u_r + u_fl
    w_tot = w_r + w_fl

    return x_coord, u_tot, w_tot


def draw_deformation3D(n_draw, v_rig, v_fl, L):
    # Suppose no displacement in zero
    assert len(v_rig) == 6
    assert len(v_fl) % 5 == 0
    n_el = int(len(v_fl)/5)

    ufl_dofs = v_fl[:n_el]
    vfl_dofs = v_fl[n_el:3*n_el]
    wfl_dofs = v_fl[3 * n_el:]

    dx_el = L/n_el

    u_P = v_rig[0]
    v_P = v_rig[1]
    w_P = v_rig[2]

    omx_P = v_rig[3]
    omy_P = v_rig[4]
    omz_P = v_rig[5]

    x_coord = np.linspace(0, L, num=n_draw)

    u_r = u_P*np.ones_like(x_coord)
    v_r = v_P * np.ones_like(x_coord) + x_coord * omz_P
    w_r = w_P*np.ones_like(x_coord) - x_coord*omy_P

    u_fl = np.zeros_like(x_coord)
    v_fl = np.zeros_like(x_coord)
    w_fl = np.zeros_like(x_coord)

    i_el = 0
    xin_elem = i_el * dx_el

    for i in range(n_draw):

        x_til = (x_coord[i] - xin_elem) / dx_el

        if x_til > 1:
            i_el = i_el + 1
            if i_el == n_el:
                i_el = i_el - 1

            xin_elem = i_el * dx_el
            x_til = (x_coord[i] - xin_elem) / dx_el

        phi1_u = (1 - x_til)
        phi2_u = x_til

        phi1_w = 1 - 3 * x_til ** 2 + 2 * x_til ** 3
        phi2_w = x_til - 2 * x_til ** 2 + x_til ** 3
        phi3_w = + 3 * x_til ** 2 - 2 * x_til ** 3
        phi4_w = + x_til ** 3 - x_til ** 2

        if i_el == 0:
            u_fl[i] = phi2_u * ufl_dofs[i_el]
            v_fl[i] = phi3_w * vfl_dofs[2 * i_el] + phi4_w * vfl_dofs[2 * i_el + 1]
            w_fl[i] = phi3_w * wfl_dofs[2 * i_el] + phi4_w * wfl_dofs[2 * i_el + 1]
        else:
            u_fl[i] = phi1_u * ufl_dofs[i_el-1] + phi2_u * ufl_dofs[i_el]

            v_fl[i] = phi1_w * vfl_dofs[2 * (i_el - 1)] + phi2_w * vfl_dofs[2 * i_el - 1] + \
                      phi3_w * vfl_dofs[2 * i_el] + phi4_w * vfl_dofs[2 * i_el + 1]

            w_fl[i] = phi1_w * wfl_dofs[2 * (i_el - 1)] + phi2_w * wfl_dofs[2 * i_el - 1] + \
                        phi3_w * wfl_dofs[2 * i_el] + phi4_w * wfl_dofs[2 * i_el + 1]

    u_tot = u_r + u_fl
    v_tot = v_r + v_fl
    w_tot = w_r + w_fl

    return x_coord, u_tot, v_tot, w_tot


def draw_bending(n_draw, v_rig, v_fl, L):
    # Suppose no displacement in zero
    assert len(v_rig) == 3
    assert len(v_fl) % 2 == 0
    n_el = int(len(v_fl) / 2)

    wfl_dofs = v_fl

    dx_el = L / n_el

    u_P = v_rig[0]
    w_P = v_rig[1]
    th_P = v_rig[2]

    x_coord = np.linspace(0, L, num=n_draw)

    u_r = u_P * np.ones_like(x_coord)
    w_r = w_P * np.ones_like(x_coord) + x_coord * th_P

    w_fl = np.zeros_like(x_coord)

    i_el = 0
    xin_elem = i_el * dx_el

    for i in range(n_draw):

        x_til = (x_coord[i] - xin_elem) / dx_el

        if x_til > 1:
            i_el = i_el + 1
            if i_el == n_el:
                i_el = i_el - 1

            xin_elem = i_el * dx_el
            x_til = (x_coord[i] - xin_elem) / dx_el

        phi1_w = 1 - 3 * x_til ** 2 + 2 * x_til ** 3
        phi2_w = x_til - 2 * x_til ** 2 + x_til ** 3
        phi3_w = + 3 * x_til ** 2 - 2 * x_til ** 3
        phi4_w = + x_til ** 3 - x_til ** 2

        if i_el == 0:
            w_fl[i] = phi3_w * wfl_dofs[2 * i_el] + phi4_w * wfl_dofs[2 * i_el + 1]
        else:
            w_fl[i] = phi1_w * wfl_dofs[2 * (i_el - 1)] + phi2_w * wfl_dofs[2 * i_el - 1] + \
                        phi3_w * wfl_dofs[2 * i_el] + phi4_w * wfl_dofs[2 * i_el + 1]

    u_tot = u_r
    w_tot = w_r + w_fl

    return x_coord, u_tot, w_tot


def draw_allbending(n_draw, v_rig, v_fl, L):
    # Suppose displacement in every point
    assert len(v_rig) == 3
    assert len(v_fl) % 2 == 0
    n_el = int(len(v_fl) / 2) - 1

    wfl_dofs = v_fl

    dx_el = L / n_el

    u_P = v_rig[0]
    w_P = v_rig[1]
    th_P = v_rig[2]

    x_coord = np.linspace(0, L, num=n_draw)

    u_r = u_P * np.ones_like(x_coord)
    w_r = w_P * np.ones_like(x_coord) + x_coord * th_P

    w_fl = np.zeros_like(x_coord)

    i_el = 0
    xin_elem = i_el * dx_el

    for i in range(n_draw):

        x_til = (x_coord[i] - xin_elem) / dx_el

        if x_til > 1:
            i_el = i_el + 1
            if i_el == n_el:
                i_el = i_el - 1

            xin_elem = i_el * dx_el
            x_til = (x_coord[i] - xin_elem) / dx_el

        phi1_w = 1 - 3 * x_til ** 2 + 2 * x_til ** 3
        phi2_w = x_til - 2 * x_til ** 2 + x_til ** 3
        phi3_w = + 3 * x_til ** 2 - 2 * x_til ** 3
        phi4_w = + x_til ** 3 - x_til ** 2

        w_fl[i] = phi1_w * wfl_dofs[2 * i_el] + phi2_w * wfl_dofs[2 * i_el + 1] + \
                    phi3_w * wfl_dofs[2 * i_el + 2] + phi4_w * wfl_dofs[2 * i_el + 3]

    u_tot = u_r
    w_tot = w_r + w_fl

    return x_coord, u_tot, w_tot


def matrices_j2d(n_el, L, rho, A):

    mesh = IntervalMesh(n_el, L)
    x = SpatialCoordinate(mesh)

    # Finite element defition
    Vp_x = FunctionSpace(mesh, "Lagrange", 1)
    Vp_y = FunctionSpace(mesh, "Hermite", 3)

    np_x = Vp_x.dim()
    np_y = Vp_y.dim()

    vp_x = TestFunction(Vp_x)
    vp_y = TestFunction(Vp_y)

    ep_x = TrialFunction(Vp_x)
    ep_y = TrialFunction(Vp_y)

    dofs2dump_x = list([0])
    dofs2keep_x = list(set(range(np_x)).difference(set(dofs2dump_x)))

    dofs2dump_y = list([0, 1])
    dofs2keep_y = list(set(range(np_y)).difference(set(dofs2dump_y)))

    dx = Measure('dx')

    Jf_tx = assemble(vp_y * rho * A * dx).vector().get_local()[dofs2keep_y]
    Jf_ty = assemble(vp_x * rho * A * dx).vector().get_local()[dofs2keep_x]

    Jf_rz = assemble(vp_x * rho * A * x[0] * dx).vector().get_local()[dofs2keep_x]

    jf_fx = vp_y * rho * A * ep_x * dx
    jf_fy = vp_x * rho * A * ep_y * dx

    petsc_jf_x = assemble(jf_fx, mat_type='aij').M.handle
    petsc_jf_y = assemble(jf_fy, mat_type='aij').M.handle

    Jf_fx = np.array(petsc_jf_x.convert("dense").getDenseArray())
    Jf_fy = np.array(petsc_jf_y.convert("dense").getDenseArray())

    Jf_fx = Jf_fx[:, dofs2keep_x]
    Jf_fx = Jf_fx[dofs2keep_y, :]

    Jf_fy = Jf_fy[:, dofs2keep_y]
    Jf_fy = Jf_fy[dofs2keep_x, :]

    return Jf_tx, Jf_ty, Jf_rz, Jf_fx, Jf_fy


def matrices_j3d(n_el, L, rho, A):

    mesh = IntervalMesh(n_el, L)
    x = SpatialCoordinate(mesh)

    # Finite element defition
    Vp_x = FunctionSpace(mesh, "Lagrange", 1)
    Vp_y = FunctionSpace(mesh, "Hermite", 3)
    Vp_z = FunctionSpace(mesh, "Hermite", 3)

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

    dofs2dump_y = list([0, 1])
    dofs2keep_y = list(set(range(np_y)).difference(set(dofs2dump_y)))

    dofs2dump_z = list([0, 1])
    dofs2keep_z = list(set(range(np_z)).difference(set(dofs2dump_z)))

    npx_keep = len(dofs2keep_x)
    npy_keep = len(dofs2keep_y)
    npz_keep = len(dofs2keep_z)

    np_keep = npx_keep + npy_keep + npz_keep

    dx = Measure('dx')

    Jf_01tz = assemble(vp_x * rho * A * dx).vector().get_local()[dofs2keep_x]
    Jf_01ry = assemble(- vp_x * rho * A * x[0] * dx).vector().get_local()[dofs2keep_x]

    jf_01fz = vp_x * rho * A * ep_z * dx
    petsc_jf_01fz = assemble(jf_01fz, mat_type='aij').M.handle
    Jf_01fz = np.array(petsc_jf_01fz.convert("dense").getDenseArray())

    Jf_01fz = Jf_01fz[dofs2keep_x, :]
    Jf_01fz = Jf_01fz[:, dofs2keep_z]

    Jf_10tz = assemble(vp_y * rho * A * dx).vector().get_local()[dofs2keep_y]
    Jf_10ry = assemble(-vp_y * rho * A * x[0] * dx).vector().get_local()[dofs2keep_y]

    jf_10fz = vp_y * rho * A * ep_z * dx

    petsc_jf_10fz = assemble(jf_10fz, mat_type='aij').M.handle

    Jf_10fz = np.array(petsc_jf_10fz.convert("dense").getDenseArray())

    Jf_10fz = Jf_10fz[dofs2keep_y, :]
    Jf_10fz = Jf_10fz[:, dofs2keep_z]

    Jf_tz = np.zeros((np_keep, 3))
    Jf_tz[:npx_keep, 1] = -Jf_01tz
    Jf_tz[npx_keep:npx_keep+npy_keep, 0] = Jf_10tz

    Jf_ry = np.zeros((np_keep, 3))
    Jf_ry[:npx_keep, 1] = -Jf_01ry
    Jf_ry[npx_keep:npx_keep+npy_keep, 0] = Jf_10ry

    Jf_fz = np.zeros((np_keep, 3, npz_keep))
    Jf_fz[:npx_keep, 1, :] = -Jf_01fz
    Jf_fz[npx_keep:npx_keep+npy_keep, 0, :] = Jf_10fz




    Jf_02ty = assemble(vp_x * rho * A * dx).vector().get_local()[dofs2keep_x]
    Jf_02rz = assemble(vp_x * rho * A * x[0] * dx).vector().get_local()[dofs2keep_x]

    jf_02fy = vp_x * rho * A * ep_y * dx

    petsc_jf_02fy = assemble(jf_02fy, mat_type='aij').M.handle

    Jf_02fy = np.array(petsc_jf_02fy.convert("dense").getDenseArray())

    Jf_02fy = Jf_02fy[dofs2keep_x, :]
    Jf_02fy = Jf_02fy[:, dofs2keep_y]

    Jf_20ty = assemble(vp_z * rho * A * dx).vector().get_local()[dofs2keep_z]
    Jf_20rz = assemble(vp_z * rho * A * x[0] * dx).vector().get_local()[dofs2keep_z]

    jf_20fy = vp_z * rho * A * ep_y * dx

    petsc_jf_20fy = assemble(jf_20fy, mat_type='aij').M.handle

    Jf_20fy = np.array(petsc_jf_20fy.convert("dense").getDenseArray())

    Jf_20fy = Jf_20fy[dofs2keep_z, :]
    Jf_20fy = Jf_20fy[:, dofs2keep_y]

    Jf_ty = np.zeros((np_keep, 3))
    Jf_ty[:npx_keep, 2] = Jf_02ty
    Jf_ty[npx_keep + npy_keep:, 0] = -Jf_20ty

    Jf_rz = np.zeros((np_keep, 3))
    Jf_rz[:npx_keep, 2] = Jf_02rz
    Jf_rz[npx_keep + npy_keep:, 0] = -Jf_20rz

    Jf_fy = np.zeros((np_keep, 3, npy_keep))
    Jf_fy[:npx_keep, 2, :] = Jf_02fy
    Jf_fy[npx_keep + npy_keep:, 0, :] = -Jf_20fy



    Jf_12tx = assemble(vp_y * rho * A * dx).vector().get_local()[dofs2keep_y]

    jf_12fx = vp_y * rho * A * ep_x * dx

    petsc_jf_12fx = assemble(jf_12fx, mat_type='aij').M.handle

    Jf_12fx = np.array(petsc_jf_12fx.convert("dense").getDenseArray())

    Jf_12fx = Jf_12fx[dofs2keep_y, :]
    Jf_12fx = Jf_12fx[:, dofs2keep_x]

    Jf_21tx = assemble(vp_z * rho * A * dx).vector().get_local()[dofs2keep_z]

    jf_21fx = vp_z * rho * A * ep_x * dx

    petsc_jf_21fx = assemble(jf_21fx, mat_type='aij').M.handle

    Jf_21fx = np.array(petsc_jf_21fx.convert("dense").getDenseArray())

    Jf_21fx = Jf_21fx[dofs2keep_z, :]
    Jf_21fx = Jf_21fx[:, dofs2keep_x]

    Jf_tx = np.zeros((np_keep, 3))
    Jf_tx[npx_keep:npx_keep+npy_keep, 2] = - Jf_12tx
    Jf_tx[npx_keep + npy_keep:, 1] = Jf_21tx

    Jf_fx = np.zeros((np_keep, 3, npx_keep))
    Jf_fx[npx_keep:npx_keep+ npy_keep, 2, :] = - Jf_12fx
    Jf_fx[npx_keep + npy_keep:, 1, :] = Jf_21fx

    return Jf_tx, Jf_ty, Jf_tz, Jf_ry, Jf_rz, Jf_fx, Jf_fy, Jf_fz


def matrices_j3d_notraction(n_el, L, rho, A):

    mesh = IntervalMesh(n_el, L)
    x = SpatialCoordinate(mesh)

    # Finite element defition
    Vp_y = FunctionSpace(mesh, "Hermite", 3)
    Vp_z = FunctionSpace(mesh, "Hermite", 3)

    np_y = Vp_y.dim()
    np_z = Vp_z.dim()

    vp_y = TestFunction(Vp_y)
    vp_z = TestFunction(Vp_z)

    ep_y = TrialFunction(Vp_y)
    ep_z = TrialFunction(Vp_z)

    dofs2dump_y = list([0, 1])
    dofs2keep_y = list(set(range(np_y)).difference(set(dofs2dump_y)))

    dofs2dump_z = list([0, 1])
    dofs2keep_z = list(set(range(np_z)).difference(set(dofs2dump_z)))

    npy_keep = len(dofs2keep_y)
    npz_keep = len(dofs2keep_z)

    np_keep = npy_keep + npz_keep

    dx = Measure('dx')

    Jf_10tz = assemble(vp_y * rho * A * dx).vector().get_local()[dofs2keep_y]
    Jf_10ry = assemble(-vp_y * rho * A * x[0] * dx).vector().get_local()[dofs2keep_y]

    jf_10fz = vp_y * rho * A * ep_z * dx

    petsc_jf_10fz = assemble(jf_10fz, mat_type='aij').M.handle

    Jf_10fz = np.array(petsc_jf_10fz.convert("dense").getDenseArray())

    Jf_10fz = Jf_10fz[dofs2keep_y, :]
    Jf_10fz = Jf_10fz[:, dofs2keep_z]

    Jf_tz = np.zeros((np_keep, 3))
    Jf_tz[:npy_keep, 0] = Jf_10tz

    Jf_ry = np.zeros((np_keep, 3))
    Jf_ry[:npy_keep, 0] = Jf_10ry

    Jf_fz = np.zeros((np_keep, 3, npz_keep))
    Jf_fz[:+npy_keep, 0, :] = Jf_10fz

    Jf_20ty = assemble(vp_z * rho * A * dx).vector().get_local()[dofs2keep_z]
    Jf_20rz = assemble(vp_z * rho * A * x[0] * dx).vector().get_local()[dofs2keep_z]

    jf_20fy = vp_z * rho * A * ep_y * dx

    petsc_jf_20fy = assemble(jf_20fy, mat_type='aij').M.handle

    Jf_20fy = np.array(petsc_jf_20fy.convert("dense").getDenseArray())

    Jf_20fy = Jf_20fy[dofs2keep_z, :]
    Jf_20fy = Jf_20fy[:, dofs2keep_y]

    Jf_ty = np.zeros((np_keep, 3))
    Jf_ty[npy_keep:, 0] = -Jf_20ty

    Jf_rz = np.zeros((np_keep, 3))
    Jf_rz[npy_keep:, 0] = -Jf_20rz

    Jf_fy = np.zeros((np_keep, 3, npy_keep))
    Jf_fy[npy_keep:, 0, :] = -Jf_20fy


    return Jf_ty, Jf_tz, Jf_ry, Jf_rz, Jf_fy, Jf_fz


def massmatrices_j3d(n_el, L, rho, A):

    mesh = IntervalMesh(n_el, L)
    x = SpatialCoordinate(mesh)

    # Finite element defition
    Vp_x = FunctionSpace(mesh, "Lagrange", 1)
    Vp_y = FunctionSpace(mesh, "Hermite", 3)
    Vp_z = FunctionSpace(mesh, "Hermite", 3)

    V = Vp_x * Vp_y * Vp_z
    n_p = V.dim()
    np_x = Vp_x.dim()
    np_y = Vp_y.dim()
    np_z = Vp_z.dim()

    v = TestFunction(V)
    vp_x, vp_y, vp_z = split(v)

    e = TrialFunction(V)
    ep_x, ep_y, ep_z = split(e)

    dofs2dump = list([0, np_x, np_x+1, np_x+np_y, np_x+np_y+1])
    dofs2keep = list(set(range(n_p)).difference(set(dofs2dump)))

    np_keep = len(dofs2keep)

    dx = Measure('dx')

    int_vx = assemble(vp_x * rho * A * dx).vector().get_local()[dofs2keep]
    int_vy = assemble(vp_y * rho * A * dx).vector().get_local()[dofs2keep]
    int_vz = assemble(vp_z * rho * A * dx).vector().get_local()[dofs2keep]

    s_u = np.zeros((3, 3, np_keep))

    s_u[0, 1, :] = -int_vz
    s_u[1, 0, :] = +int_vz

    s_u[0, 2, :] = +int_vy
    s_u[2, 0, :] = -int_vy

    s_u[1, 2, :] = -int_vx
    s_u[1, 2, :] = +int_vx

    intx_vx = assemble(vp_x * rho * A * x[0] * dx).vector().get_local()[dofs2keep]
    intx_vy = assemble(vp_y * rho * A * x[0] * dx).vector().get_local()[dofs2keep]
    intx_vz = assemble(vp_z * rho * A * x[0] * dx).vector().get_local()[dofs2keep]

    J_xu = np.zeros((3, 3, np_keep))

    J_xu[0, 1, :] = - intx_vy
    J_xu[1, 0, :] = - intx_vy

    J_xu[0, 2, :] = - intx_vz
    J_xu[2, 0, :] = - intx_vz

    J_xu[1, 1, :] = 2 * intx_vx
    J_xu[2, 2, :] = 2 * intx_vx

    j_ex2 = assemble(vp_x * rho * A * ep_x * dx, mat_type='aij').M.handle
    J_ex2 = np.array(j_ex2.convert("dense").getDenseArray())
    J_ex2 = J_ex2[dofs2keep, :]
    J_ex2 = J_ex2[:, dofs2keep]

    j_ey2 = assemble(vp_y * rho * A * ep_y * dx, mat_type='aij').M.handle
    J_ey2 = np.array(j_ex2.convert("dense").getDenseArray())
    J_ey2 = J_ey2[dofs2keep, :]
    J_ey2 = J_ey2[:, dofs2keep]

    j_ez2 = assemble(vp_z * rho * A * ep_z * dx, mat_type='aij').M.handle
    J_ez2 = np.array(j_ex2.convert("dense").getDenseArray())
    J_ez2 = J_ez2[dofs2keep, :]
    J_ez2 = J_ez2[:, dofs2keep]

    j_exy = assemble(vp_x * rho * A * ep_y * dx, mat_type='aij').M.handle
    J_exy = np.array(j_exy.convert("dense").getDenseArray())
    J_exy = J_exy[dofs2keep, :]
    J_exy = J_exy[:, dofs2keep]

    j_exz = assemble(vp_x * rho * A * ep_z * dx, mat_type='aij').M.handle
    J_exz = np.array(j_exz.convert("dense").getDenseArray())
    J_exz = J_exz[dofs2keep, :]
    J_exz = J_exz[:, dofs2keep]

    j_eyz = assemble(vp_y * rho * A * ep_z * dx, mat_type='aij').M.handle
    J_eyz = np.array(j_eyz.convert("dense").getDenseArray())
    J_eyz = J_eyz[dofs2keep, :]
    J_eyz = J_eyz[:, dofs2keep]

    J_uu = np.zeros((3, 3, np_keep, np_keep))

    J_uu[0, 0, :] = J_ey2 + J_ez2
    J_uu[1, 1, :] = J_ex2 + J_ez2
    J_uu[2, 2, :] = J_ex2 + J_ey2

    J_uu[0, 1, :] = - J_exy
    J_uu[1, 0, :] = - J_exy

    J_uu[0, 2, :] = - J_exz
    J_uu[2, 0, :] = - J_exz

    J_uu[1, 2, :] = - J_eyz
    J_uu[2, 1, :] = - J_eyz

    return s_u, J_xu, J_uu
