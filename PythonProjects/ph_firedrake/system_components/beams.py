# EB beam written with the port Hamiltonian approach

from firedrake import *
import numpy as np
import scipy.linalg as la
from modules_phdae.classes_phsystem import SysPhdaeRig, check_positive_matrix
import warnings
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

        n_rig = 3  # Planar motion
        n_fl = n_V - 2
        n_tot = n_rig + n_fl
        M_f = M_FEM[-n_fl:, -n_fl:]

        M_fr = np.zeros((n_fl, n_rig))
        M_fr[:, 1] = assemble(v_p * rho * A * dx).vector().get_local()[2:]
        M_fr[:, 2] = assemble(v_p * rho * A * x[0] * dx).vector().get_local()[2:]

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

        assert check_positive_matrix(M)

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


class FloatFlexBeam(SysPhdaeRig):

    def __init__(self, n_el, L, rho, A, E, I, m_joint=0.0, J_joint=0.0):

        mesh = IntervalMesh(n_el, L)
        x = SpatialCoordinate(mesh)

        # Finite element defition
        Vp_x = FunctionSpace(mesh, "Lagrange", 1)
        Vq_x = FunctionSpace(mesh, "Lagrange", 1)

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
        n_fl = n_V - n_rig
        n_p = n_p - n_rig
        n_tot = n_rig + n_fl

        M_f = M_FEM
        M_f = M_f[:, dofs2keep]
        M_f = M_f[dofs2keep, :]

        assert len(M_f) == n_fl

        M_fr = np.zeros((n_fl, n_rig))
        M_fr[:, 0] = assemble(vp_x * rho * A * dx).vector().get_local()[dofs2keep]
        M_fr[:, 1] = assemble(vp_y * rho * A * dx).vector().get_local()[dofs2keep]
        M_fr[:, 2] = assemble(vp_y * rho * A * x[0] * dx).vector().get_local()[dofs2keep]

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


def draw_deformation(n_draw, v_rig, v_fl, L):
    # Suppose no displacement in zero
    assert len(v_rig)  == 3
    assert len(v_fl) % 3 == 0
    n_el = int(len(v_fl)/3)

    ufl_dofs = v_fl[:n_el]
    wfl_dofs = v_fl[n_el:]

    dx_el = L/n_el

    u_P = v_rig[0]
    w_P = v_rig[1]
    th_P = v_rig[2]

    ndr_el = int(n_draw/n_el)
    n_draw = ndr_el*n_el
    x_coord = np.linspace(0, L, num=n_draw+1)

    u_r = u_P*np.ones_like(x_coord)
    w_r = w_P*np.ones_like(x_coord) + x_coord*th_P

    u_fl = np.zeros_like(x_coord)
    w_fl = np.zeros_like(x_coord)

    for i in range(n_el):
            for j in range(ndr_el):
                ind = i*ndr_el+j+1
                i_el = i*ndr_el
                x_til = (x_coord[ind] - x_coord[i_el])/dx_el
                phi1_u = (1 - x_til)
                phi2_u = x_til

                phi1_w = 1 - 3 * x_til**2 + 2 * x_til**3
                phi2_w = x_til - 2 * x_til**2 + x_til**3
                phi3_w = + 3 * x_til ** 2 -2 * x_til ** 3
                phi4_w = + x_til**3 -x_til**2

                if i == 0:
                    u_fl[ind] = phi2_u*ufl_dofs[i]
                    w_fl[ind] = phi3_w*wfl_dofs[2*i] + phi4_w*wfl_dofs[2*i+1]
                else:
                    u_fl[ind] = phi1_u * ufl_dofs[i-1] + phi2_u * ufl_dofs[i]
                    w_fl[ind] = phi1_w*wfl_dofs[2*(i-1)] + phi2_w*wfl_dofs[2*i-1] + \
                                   phi3_w*wfl_dofs[2*i] + phi4_w*wfl_dofs[2*i+1]

    u_tot = u_r + u_fl
    w_tot = w_r + w_fl

    return x_coord, u_tot, w_tot


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

    ndr_el = int(n_draw / n_el)
    n_draw = ndr_el * n_el
    x_coord = np.linspace(0, L, num=n_draw + 1)

    u_r = u_P * np.ones_like(x_coord)
    w_r = w_P * np.ones_like(x_coord) + x_coord * th_P

    u_fl = np.zeros_like(x_coord)
    w_fl = np.zeros_like(x_coord)

    for i in range(n_el):
        for j in range(ndr_el):
            ind = i * ndr_el + j + 1
            i_el = i * ndr_el
            x_til = (x_coord[ind] - x_coord[i_el]) / dx_el

            phi1_w = 1 - 3 * x_til ** 2 + 2 * x_til ** 3
            phi2_w = x_til - 2 * x_til ** 2 + x_til ** 3
            phi3_w = + 3 * x_til ** 2 - 2 * x_til ** 3
            phi4_w = + x_til ** 3 - x_til ** 2

            if i == 0:
                w_fl[ind] = phi3_w * wfl_dofs[2 * i] + phi4_w * wfl_dofs[2 * i + 1]
            else:
                w_fl[ind] = phi1_w * wfl_dofs[2 * (i - 1)] + phi2_w * wfl_dofs[2 * i - 1] + \
                            phi3_w * wfl_dofs[2 * i] + phi4_w * wfl_dofs[2 * i + 1]

    u_tot = u_r
    w_tot = w_r + w_fl

    return x_coord, u_tot, w_tot