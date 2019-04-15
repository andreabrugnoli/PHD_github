# EB beam written with the port Hamiltonian approach

from firedrake import *
import numpy as np
import scipy.linalg as la
from module_ph_systems import SystemPHDAE


class FloatingEB(SystemPHDAE):

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
        n_V = n_Vp + n_Vq

        v = TestFunction(V)
        v_p, v_q = split(v)

        e = TrialFunction(V)
        e_p, e_q = split(e)

        al_p = rho * e_p
        al_q = 1./EI * e_q

        dx = Measure('dx')
        ds = Measure('ds')

        m = v_p * al_p * dx + v_q * al_q * dx

        petsc_m = assemble(m, mat_type='aij').M.handle
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

        j = j_gradgrad + j_gradgradIP

        petcs_j = assemble(j, mat_type='aij').M.handle
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

        SystemPHDAE.__init__(self, n_tot, E=M, J=J, B=B)
