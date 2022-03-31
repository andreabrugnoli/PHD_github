from firedrake import *
from .utilities.operators import *
from time import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

solver_param = {"mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
# solver_param = {}

def explicit_step_primal(dt_0, problem, x_n, wT_n, V, system="primal"):
    u_n = x_n[0]
    w_n = x_n[1]
    p_n = x_n[2]

    chi_pr = TestFunction(V)
    chi_u, chi_w, chi_p = split(chi_pr)

    x_pr = TrialFunction(V)
    u, w, p = split(x_pr)

    if system == "primal":

        a1_form_vel = (1 / dt_0) * m_form(chi_u, u) - gradp_form(chi_u, p) \
                      - 0.5 * wcross1_form(chi_u, u, wT_n, problem.dimM)
        a2_form_vor = m_form(chi_w, w) - curlu_form(chi_w, u, problem.dimM)
        a3_form_p = - adj_divu_form(chi_p, u)

        b1_form = (1 / dt_0) * m_form(chi_u, u_n) + 0.5 * wcross1_form(chi_u, u_n, wT_n, problem.dimM)
    else:

        a1_form_vel = (1 / dt_0) * m_form(chi_u, u) - adj_gradp_form(chi_u, p) \
                      - 0.5 * wcross2_form(chi_u, u, wT_n, problem.dimM)
        a2_form_vor = m_form(chi_w, w) - adj_curlu_form(chi_w, u, problem.dimM)
        a3_form_p = - divu_form(chi_p, u)

        b1_form = (1 / dt_0) * m_form(chi_u, u_n) + 0.5 * wcross2_form(chi_u, u_n, wT_n, problem.dimM)

    V_nullspace = MixedVectorSpaceBasis(V, [V.sub(0), V.sub(1), VectorSpaceBasis(constant=True)])

    a_form = a1_form_vel + a2_form_vor + a3_form_p
    A0 = assemble(a_form, mat_type='aij')
    b0 = assemble(b1_form)

    x_sol = Function(V)
    solve(A0, x_sol, b0, nullspace=V_nullspace, solver_parameters=solver_param)
    # solve(A0, x_sol, b0, solver_parameters=solver_param)



    return x_sol


def compute_sol(problem, pol_deg, n_t, t_fin=1):
    # Implementation of the dual field formulation for periodic navier stokes
    mesh = problem.mesh
    problem.init_mesh()

    # Primal trimmed polynomial finite element families
    if problem.quad == False:
        V_1 = FunctionSpace(mesh, "N1curl", pol_deg)
        V_0 = FunctionSpace(mesh, "CG", pol_deg)
        if problem.dimM == 3:
            V_2 = FunctionSpace(mesh, "RT", pol_deg)
        elif problem.dimM == 2:
            V_2 = FunctionSpace(mesh, "DG", pol_deg - 1)
        # Dual trimmed polynomial finite element families
        VT_n1 = FunctionSpace(mesh, "RT", pol_deg)
        VT_n = FunctionSpace(mesh, "DG", pol_deg - 1)
        if problem.dimM == 3:
            VT_n2 = FunctionSpace(mesh, "N1curl", pol_deg)
        elif problem.dimM == 2:
            VT_n2 = FunctionSpace(mesh, "CG", pol_deg)
    else:
        V_0 = FunctionSpace(mesh, "CG", pol_deg)
        if problem.dimM == 3:
            V_1 = FunctionSpace(mesh, "NCE", pol_deg)
            V_2 = FunctionSpace(mesh, "NCF", pol_deg)
        elif problem.dimM == 2:
            V_1 = FunctionSpace(mesh, "RTCE", pol_deg)
            V_2 = FunctionSpace(mesh, "DG", pol_deg - 1)
        # Dual trimmed polynomial finite element families
        VT_n = FunctionSpace(mesh, "DG", pol_deg - 1)
        if problem.dimM == 3:
            VT_n1 = FunctionSpace(mesh, "NCF", pol_deg)
            VT_n2 = FunctionSpace(mesh, "NCE", pol_deg)
        elif problem.dimM == 2:
            VT_n2 = FunctionSpace(mesh, "CG", pol_deg)
            VT_n1 = FunctionSpace(mesh, "RTCF", pol_deg)

    V_primal = V_1 * V_2 * V_0
    V_dual = VT_n1 * VT_n2 * VT_n

    print("Function Space dimensions, Primal - Dual: ", [V_primal.dim(), V_dual.dim()])

    # Define Function assigners
    # Set initial condition at t=0
    xprimal_0 = Function(V_primal, name="x_0 primal")
    xdual_0 = Function(V_dual, name="x_0 dual")

    x_pr_0 = problem.initial_conditions(V_1, V_2, V_0)
    u_pr_0 = x_pr_0[0]
    w_pr_0 = x_pr_0[1]
    p_pr_0 = x_pr_0[2]

    xprimal_0.sub(0).assign(u_pr_0)
    xprimal_0.sub(1).assign(w_pr_0)
    xprimal_0.sub(2).assign(p_pr_0)

    x_dl_0 = problem.initial_conditions(VT_n1, VT_n2, VT_n)
    u_dl_0 = x_dl_0[0]
    w_dl_0 = x_dl_0[1]
    p_dl_0 = x_dl_0[2]

    xdual_0.sub(0).assign(u_dl_0)
    xdual_0.sub(1).assign(w_dl_0)
    xdual_0.sub(2).assign(p_dl_0)

    dt = Constant(t_fin / n_t)
    tvec_int = np.linspace(0, n_t * float(dt), 1 + n_t)
    tvec_stag = np.zeros((n_t + 1))
    tvec_stag[1:] = np.linspace(float(dt) / 2, float(dt) * (n_t - 1 / 2), n_t)

    xprimal_n12 = explicit_step_primal(dt / 2, problem, x_pr_0, w_dl_0, V_primal)
    # u_pr_12, w_pr_12, p_pr_init = xprimal_n12.split()

    print("Explicit step solved")

    # Primal intermediate variables
    xprimal_n32 = Function(V_primal, name="u, w at n+3/2, p at n+1")

    xprimal_n1 = Function(V_primal, name="u, w at n+1, p at n+1/2")

    # Dual intermediate variables
    xdual_n = Function(V_dual, name="uT, wT at n, pT at n-1/2")
    xdual_n.assign(xdual_0)

    xdual_n1 = Function(V_dual, name="u, w at n+1, p at n+1/2")

    # Kinetic energy and Enstrophy definition
    # Primal
    H_pr_vec = np.zeros((n_t + 1,))
    H_pr_0 = 0.5 * dot(u_pr_0, u_pr_0) * dx
    H_pr_vec[0] = assemble(H_pr_0)

    E_pr_vec = np.zeros((n_t + 1,))
    E_pr_0 = 0.5 * dot(w_pr_0, w_pr_0) * dx
    E_pr_vec[0] = assemble(E_pr_0)

    # Dual
    u_dl_0, w_dl_0, p_dl_0 = xdual_0.split()

    H_dl_vec = np.zeros((n_t + 1,))
    H_dl_0 = 0.5 * dot(u_dl_0, u_dl_0) * dx
    H_dl_vec[0] = assemble(H_dl_0)

    E_dl_vec = np.zeros((n_t + 1,))
    E_dl_0 = 0.5 * dot(w_dl_0, w_dl_0) * dx
    E_dl_vec[0] = assemble(E_dl_0)

    # Incompressibility constraint
    div_u_pr_L2vec = np.zeros((n_t + 1,))
    div_u_dl_L2vec = np.zeros((n_t + 1,))

    divu_pr_0 = div(u_pr_0) ** 2 * dx
    divu_dl_0 = div(u_dl_0) ** 2 * dx

    div_u_pr_L2vec[0] = np.sqrt(assemble(divu_pr_0))
    div_u_dl_L2vec[0] = np.sqrt(assemble(divu_dl_0))

    # Compute vorticity at a given point to check correctness of the solver
    u_pr_P_vec = np.zeros((n_t + 1, problem.dimM))
    u_dl_P_vec = np.zeros((n_t + 1, problem.dimM))

    if problem.dimM == 2:
        w_pr_P_vec = np.zeros((n_t + 1, 1))
        w_dl_P_vec = np.zeros((n_t + 1, 1))
    elif problem.dimM == 3:
        w_pr_P_vec = np.zeros((n_t + 1, problem.dimM))
        w_dl_P_vec = np.zeros((n_t + 1, problem.dimM))

    pdyn_pr_P_vec = np.zeros((n_t + 1,))
    pdyn_dl_P_vec = np.zeros((n_t + 1,))

    p_pr_P_vec = np.zeros((n_t + 1,))
    p_dl_P_vec = np.zeros((n_t + 1,))

    # Only in 3D Helicity
    Hel_pr_vec = np.zeros((n_t + 1,))
    Hel_dl_vec = np.zeros((n_t + 1,))

    if problem.dimM == 2:
        point_P = (1 / 5, 2/ 7)
    elif problem.dimM == 3:
        point_P = (1 / 3, 4 / 7, 3 / 7)

        Hel_pr_0 = dot(u_pr_0, w_dl_0) * dx
        Hel_dl_0 = dot(u_dl_0, w_pr_0) * dx

        Hel_pr_vec[0] = assemble(Hel_pr_0)
        Hel_dl_vec[0] = assemble(Hel_dl_0)

    # Primal
    u_pr_P_vec[0, :] = u_pr_0.at(point_P)
    w_pr_P_vec[0, :] = w_pr_0.at(point_P)
    p_pr_P_vec[0] = p_pr_0.at(point_P)
    pdyn_pr_P_vec[0] = p_pr_P_vec[0] + 0.5 * np.dot(u_pr_P_vec[0, :], u_pr_P_vec[0, :])
    # Dual
    u_dl_P_vec[0, :] = u_dl_0.at(point_P)
    w_dl_P_vec[0, :] = w_dl_0.at(point_P)
    p_dl_P_vec[0] = p_dl_0.at(point_P)
    pdyn_dl_P_vec[0] = p_dl_P_vec[0] + 0.5 * np.dot(u_dl_P_vec[0, :], u_dl_P_vec[0, :])

    # Exact quantities
    # Energy and Vorticity at P
    H_ex_vec = np.zeros((n_t + 1,))
    E_ex_vec = np.zeros((n_t + 1,))
    Hel_ex_vec = np.zeros((n_t + 1,))

    u_ex_P_vec = np.zeros((n_t + 1, problem.dimM))

    if problem.dimM == 2:
        w_ex_P_vec = np.zeros((n_t + 1, 1))
    elif problem.dimM == 3:
        w_ex_P_vec = np.zeros((n_t + 1, problem.dimM))

    pdyn_ex_P_vec = np.zeros((n_t + 1,))
    p_ex_P_vec = np.zeros((n_t + 1,))

    if problem.exact == True:
        u_ex_0, w_ex_0, p_ex_0, H_ex_0, E_ex_0, Hel_ex_0 = problem.init_outputs(0)

        H_ex_vec[0] = assemble(H_ex_0)
        E_ex_vec[0] = assemble(E_ex_0)
        if problem.dimM == 3:
            Hel_ex_vec[0] = assemble(Hel_ex_0)
            for jj in range(np.shape(w_ex_P_vec)[1]):
                w_ex_P_vec[0, jj] = w_ex_0[jj](point_P)
        else:
            w_ex_P_vec[0, :] = w_ex_0(point_P)

        for jj in range(np.shape(u_ex_P_vec)[1]):
            u_ex_P_vec[0, jj] = u_ex_0[jj](point_P)

        p_ex_P_vec[0] = p_ex_0(point_P)
        pdyn_ex_P_vec[0] = p_ex_P_vec[0] + 0.5 * np.dot(u_ex_P_vec[0, :], u_ex_P_vec[0, :])

    # Primal Test and trial functions definition
    chi_primal = TestFunction(V_primal)
    chi_u_pr, chi_w_pr, chi_p_pr = split(chi_primal)

    x_primal = TrialFunction(V_primal)
    u_pr, w_pr, p_pr = split(x_primal)

    # Static part of the primal A operator
    a1_primal_static = (1 / dt) * m_form(chi_u_pr, u_pr) - gradp_form(chi_u_pr, p_pr)
    a2_primal_static = m_form(chi_w_pr, w_pr) - curlu_form(chi_w_pr, u_pr, problem.dimM)
    a3_primal_static = - adj_divu_form(chi_p_pr, u_pr)

    # Primal Test and trial functions definition
    chi_dual = TestFunction(V_dual)
    chi_u_dl, chi_w_dl, chi_p_dl = split(chi_dual)

    x_dual = TrialFunction(V_dual)
    u_dl, w_dl, p_dl = split(x_dual)

    # Static part of the dual A operator
    a1_dual_static = (1 / dt) * m_form(chi_u_dl, u_dl) - adj_gradp_form(chi_u_dl, p_dl)
    a2_dual_static = m_form(chi_w_dl, w_dl) - adj_curlu_form(chi_w_dl, u_dl, problem.dimM)
    a3_dual_static = - divu_form(chi_p_dl, u_dl)

    Vprimal_nullspace = MixedVectorSpaceBasis(V_primal, [V_primal.sub(0), V_primal.sub(1), VectorSpaceBasis(constant=True)])
    Vdual_nullspace = MixedVectorSpaceBasis(V_dual, [V_dual.sub(0), V_dual.sub(1), VectorSpaceBasis(constant=True)])

    # Time loop from 1 onwards
    for ii in tqdm(range(1, n_t+1)):

        # Solve dual system for n+1
        u_pr_n12, w_pr_n12, p_pr_n12 = xprimal_n12.split()
        a_dual_dynamic = - 0.5*wcross2_form(chi_u_dl, u_dl, w_pr_n12, problem.dimM)
        A_dual = assemble(a1_dual_static + a2_dual_static + a3_dual_static + a_dual_dynamic, mat_type='aij')
        # A_dual_dynamic = assemble(a_dual_dynamic, mat_type='aij')
        # A_dual = A_dual_static + A_dual_dynamic

        u_dl_n, w_dl_n, p_dl_12n = xdual_n.split()

        b1_dual = (1/dt) * m_form(chi_u_dl, u_dl_n) + 0.5*wcross2_form(chi_u_dl, u_dl_n, w_pr_n12, problem.dimM)
        bvec_dual = assemble(b1_dual)
        # solve(A_dual, xdual_n1, bvec_dual, solver_parameters=solver_param)
        solve(A_dual, xdual_n1, bvec_dual, nullspace=Vdual_nullspace, solver_parameters=solver_param)


        u_dl_n1, w_dl_n1, p_dl_n12 = xdual_n1.split()
        H_dl_n1 = 0.5 * dot(u_dl_n1, u_dl_n1) * dx
        H_dl_vec[ii] = assemble(H_dl_n1)

        # Solve primal system at n_32
        a_primal_dynamic = - 0.5*wcross1_form(chi_u_pr, u_pr, w_dl_n1, problem.dimM)
        A_primal = assemble(a1_primal_static + a2_primal_static + a3_primal_static + a_primal_dynamic, mat_type='aij')
        # A_primal_dynamic = assemble(a_primal_dynamic)
        # A_primal = A_primal_static + A_primal_dynamic

        u_pr_n12, w_pr_n12, p_pr_n12 = xprimal_n12.split()
        b1_primal = (1/dt) * m_form(chi_u_pr, u_pr_n12) + 0.5*wcross1_form(chi_u_pr, u_pr_n12, w_dl_n1, problem.dimM)
        bvec_primal = assemble(b1_primal)
        solve(A_primal, xprimal_n32, bvec_primal, nullspace=Vprimal_nullspace, solver_parameters=solver_param)

        u_pr_n32, w_pr_n32, p_pr_n1 = xprimal_n32.split()
        # Use the implicit midpoint rule to find primal variables at integer

        xprimal_n1.assign(0.5 * (xprimal_n12 + xprimal_n32))
        u_pr_n1, w_pr_n1, p_pr_n12 = xprimal_n1.split()

        # Compute the Hamiltonian
        H_dl_n1 = 0.5 * dot(u_dl_n1, u_dl_n1) * dx
        H_dl_vec[ii] = assemble(H_dl_n1)

        H_pr_n1 = 0.5 * dot(u_pr_n1, u_pr_n1) * dx
        H_pr_vec[ii] = assemble(H_pr_n1)

        # The Enstrophy
        E_dl_n1 = 0.5 * dot(w_dl_n1, w_dl_n1) * dx
        E_dl_vec[ii] = assemble(E_dl_n1)

        E_pr_n1 = 0.5 * dot(w_pr_n1, w_pr_n1) * dx
        E_pr_vec[ii] = assemble(E_pr_n1)

        # The divergence constraint
        divu_pr_n1 = div(u_pr_n1) ** 2 * dx
        divu_dl_n1 = div(u_dl_n1) ** 2 * dx

        div_u_pr_L2vec[ii] = np.sqrt(assemble(divu_pr_n1))
        div_u_dl_L2vec[ii] = np.sqrt(assemble(divu_dl_n1))
        # If problem is 3D also the Helicity
        if problem.dimM == 3:
            Hel_pr_n1 = dot(u_pr_n1, w_dl_n1) * dx
            Hel_dl_n1 = dot(u_dl_n1, w_pr_n1) * dx

            Hel_pr_vec[ii] = assemble(Hel_pr_n1)
            Hel_dl_vec[ii] = assemble(Hel_dl_n1)

        # Compute solution at a given point to assess convergence
        u_pr_P_vec[ii, :] = u_pr_n1.at(point_P)
        w_pr_P_vec[ii, :] = w_pr_n1.at(point_P)
        pdyn_pr_P_vec[ii] = p_pr_n1.at(point_P)
        p_pr_P_vec[ii] = pdyn_pr_P_vec[ii] - 0.5*np.dot(u_pr_P_vec[ii, :],u_pr_P_vec[ii, :])


        u_dl_P_vec[ii, :] = u_dl_n1.at(point_P)
        w_dl_P_vec[ii, :] = w_dl_n1.at(point_P)
        pdyn_dl_P_vec[ii] = p_dl_n12.at(point_P)
        p_dl_P_vec[ii] = pdyn_dl_P_vec[ii] - 0.5*np.dot(0.5*(u_dl_P_vec[ii, :] + u_dl_P_vec[ii-1, :]),\
                                                         0.5*(u_dl_P_vec[ii, :] + u_dl_P_vec[ii-1, :]))

        # Reassign dual, primal for next iteration
        xdual_n.assign(xdual_n1)
        xprimal_n12.assign(xprimal_n32)

        # # Test for skew symmetry rotational form
        # print("Rotational term")
        # print(assemble(wcross2_form(w_pr_n12, u_dl_n1, w_pr_n12, problem.dimM)))

        # Compute exact energy and vorticity
    if problem.exact == True:
        for ii in tqdm(range(1, n_t + 1)):
            t_act = ii * dt
            u_ex_t, w_ex_t, p_ex_t, H_ex_t, E_ex_t, Hel_ex_t = problem.init_outputs(t_act)
            H_ex_vec[ii] = assemble(H_ex_t)
            E_ex_vec[ii] = assemble(E_ex_t)
            if problem.dimM == 3:
                Hel_ex_vec[ii] = assemble(Hel_ex_t)
                for jj in range(np.shape(w_ex_P_vec)[1]):
                    w_ex_P_vec[ii, jj] = w_ex_t[jj](point_P)
            else:
                w_ex_P_vec[ii, :] = w_ex_t(point_P)

            for jj in range(np.shape(u_ex_P_vec)[1]):
                u_ex_P_vec[ii, jj] = u_ex_t[jj](point_P)
            p_ex_P_vec[ii] = p_ex_t(point_P)
            pdyn_ex_P_vec[ii] = p_ex_P_vec[ii] + 0.5 * np.dot(u_ex_P_vec[ii, :], u_ex_P_vec[ii, :])

    dict_res = {"tspan_int": tvec_int, "tspan_stag": tvec_stag, \
                "energy_ex": H_ex_vec, "energy_pr": H_pr_vec, "energy_dl": H_dl_vec, \
                "enstrophy_ex": E_ex_vec, "enstrophy_pr": E_pr_vec, "enstrophy_dl": E_dl_vec, \
                "helicity_ex": Hel_ex_vec, "helicity_pr": Hel_pr_vec, "helicity_dl": Hel_dl_vec, \
                "uP_ex": u_ex_P_vec, "uP_pr": u_pr_P_vec, "uP_dl": u_dl_P_vec, \
                "wP_ex": w_ex_P_vec, "wP_pr": w_pr_P_vec, "wP_dl": w_dl_P_vec, \
                "pdynP_ex": pdyn_ex_P_vec, "pdynP_pr": pdyn_pr_P_vec, "pdynP_dl": pdyn_dl_P_vec, \
                "pP_ex": p_ex_P_vec, "pP_pr": p_pr_P_vec, "pP_dl": p_dl_P_vec, \
                "divu_pr_L2": div_u_pr_L2vec, "divu_dl_L2": div_u_dl_L2vec}

    return dict_res

