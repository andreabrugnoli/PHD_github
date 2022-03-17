from firedrake import *
from time import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def explicit_step_primal(dt_0, problem, x_n, V_vel, V_vor, param={"ksp_type": "preonly", "pc_type": "lu"}):
    v_n = x_n[0]
    w_n = x_n[1]
    p_n = x_n[2]

    chi_1 = TestFunction(V_vel)
    u_1 = TrialFunction(V_vel)

    a_form_vel = (1 / dt_0) * m_form(chi_1, u_1)
    A_vel = assemble(a_form_vel, mat_type='aij')

    b_form_vel = (1 / dt_0) * m_form(chi_1, v_n) + wcross1_form(chi_1, v_n, w_n, problem.dimM) \
                 + gradp_form(chi_1, p_n) + adj_curlw_form(chi_1, w_n, problem.dimM, problem.Re)
    b_vel = assemble(b_form_vel)

    v_sol = Function(V_vel)
    solve(A_vel, v_sol, b_vel, solver_parameters=param)

    chi_w = TestFunction(V_vor)
    w_trial = TrialFunction(V_vor)

    a_form_vor = m_form(chi_w, w_trial)
    A_vor = assemble(a_form_vor)

    b_form_vor = curlu_form(chi_w, v_sol, problem.dimM)
    b_vor = assemble(b_form_vor)

    w_sol = Function(V_vor)

    solve(A_vor, w_sol, b_vor, solver_parameters=param)

    return v_sol, w_sol

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
    tvec_stag = np.linspace(float(dt)/2, float(dt)*(n_t + 1/2), n_t+1)

    u_pr_half, w_pr_half = explicit_step_primal(dt / 2, problem, x_pr_0, V_1, V_2)

    print("Explicit step solved")

    # u_pr_12 = interpolate(u_pr_half, V_1)
    # w_pr_12 = interpolate(w_pr_half, V_2)
    # p_pr_init = interpolate(p_pr_0, V_0)

    # Primal intermediate variables
    xprimal_n12 = Function(V_primal, name="u, w at n+1/2, p at n")
    xprimal_n12.sub(0).assign(u_pr_half)
    xprimal_n12.sub(1).assign(w_pr_half)
    xprimal_n12.sub(2).assign(p_pr_0)

    xprimal_n32 = Function(V_primal, name="u, w at n+3/2, p at n+1")

    # xprimal_n1 = Function(V_primal, name="u, w at n+1, p at n+1/2")


    # Dual intermediate variables
    xdual_n = Function(V_dual, name="uT, wT at n, pT at n-1/2")
    xdual_n.assign(xdual_0)

    xdual_n1 = Function(V_dual, name="u, w at n+1, p at n+1/2")

    # Kinetic energy definition
    # Primal
    H_pr_vec = np.zeros((n_t + 1))
    H_pr_12 = 0.5*dot(u_pr_half, u_pr_half) * dx
    H_pr_vec[0] = assemble(H_pr_12)

    # Dual
    u_dl_0, w_dl_0, p_dl_0 = xdual_0.split()
    H_dl_vec = np.zeros((n_t + 1))
    H_dl_0 = 0.5*dot(u_dl_0, u_dl_0) * dx
    H_dl_vec[0] = assemble(H_dl_0)

    # Compute vorticity at a given point to check correctness of the solver
    w_pr_P_vec = np.zeros((n_t + 1))
    w_dl_P_vec = np.zeros((n_t + 1))
    if problem.dimM == 2:
        point_P = (1/3, 5/7)
        # Primal
        w_pr_P_vec[0] = w_pr_half.at(point_P)
        # Dual
        w_dl_P_vec[0] = w_dl_0.at(point_P)
    else:
        point_P = (1 / 3, 5 / 7, 3/7)


    # Exact quantities
    # Energy and Vorticity at P
    H_ex_vec = np.zeros((n_t + 1))
    w_ex_P_vec = np.zeros((n_t + 1))

    if problem.exact == True:
        u_ex_0, w_ex_0, p_ex_0, H_ex_0, E_ex_0, Ch_ex_0 = problem.init_outputs(0)
        H_ex_vec[0] = assemble(H_ex_0)
        w_ex_P_vec[0] = w_ex_0(point_P)

    # Primal Test and trial functions definition
    chi_primal = TestFunction(V_primal)
    chi_u_pr, chi_w_pr, chi_p_pr = split(chi_primal)

    x_primal = TrialFunction(V_primal)
    u_pr, w_pr, p_pr = split(x_primal)

    # Static part of the primal A operator
    a1_primal_static = (1/dt) * m_form(chi_u_pr, u_pr) - gradp_form(chi_u_pr, p_pr) \
                       - 0.5*adj_curlw_form(chi_u_pr, w_pr, problem.dimM, problem.Re)
    a2_primal_static = m_form(chi_w_pr, w_pr) - curlu_form(chi_w_pr, u_pr, problem.dimM)
    a3_primal_static = - adj_divu_form(chi_p_pr, u_pr)

    # A_primal_static = assemble(a1_primal_static + a2_primal_static + a3_primal_static, mat_type='aij')

    # Primal Test and trial functions definition
    chi_dual = TestFunction(V_dual)
    chi_u_dl, chi_w_dl, chi_p_dl = split(chi_dual)

    x_dual = TrialFunction(V_dual)
    u_dl, w_dl, p_dl = split(x_dual)

    # Static part of the dual A operator
    a1_dual_static = (1/dt) * m_form(chi_u_dl, u_dl) - adj_gradp_form(chi_u_dl, p_dl) \
                       - 0.5 * curlw_form(chi_u_dl, w_dl, problem.dimM, problem.Re)
    a2_dual_static = m_form(chi_w_dl, w_dl) - adj_curlu_form(chi_w_dl, u_dl, problem.dimM)
    a3_dual_static = - divu_form(chi_p_dl, u_dl)

    # A_dual_static = assemble(a1_dual_static + a2_dual_static + a3_dual_static, mat_type='aij')

    param_solver_saddlepoint = {"ksp_type": "gmres", "ksp_gmres_restart": 100, \
             "pc_type": "hypre", 'pc_hypre_type': 'boomeramg'}

    # Time loop from 1 onwards
    for ii in tqdm(range(1, n_t+1)):

        # Solve dual system for n+1
        u_pr_n12, w_pr_n12, p_pr_n12 = xprimal_n12.split()
        a_dual_dynamic = - 0.5*wcross2_form(chi_u_dl, u_dl, w_pr_n12, problem.dimM)
        A_dual = assemble(a1_dual_static + a2_dual_static + a3_dual_static + a_dual_dynamic, mat_type='aij')
        # A_dual_dynamic = assemble(a_dual_dynamic, mat_type='aij')
        # A_dual = A_dual_static + A_dual_dynamic

        u_dl_n, w_dl_n, p_dl_12n = xdual_n.split()

        b1_dual = (1/dt) * m_form(chi_u_dl, u_dl_n) + 0.5*wcross2_form(chi_u_dl, u_dl_n, w_pr_n12, problem.dimM) \
                  + 0.5*curlw_form(chi_u_dl, w_dl_n, problem.dimM, problem.Re)
        bvec_dual = assemble(b1_dual)
        solve(A_dual, xdual_n1.vector(), bvec_dual, solver_parameters=param_solver_saddlepoint)

        u_dl_n1, w_dl_n1, p_dl_n12 = xdual_n1.split()
        H_dl_n1 = 0.5 * dot(u_dl_n1, u_dl_n1) * dx
        H_dl_vec[ii] = assemble(H_dl_n1)

        # Solve primal system at n_32
        a_primal_dynamic = - 0.5*wcross1_form(chi_u_pr, u_pr, w_dl_n1, problem.dimM)
        A_primal = assemble(a1_primal_static + a2_primal_static + a3_primal_static + a_primal_dynamic, mat_type='aij')
        # A_primal_dynamic = assemble(a_primal_dynamic)
        # A_primal = A_primal_static + A_primal_dynamic

        u_pr_n12, w_pr_n12, p_pr_n12 = xprimal_n12.split()
        b1_primal = (1/dt) * m_form(chi_u_pr, u_pr_n12) + 0.5*wcross1_form(chi_u_pr, u_pr_n12, w_dl_n1, problem.dimM) \
                    + 0.5*adj_curlw_form(chi_u_pr, w_pr_n12, problem.dimM, problem.Re)
        bvec_primal = assemble(b1_primal)
        solve(A_primal, xprimal_n32.vector(), bvec_primal, solver_parameters=param_solver_saddlepoint)

        # xprimal_n1.assign(0.5*(xprimal_n12 + xprimal_n32))
        # u_pr_n1, w_pr_n1, p_pr_n1 = xprimal_n1.split(deepcopy=True)
        # H_pr_n1 = 0.5 * dot(u_pr_n1, u_pr_n1) * dx

        u_pr_n32, w_pr_n32, p_pr_n1 = xprimal_n32.split()
        H_pr_n32 = 0.5 * dot(u_pr_n32, u_pr_n32) * dx
        H_pr_vec[ii] = assemble(H_pr_n32)

        xdual_n.assign(xdual_n1)
        xprimal_n12.assign(xprimal_n32)

        if problem.dimM==2:
            w_dl_P_vec[ii] = w_dl_n1(point_P)
            w_pr_P_vec[ii] = w_pr_n32(point_P)

        # Reassign dual, primal, exact

    # Compute exact energy and vorticity
    if problem.exact == True:
        for ii in tqdm(range(1, n_t + 1)):
            t_act = ii * dt
            u_ex_t, w_ex_t, p_ex_t, H_ex_t, E_ex_t, Ch_ex_t = problem.init_outputs(t_act)
            H_ex_vec[ii] = assemble(H_ex_t)
            w_ex_P_vec[ii] = w_ex_t(point_P)

    return tvec_int, tvec_stag,  H_pr_vec, H_dl_vec, H_ex_vec, w_pr_P_vec, w_dl_P_vec, w_ex_P_vec

# Common forms
def m_form(chi_i, alpha_i):
    form = inner(chi_i,alpha_i) * dx
    return form

def curl2D(v):
    return v[1].dx(0) - v[0].dx(1)

def rot2D(w):
    return as_vector((w.dx(1), -w.dx(0)))


# Primal system forms
def wcross1_form(chi_1, v_1, wT_n2, dimM):
    if dimM==3:
        form = inner(chi_1,cross(v_1, wT_n2)) * dx
    elif dimM==2:
        form = dot(wT_n2, v_1[1]*chi_1[0] - v_1[0]*chi_1[1]) * dx
    return form

def gradp_form(chi_1, p_0):
    form = -inner(chi_1,grad(p_0)) * dx
    return form

def adj_curlw_form(chi_1, w_2, dimM, Re):
    # if dimM==3:
    #     form = -1./Re*inner(curl(chi_1),w_2) * dx
    # elif dimM==2:
    #     form = -1./Re*dot(curl2D(chi_1),w_2) * dx
    # return form
    return 0

def adj_divu_form(chi_0, v_1):
    form = inner(grad(chi_0),v_1) * dx
    return form

def curlu_form(chi_2, v_1, dimM):
    if dimM==3:
        form = inner(chi_2,curl(v_1)) * dx
    elif dimM==2:
        form = dot(chi_2,curl2D(v_1)) * dx
    return form

def tantrace_w_form(chi_1, wT_n2, n_vec, dimM, Re):
    if dimM==3:
        form = 1./Re*dot(cross(chi_1,wT_n2),n_vec) * ds
    elif dimM==2:
        form = 1./Re*wT_n2*dot(as_vector((chi_1[1], -chi_1[0])), n_vec) * ds
    return form

def normtrace_v_form(chi_0, vT_n1, n_vec):
    form = -chi_0*dot(vT_n1,n_vec) * ds
    return form

# Dual system weak forms
def wcross2_form(chi_2, vT_2, w_2, dimM):
    if dimM==3:
        form = inner(chi_2,cross(vT_2, w_2)) * dx
    elif dimM==2:
        form = dot(w_2, vT_2[1]*chi_2[0] - vT_2[0]*chi_2[1]) * dx
    return form

def adj_gradp_form(chi_2,pT_3):
    form = inner(div(chi_2),pT_3) * dx
    return form

def curlw_form(chi_2,wT_1,dimM, Re):
    # if dimM == 3:
    #     form = -1./Re*inner(chi_2, curl(wT_1)) * dx
    # elif dimM == 2:
    #     form = -1./Re*dot(chi_2, rot2D(wT_1)) * dx
    #     # 2D Curl i.e. rotated grad:  // ux = u.dx(0) // uy = u.dx(1) // as_vector((uy, -ux))
    # return form
    return 0

def divu_form(chi_3, vT_2):
    form = -inner(chi_3, div(vT_2)) * dx
    return form

def adj_curlu_form(chi_1, vT_2, dimM):
    if dimM == 3:
        form = inner(curl(chi_1), vT_2) * dx
    elif dimM == 2:
        form = dot(rot2D(chi_1), vT_2) * dx
    return form

def dirtrace_p_form(chi_2, p_0, n_vec):
    form = -p_0*dot(chi_2,n_vec) * ds
    return form

def tantrace_v_form(chi_1, v_1, n_vec, dimM):
    if dimM == 3:
        form = -dot(cross(chi_1, v_1), n_vec) * ds
    elif dimM == 2:
        form = chi_1*dot(as_vector((v_1[1], -v_1[0])), n_vec) * ds
    return form
