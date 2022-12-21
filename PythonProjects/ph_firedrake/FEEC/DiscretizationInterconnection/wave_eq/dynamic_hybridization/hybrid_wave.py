import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "1"

from firedrake import *
from tqdm import tqdm
# from time import sleep
import matplotlib.pyplot as plt
from tools_plotting import setup
import pickle


from FEEC.DiscretizationInterconnection.wave_eq.exact_eigensolution import exact_sol_wave3D, exact_sol_wave2D
from FEEC.DiscretizationInterconnection.slate_syntax.solve_hybrid_system import solve_hybrid

from spaces_forms_hybridwave import spaces01, spaces32, \
    m_form01, m_form32, j_form01, j_form32, constr_loc01, constr_loc32, constr_global01, constr_global32, \
    assign_exact01, assign_exact32, \
    neumann_flow0, dirichlet_flow2,  \
    project_uex_W0nor, project_pex_W2nor

# from spaces_forms_postprocessingwave import spaces_postprocessing01, spaces_postprocessing32,\
#     assign_exact01_pp, assign_exact32_pp, neumann_flow0_pp, dirichlet_flow2_pp

from FEEC.DiscretizationInterconnection.dofs_bd_hybrid import dofs_ess_nat


def compute_err(n_el, n_t, deg=1, t_fin=1, bd_cond="D", dim="2D"):
    """Compute the numerical solution of the wave equation with a DG method based on interconnection

        Parameters:
        n_el: number of elements for the discretization
        n_t: number of time instants
        deg: polynomial degree for finite
        Returns:
        some plots
       """

    if dim=="2D":
        mesh = RectangleMesh(n_el, n_el, 1, 1)
    else:
        mesh = BoxMesh(n_el, n_el, n_el, 1, 1, 1)
    n_ver = FacetNormal(mesh)
    h_cell = CellDiameter(mesh)

    W01_loc, V0_tan, V01 = spaces01(mesh, deg)
    V_grad = W01_loc * V0_tan

    v_grad = TestFunction(V_grad)
    v0, v1, v0_nor, v0_tan = split(v_grad)

    e_grad = TrialFunction(V_grad)
    p0, u1, u0_nor, p0_tan = split(e_grad)

    print("Conforming Galerkin 01 dim: " + str(V01.dim()))
    print("Conforming Galerkin 01 (1 broken) dim: " + str(V01.sub(0).dim()+W01_loc.sub(1).dim()))
    print("Hybrid 01 dim: " + str(V0_tan.dim()))

    W32_loc, V2_tan, V32 = spaces32(mesh, deg)
    V_div = W32_loc * V2_tan

    print("Conforming Galerkin 32 dim: " + str(V32.dim()))
    print("Hybrid 32 dim: " + str(V2_tan.dim()))

    v_div = TestFunction(V_div)
    v3, v2, v2_nor, v2_tan = split(v_div)

    e_div = TrialFunction(V_div)
    p3, u2, p2_nor, u2_tan = split(e_div)

    # # Post-processing variables
    # W01_pp = spaces_postprocessing01(mesh, deg)
    #
    # v_grad_pp = TestFunction(W01_pp)
    # v0_pp, v1_pp = split(v_grad_pp)
    #
    # e_grad_pp = TrialFunction(W01_pp)
    # p0_pp, u1_pp = split(e_grad_pp)
    #
    # W32_pp = spaces_postprocessing32(mesh, deg)
    #
    # v_div_pp = TestFunction(W32_pp)
    # v3_pp, v2_pp = split(v_div_pp)
    #
    # e_div_pp = TrialFunction(W32_pp)
    # p3_pp, u2_pp = split(e_div_pp)

    dx = Measure('dx')
    ds = Measure('ds')
    dS = Measure('dS')

    dt = Constant(t_fin / n_t)

    t_vec = np.linspace(0, n_t * float(dt), 1 + n_t)

    t = Constant(0.0)
    t_1 = Constant(dt)

    if dim=="2D":
        p_ex, u_ex, p_ex_1, u_ex_1 = exact_sol_wave2D(mesh, t, t_1)
    else:
        p_ex, u_ex, p_ex_1, u_ex_1 = exact_sol_wave3D(mesh, t, t_1)
    u_ex_mid = 0.5 * (u_ex + u_ex_1)
    p_ex_mid = 0.5 * (p_ex + p_ex_1)

    if bd_cond == "D":
        # bc_D = [DirichletBC(V_grad.sub(3), p_ex_1, "on_boundary")]
        bc_D = [DirichletBC(V0_tan, p_ex_1, "on_boundary")]
        bc_N = []
    elif bd_cond == "N":
        bc_D = []
        bc_N = [DirichletBC(V2_tan, u_ex_1, "on_boundary")]

    else:
        # bc_D = [DirichletBC(V_grad.sub(3), p_ex_1, 1), \
        #         DirichletBC(V_grad.sub(3), p_ex_1, 3)]
        if dim=="2D":
            bc_D = [DirichletBC(V0_tan, p_ex_1, 1),
                    DirichletBC(V0_tan, p_ex_1, 3)]

            bc_N = [DirichletBC(V2_tan, u_ex_1, 2), \
                    DirichletBC(V2_tan, u_ex_1, 4)]
        else:
            bc_D = [DirichletBC(V0_tan, p_ex_1, 1),
                    DirichletBC(V0_tan, p_ex_1, 3),
                    DirichletBC(V0_tan, p_ex_1, 5)]

            bc_N = [DirichletBC(V2_tan, u_ex_1, 2), \
                    DirichletBC(V2_tan, u_ex_1, 4),
                    DirichletBC(V2_tan, u_ex_1, 6)]


    dofsV0_tan_D, dofsV0_tan_N = dofs_ess_nat(bc_D, W01_loc, V0_tan)
    dofsV2_tan_N, dofsV2_tan_D = dofs_ess_nat(bc_N, W32_loc, V2_tan)

    # Initial condition 01 and 32
    en_grad = Function(V_grad, name="e n")
    assign_exact01(p_ex, u_ex, en_grad, W01_loc, V0_tan, V01)
    pn_0, un_1, un_0_nor, pn_0_tan = en_grad.split()

    enmid_grad = Function(V_grad, name="e n+1/2")
    pnmid_0, unmid_1, unmid_0_nor, pnmid_0_tan = enmid_grad.split()

    en_div = Function(V_div, name="e n")
    assign_exact32(p_ex, u_ex, en_div, W32_loc, V2_tan, V32)
    pn_3, un_2, pn_2_nor, un_2_tan = en_div.split()

    enmid_div = Function(V_div, name="e n+1/2")
    pnmid_3, unmid_2, pnmid_2_nor, unmid_2_tan = enmid_div.split()

    # # Initial conditions post-processing
    # en_grad_pp = Function(W01_pp, name="e n")
    # assign_exact01_pp(p_ex, u_ex, en_grad_pp, W01_pp, V01)
    # pn_0_pp, un_1_pp = en_grad_pp.split()
    #
    # en_div_pp = Function(W32_pp, name="e n")
    # assign_exact32_pp(p_ex, u_ex, en_div_pp, W32_pp, V32)
    # pn_3_pp, un_2_pp = en_div_pp.split()

    # Exact quantities
    Hn_ex = 0.5 * (inner(p_ex, p_ex) * dx(domain=mesh) + inner(u_ex, u_ex) * dx(domain=mesh))
    bdflow_ex_nmid = p_ex_mid * dot(u_ex_mid, n_ver) * ds(domain=mesh)

    bdflow_ex_mid_vec = np.zeros((n_t,))
    H_ex_vec = np.zeros((1 + n_t,))

    H_ex_vec[0] = assemble(Hn_ex)

    # Power balance combining primal and dual
    bdflow_num_nmid = pnmid_0 * dot(unmid_2, n_ver) * ds(domain=mesh)

    bdflow_num_mid_vec = np.zeros((n_t,))
    Hdot_nmid_vec = np.zeros((n_t,))

    # Results 01
    Hn_01 = 0.5 * (inner(pn_0, pn_0) * dx(domain=mesh) + inner(un_1, un_1) * dx(domain=mesh))
    H_01_vec = np.zeros((1 + n_t,))
    errH_01_vec = np.zeros((1 + n_t,))

    errL2_p_0_vec = np.zeros((1 + n_t,))
    errL2_u_1_vec = np.zeros((1 + n_t,))
    errH1_p_0_vec = np.zeros((1 + n_t,))
    errHcurl_u_1_vec = np.zeros((1 + n_t,))
    errL2_p_0tan_vec = np.zeros((1 + n_t,))

    errL2_p_3_pp_vec = np.zeros((1 + n_t,))
    errL2_u_2_pp_vec = np.zeros((1 + n_t,))
    errHdiv_u_2_pp_vec = np.zeros((1 + n_t,))

    bdflow_01_mid_vec = np.zeros((n_t,))
    errL2_u_0nor_vec = np.zeros((n_t,))

    H_01_vec[0] = assemble(Hn_01)
    errH_01_vec[0] = np.abs(H_01_vec[0] - H_ex_vec[0])

    errL2_p_0_vec[0] = norm(p_ex-pn_0)
    errH1_p_0_vec[0] = errornorm(p_ex, pn_0, norm_type="H1")
    errL2_u_1_vec[0] = norm(u_ex-un_1)
    errHcurl_u_1_vec[0] = errornorm(u_ex, un_1, norm_type="Hcurl")
    err_p_0tan = h_cell * (p_ex - pn_0_tan) ** 2
    errL2_p_0tan_vec[0] = sqrt(assemble((err_p_0tan('+') + err_p_0tan('-')) * dS + err_p_0tan * ds))

    # errL2_p_3_pp_vec[0] = norm(p_ex - pn_3_pp)
    # errL2_u_2_pp_vec[0] = norm(u_ex - un_2_pp)
    # errHdiv_u_2_pp_vec[0] = norm(div(u_ex-un_2_pp))

    # Results 32
    Hn_32 = 0.5 * (inner(pn_3, pn_3) * dx(domain=mesh) + inner(un_2, un_2) * dx(domain=mesh))
    H_32_vec = np.zeros((1 + n_t,))
    errH_32_vec = np.zeros((1 + n_t,))

    errL2_p_3_vec = np.zeros((1 + n_t,))
    errL2_u_2_vec = np.zeros((1 + n_t,))
    errHdiv_u_2_vec = np.zeros((1 + n_t,))
    errL2_u_2tan_vec = np.zeros((1 + n_t,))

    errL2_p_0_pp_vec = np.zeros((1 + n_t,))
    errL2_u_1_pp_vec = np.zeros((1 + n_t,))
    errH1_p_0_pp_vec = np.zeros((1 + n_t,))
    errHcurl_u_1_pp_vec = np.zeros((1 + n_t,))

    bdflow_32_mid_vec = np.zeros((n_t,))
    errL2_p_2nor_vec = np.zeros((n_t,))

    H_32_vec[0] = assemble(Hn_32)
    errH_32_vec[0] = np.abs(H_32_vec[0] - H_ex_vec[0])

    errL2_p_3_vec[0] = norm(p_ex - pn_3)
    errL2_u_2_vec[0] = norm(u_ex - un_2)
    errHdiv_u_2_vec[0] = errornorm(u_ex, un_2, norm_type="Hdiv")
    err_u_2tan = h_cell * (u_ex - un_2_tan) ** 2
    errL2_u_2tan_vec[0] = sqrt(assemble((err_u_2tan('+') + err_u_2tan('-')) * dS + err_u_2tan * ds))

    # errL2_p_0_pp_vec[0] = norm(p_ex - pn_0_pp)
    # errH1_p_0_pp_vec[0] = norm(grad(p_ex-pn_0_pp))
    # errL2_u_1_pp_vec[0] = norm(u_ex - un_1_pp)
    # errHcurl_u_1_pp_vec[0] = norm(curl(u_ex-un_1_pp))

    # Dual Field representation
    err_p30_vec = np.zeros((1 + n_t,))
    err_u12_vec = np.zeros((1 + n_t,))

    err_p30_pp_vec = np.zeros((1 + n_t,))
    err_u12_pp_vec = np.zeros((1 + n_t,))

    err_p30_vec[0] = norm(pn_0 - pn_3)
    err_u12_vec[0] = norm(un_1 - un_2)

    # err_p30_pp_vec[0] = norm(pn_0_pp - pn_3_pp)
    # err_u12_pp_vec[0] = norm(un_1_pp - un_2_pp)

    ## Settings of intermediate variables and matrices for the 2 linear systems
    # Bilinear form 01
    a_form01 = m_form01(v0, p0, v1, u1) - 0.5 * dt * j_form01(v0, p0, v1, u1) \
               - 0.5 * dt * constr_loc01(v0, p0, v0_nor, u0_nor) \
               - 0.5 * dt * constr_global01(v0_nor, u0_nor, v0_tan, p0_tan)

    # Bilinear form 32
    a_form32 = m_form32(v3, p3, v2, u2) - 0.5 * dt * j_form32(v3, p3, v2, u2) \
               - 0.5 * dt * constr_loc32(v2, u2, v2_nor, p2_nor, n_ver) \
               - 0.5 * dt * constr_global32(v2_nor, p2_nor, v2_tan, u2_tan, n_ver)

    # # Post-processed solution
    # a_form01_pp = m_form01(v0_pp, p0_pp, v1_pp, u1_pp) - 0.5 * dt * j_form01(v0_pp, p0_pp, v1_pp, u1_pp)
    # A01_pp = Tensor(a_form01_pp).blocks
    #
    # # Bilinear form 32
    # a_form32_pp = m_form32(v3_pp, p3_pp, v2_pp, u2_pp) - 0.5 * dt * j_form32(v3_pp, p3_pp, v2_pp, u2_pp)
    # A32_pp = Tensor(a_form32_pp).blocks


    print("Computation of the solution with n elem " + str(n_el) + " n time " + str(n_t) + " deg " + str(deg))
    print("==============")

    for ii in tqdm(range(n_t)):

        input_u = interpolate(u_ex_mid, V32.sub(1))
        input_p = interpolate(p_ex_mid, V01.sub(0))

        ## Integration of 10 system (Neumann natural)

        b_form01 = m_form01(v0, pn_0, v1, un_1) + 0.5 * dt * j_form01(v0, pn_0, v1, un_1) \
                   + 0.5 * dt * constr_loc01(v0, pn_0, v0_nor, un_0_nor) \
                   + 0.5 * dt * constr_global01(v0_nor, un_0_nor, v0_tan, pn_0_tan)\
                   + dt * neumann_flow0(v0_tan, input_u, n_ver)

        en1_grad = solve_hybrid(a_form01, b_form01, bc_D, V0_tan, W01_loc)
        pn1_0, un1_1, un1_0_nor, pn1_0_tan = en1_grad.split()

        enmid_grad.assign(0.5 * (en_grad + en1_grad))
        pnmid_0, unmid_1, unmid_0_nor, pnmid_0_tan = enmid_grad.split()

        ## Integration of 32 system (Dirichlet natural)

        b_form32 = m_form32(v3, pn_3, v2, un_2) + 0.5 * dt * j_form32(v3, pn_3, v2, un_2) \
                   + 0.5 * dt * constr_loc32(v2, un_2, v2_nor, pn_2_nor, n_ver) \
                   + 0.5 * dt * constr_global32(v2_nor, pn_2_nor, v2_tan, un_2_tan, n_ver) \
                   + dt * dirichlet_flow2(v2_tan, input_p, n_ver)

        en1_div = solve_hybrid(a_form32, b_form32, bc_N, V2_tan, W32_loc)
        pn1_3, un1_2, pn1_2_nor, un1_2_tan = en1_div.split()

        enmid_div.assign(0.5 * (en_div + en1_div))
        pnmid_3, unmid_2, pnmid_2_nor, unmid_2_tan = enmid_div.split()

        # # Post-processed solution
        #
        # # 01 system
        # b_form01_pp = m_form01(v0_pp, pn_0_pp, v1_pp, un_1_pp) + 0.5 * dt * j_form01(v0_pp, pn_0_pp, v1_pp, un_1_pp) \
        #            + dt * neumann_flow0_pp(v0_pp, unmid_2_tan, n_ver)
        # b01_pp = Tensor(b_form01_pp).blocks
        #
        # en1_grad_pp = assemble(A01_pp[:, :].inv * b01_pp[:])
        #
        # # 32 system
        # b_form32_pp = m_form32(v3_pp, pn_3_pp, v2_pp, un_2_pp) + 0.5 * dt * j_form32(v3_pp, pn_3_pp, v2_pp, un_2_pp) \
        #            + dt * dirichlet_flow2_pp(v2_pp, pnmid_0_tan, n_ver)
        #
        # b32_pp = Tensor(b_form32_pp).blocks
        #
        # en1_div_pp = assemble(A32_pp[:, :].inv * b32_pp[:])

        if len(bc_N) == 0:
            bdflow_nat01 = 0
            bdflow_ess32 = 0
        else:
            y_nat01 = enmid_grad.vector().get_local()[dofsV0_tan_N]
            u_nat01 = assemble(neumann_flow0(v0_tan, input_u, n_ver)).vector().get_local()[dofsV0_tan_N]
            bdflow_nat01 = np.dot(y_nat01, u_nat01)

            form_tn_32 = inner(v2_tan, n_ver) * inner(pnmid_2_nor, n_ver)
            y_ess32_form = form_tn_32 * ds(domain=mesh)
            y_ess32 = assemble(y_ess32_form).vector().get_local()[dofsV2_tan_N]
            u_ess32 = assemble(enmid_div).vector().get_local()[dofsV2_tan_N]
            bdflow_ess32 = np.dot(y_ess32, u_ess32)

        if len(bc_D) == 0:
            bdflow_ess01 = 0
            bdflow_nat32 = 0

        else:
            y_nat32 = enmid_div.vector().get_local()[dofsV2_tan_D]
            u_nat32 = assemble(dirichlet_flow2(v2_tan, input_p, n_ver)).vector().get_local()[dofsV2_tan_D]
            bdflow_nat32 = np.dot(y_nat32, u_nat32)

            form_tn_01 = v0_tan * unmid_0_nor
            # Include dS because of vertices
            y_ess01_form = (form_tn_01('+') + form_tn_01('-')) * dS + form_tn_01 * ds(domain=mesh)
            y_ess01 = assemble(y_ess01_form).vector().get_local()[dofsV0_tan_D]
            u_ess01 = assemble(enmid_grad).vector().get_local()[dofsV0_tan_D]
            bdflow_ess01 = np.dot(y_ess01, u_ess01)

        # Power
        Hdot_nmid = 1 / dt * (dot(pnmid_0, pn1_3 - pn_3) * dx(domain=mesh) + dot(unmid_2, un1_1 - un_1) * dx(domain=mesh))
        Hdot_nmid_vec[ii] = assemble(Hdot_nmid)

        # Different power flow
        bdflow_01_mid_vec[ii] = bdflow_nat01 + bdflow_ess01
        bdflow_32_mid_vec[ii] = bdflow_nat32 + bdflow_ess32

        bdflow_ex_mid_vec[ii] = assemble(bdflow_ex_nmid)
        bdflow_num_mid_vec[ii] = assemble(bdflow_num_nmid)

        # New assign
        en_grad.assign(en1_grad)
        pn_0, un_1, un_0_nor, pn_0_tan = en_grad.split()

        en_div.assign(en1_div)
        pn_3, un_2, pn_2_nor, un_2_tan = en_div.split()

        # # New assign post-processing
        # en_grad_pp.assign(en1_grad_pp)
        # pn_0_pp, un_1_pp = en_grad_pp.split()
        #
        # en_div_pp.assign(en1_div_pp)
        # pn_3_pp, un_2_pp = en_div_pp.split()

        H_01_vec[ii + 1] = assemble(Hn_01)
        H_32_vec[ii + 1] = assemble(Hn_32)

        t.assign(float(t) + float(dt))
        t_1.assign(float(t_1) + float(dt))
        H_ex_vec[ii + 1] = assemble(Hn_ex)
        # Error H
        errH_01_vec[ii + 1] = np.abs(H_01_vec[ii + 1] - H_ex_vec[ii + 1])
        errH_32_vec[ii + 1] = np.abs(H_32_vec[ii + 1] - H_ex_vec[ii + 1])
        # Error 01
        errL2_p_0_vec[ii + 1] = norm(p_ex - pn_0)
        errL2_u_1_vec[ii + 1] = norm(u_ex - un_1)
        errHcurl_u_1_vec[ii + 1] = errornorm(u_ex, un_1, norm_type="Hcurl")
        errH1_p_0_vec[ii + 1] = errornorm(p_ex, pn_0, norm_type="H1")
        err_p_0tan = h_cell * (p_ex - pn_0_tan) ** 2
        errL2_p_0tan_vec[ii+1] =  sqrt(assemble((err_p_0tan('+') + err_p_0tan('-')) * dS + err_p_0tan * ds))
        # Computation of the error for lambda nor
        # First project normal trace of u_e onto Vnor
        uex_nor_p = project_uex_W0nor(u_ex, W01_loc.sub(2))
        err_u_0nor = h_cell * (uex_nor_p - un_0_nor) ** 2
        errL2_u_0nor_vec[ii] = sqrt(assemble((err_u_0nor('+') + err_u_0nor('-')) * dS + err_u_0nor * ds))

        # Error 32
        errL2_p_3_vec[ii + 1] = norm(p_ex - pn_3)
        errL2_u_2_vec[ii + 1] = norm(u_ex - un_2)
        errHdiv_u_2_vec[ii + 1] = errornorm(u_ex, un_2, norm_type="Hdiv")
        err_u_2tan = h_cell * inner(u_ex - un_2_tan, n_ver) ** 2
        errL2_u_2tan_vec[ii + 1] = sqrt(assemble((err_u_2tan('+') + err_u_2tan('-')) * dS + err_u_2tan * ds))
        # Computation of the error for lambda nor
        # First project normal trace of u_e onto Vnor
        pex_nor_p = project_pex_W2nor(p_ex, W32_loc.sub(2))
        err_p_2nor = h_cell * inner(pex_nor_p - pn_2_nor, n_ver) ** 2
        errL2_p_2nor_vec[ii] = sqrt(assemble((err_p_2nor('+') + err_p_2nor('-')) * dS + err_p_2nor * ds))

        # ## Post-processed error
        # errL2_p_0_pp_vec[ii + 1] = norm(p_ex - pn_0_pp)
        # errH1_p_0_pp_vec[ii + 1] = norm(grad(p_ex-pn_0_pp))
        #
        # errL2_u_1_pp_vec[ii + 1] = norm(u_ex - un_1_pp)
        # errHcurl_u_1_pp_vec[ii + 1] = norm(curl(u_ex-un_1_pp))
        #
        # errL2_p_3_pp_vec[ii + 1] = norm(p_ex - pn_3_pp)
        #
        # errL2_u_2_pp_vec[ii + 1] = norm(u_ex - un_2_pp)
        # errHdiv_u_2_pp_vec[ii + 1] = norm(div(u_ex - un_2_pp))

        # Dual Field
        err_p30_vec[ii + 1] = norm(pn_3 - pn_0)
        err_u12_vec[ii + 1] = norm(un_2 - un_1)

        # err_p30_pp_vec[ii + 1] = norm(pn_3_pp - pn_0_pp)
        # err_u12_pp_vec[ii + 1] = norm(un_2_pp - un_1_pp)

    # if dim==2:
    #     fig = plt.figure()
    #     axes = fig.add_subplot(111, projection='3d')
    #     contours = trisurf(interpolate(p_ex, V01.sub(0)), axes=axes, cmap="inferno")
    #     axes.set_aspect("auto")
    #     axes.set_title("p0 ex")
    #     fig.colorbar(contours)
    #
    #     fig = plt.figure()
    #     axes = fig.add_subplot(111, projection='3d')
    #     contours = trisurf(pn_0_pp, axes=axes, cmap="inferno")
    #     axes.set_aspect("auto")
    #     axes.set_title("p0_pp h")
    #     fig.colorbar(contours)
    #
    #     fig = plt.figure()
    #     axes = fig.add_subplot(111, projection='3d')
    #     contours = trisurf(pn_3_pp, axes=axes, cmap="inferno")
    #     axes.set_aspect("auto")
    #     axes.set_title("p3_pp h")
    #     fig.colorbar(contours)
    #
    #     fig = plt.figure()
    #     axes = fig.add_subplot(111)
    #     contours = quiver(u_ex, axes=axes, cmap="inferno")
    #     axes.set_aspect("auto")
    #     axes.set_title("u1 ex")
    #     fig.colorbar(contours)
    #
    #     fig = plt.figure()
    #     axes = fig.add_subplot(111)
    #     contours = quiver(un_1_pp, axes=axes, cmap="inferno")
    #     axes.set_aspect("auto")
    #     axes.set_title("u1_pp h")
    #     fig.colorbar(contours)
    #
    #     fig = plt.figure()
    #     axes = fig.add_subplot(111)
    #     contours = quiver(un_2_pp, axes=axes, cmap="inferno")
    #     axes.set_aspect("auto")
    #     axes.set_title("u2_pp h")
    #     fig.colorbar(contours)
    #
    #     plt.show()

    # Error 01
    errH_01 = errH_01_vec[-1]

    errL2_p_0 = errL2_p_0_vec[-1]
    errL2_u_1 = errL2_u_1_vec[-1]

    errH1_p_0 = errH1_p_0_vec[-1]
    errHcurl_u_1 = errHcurl_u_1_vec[-1]

    errL2_u_0nor = errL2_u_0nor_vec[-1]
    errL2_p_0tan = errL2_p_0tan_vec[-1]

    errL2_p_3_pp = errL2_p_3_pp_vec[-1]
    errL2_u_2_pp = errL2_u_2_pp_vec[-1]
    errHdiv_u_2_pp = errHdiv_u_2_pp_vec[-1]

    # Error 32
    errH_32 = errH_01_vec[-1]

    errL2_p_3 = errL2_p_3_vec[-1]
    errL2_u_2 = errL2_u_2_vec[-1]

    errHdiv_u_2 = errHdiv_u_2_vec[-1]

    errL2_p_2nor = errL2_p_2nor_vec[-1]
    errL2_u_2tan = errL2_u_2tan_vec[-1]

    errL2_p_0_pp = errL2_p_0_pp_vec[-1]
    errL2_u_1_pp = errL2_u_1_pp_vec[-1]
    errH1_p_0_pp = errH1_p_0_pp_vec[-1]
    errHcurl_u_1_pp = errHcurl_u_1_pp_vec[-1]

    # Dual field representation error
    err_p30 = err_p30_vec[-1]
    err_u12 = err_u12_vec[-1]

    err_p30_pp = err_p30_pp_vec[-1]
    err_u12_pp = err_u12_pp_vec[-1]

    # errL2_p_0 = max(errL2_p_0_vec)
    # errL2_u_1 = max(errL2_u_1_vec)
    #
    # errH1_p_0 = max(errH1_p_0_vec)
    # errHcurl_u_1 = max(errHcurl_u_1_vec)
    #
    # errL2_lambdanor = max(errL2_lambdanor_vec)
    # errL2_p_0tan = max(errL2_p_0tan_vec)
    #
    # errH_01 = max(errH_01_vec)

    # errL2_p_0 = np.sqrt(np.sum(float(dt) * np.power(errL2_p_0_vec, 2)))
    # errL2_u_1 = np.sqrt(np.sum(float(dt) * np.power(errL2_u_1_vec, 2)))
    # errH1_p_0 = np.sqrt(np.sum(float(dt) * np.power(errH1_p_0_vec, 2)))
    # errHcurl_u_1 = np.sqrt(np.sum(float(dt) * np.power(errHcurl_u_1_vec, 2)))
    # errL2_lambdanor = np.sqrt(np.sum(float(dt) * np.power(errL2_lambdanor_vec, 2)))
    # errL2_p_0tan = np.sqrt(np.sum(float(dt) * np.power(errL2_p_0tan_vec, 2)))
    # errH_01 = np.sqrt(np.sum(float(dt) * np.power(errH_01_vec, 2)))


    dict_res = {"t_span": t_vec, "energy_ex": H_ex_vec, "flow_ex_mid": bdflow_ex_mid_vec,\
                "flow_num_mid": bdflow_num_mid_vec, "Hdot_num_mid": Hdot_nmid_vec, \
                "energy_01": H_01_vec, "energy_32": H_32_vec, \
                "flow10_mid": bdflow_01_mid_vec, "err_u1": [errL2_u_1, errHcurl_u_1], "err_p0": [errL2_p_0, errH1_p_0],\
                "err_p0tan": errL2_p_0tan, "err_u0nor": errL2_u_0nor, \
                "flow32_mid": bdflow_32_mid_vec, "err_u2": [errL2_u_2, errHdiv_u_2], \
                "err_p3": errL2_p_3, "err_u2tan": errL2_u_2tan, "err_p2nor": errL2_p_2nor, \
                "err_p0_pp": [errL2_p_0_pp, errH1_p_0_pp], "err_u1_pp": [errL2_u_1_pp, errHcurl_u_1_pp], \
                "err_p3_pp": errL2_p_3_pp, "err_u2_pp": [errL2_u_2_pp, errHdiv_u_2_pp], \
                "err_p30": err_p30, "err_u12": err_u12, "err_p30_pp": err_p30_pp, "err_u12_pp": err_u12_pp,
                "err_H": [errH_01, errH_32],
                }

    return dict_res


# bd_cond = 'DN' #input("Enter bc: ")
#
# n_elem = 4
# pol_deg = 3
#
# n_time = 100
# t_fin = 1
#
# dt = t_fin / n_time
#
# results = compute_err(n_elem, n_time, pol_deg, t_fin, bd_cond=bd_cond, dim=3)
#
# # dictres_file = open("results_hybridwave.pkl", "wb")
# # pickle.dump(results, dictres_file)
# # dictres_file.close()
#
# t_vec = results["t_span"]
#
# bdflow10_mid = results["flow10_mid"]
# bdflow32_mid = results["flow32_mid"]
#
# H_01 = results["energy_01"]
# H_32 = results["energy_32"]
#
# H_ex = results["energy_ex"]
#
# bdflow_ex_nmid = results["flow_ex_mid"]
# bdflow_num_nmid = results["flow_num_mid"]
# Hdot_num_nmid = results["Hdot_num_mid"]
#
# errL2_u1, errHcurl_u1 = results["err_u1"]
# errL2_p0, errH1_p0 = results["err_p0"]
#
# err_H01 = results["err_H"]
#
# # plt.figure()
# # plt.plot(t_vec[1:]-dt/2, np.diff(H_01)/dt-bdflow10_mid, 'r-.', label="Power bal 10")
# # plt.xlabel(r'Time $[\mathrm{s}]$')
# # plt.legend()
# #
# # plt.figure()
# # plt.plot(t_vec[1:]-dt/2, np.diff(H_01)/dt, 'r-.', label="DHdt 10")
# # plt.plot(t_vec[1:]-dt/2, bdflow10_mid, 'b-.', label="flow 10")
# # plt.xlabel(r'Time $[\mathrm{s}]$')
# # plt.legend()
# #
# # plt.figure()
# # plt.plot(t_vec[1:]-dt/2, np.diff(H_32)/dt-bdflow32_mid, 'r-.', label="Power bal 32")
# # plt.xlabel(r'Time $[\mathrm{s}]$')
# # plt.legend()
# #
# # plt.figure()
# # plt.plot(t_vec[1:]-dt/2, np.diff(H_32)/dt, 'r-.', label="DHdt 32")
# # plt.plot(t_vec[1:]-dt/2, bdflow32_mid, 'b-.', label="flow 32")
# # plt.xlabel(r'Time $[\mathrm{s}]$')
# # plt.legend()
# #
# plt.figure()
# plt.plot(t_vec[1:]-dt/2, Hdot_num_nmid-bdflow_num_nmid, 'r-.', label="Discrete Power flow")
# plt.xlabel(r'Time $[\mathrm{s}]$')
# plt.legend()
#
# plt.figure()
# plt.plot(t_vec[1:]-dt/2, bdflow_ex_nmid-bdflow_num_nmid, 'r-.', label="Err power bal")
# plt.xlabel(r'Time $[\mathrm{s}]$')
# plt.legend()
#
#
# plt.show()

# plt.figure()
# plt.plot(t_vec[1:]-dt/2, bdflow10_mid, 'r-.', label="power flow 10")
# plt.plot(t_vec[1:]-dt/2, bdflow_ex_nmid, 'b-.', label="power flow ex")
# plt.xlabel(r'Time $[\mathrm{s}]$')
# plt.legend()
#
# plt.figure()
# plt.plot(t_vec[1:]-dt/2, np.diff(H_01)/dt, 'r-.', label="DHdt")
# plt.plot(t_vec[1:]-dt/2, bdflow_ex_nmid, 'b-.', label="flow exact")
# plt.xlabel(r'Time $[\mathrm{s}]$')
# plt.legend()

#
# plt.figure()
# plt.plot(t_vec[1:]-dt/2, abs(bdflow_ex_nmid - bdflow10_mid), 'r-.')
# plt.xlabel(r'Time $[\mathrm{s}]$')
# plt.title(r'Power balance conservation')

# dictres_file = open("results_wave.pkl", "wb")
# pickle.dump(results, dictres_file)
# dictres_file.close()
#


