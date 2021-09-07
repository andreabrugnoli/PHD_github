## This is a first test to solve the wave equation in 3d domains using the dual field method
## A staggering method is used for the time discretization

import os
os.environ["OMP_NUM_THREADS"] = "1"

# import numpy as np
from firedrake import *
import matplotlib.pyplot as plt
from tools_plotting import setup
from tqdm import tqdm
# from time import sleep


def compute_err(n_el, n_t, deg=1, t_fin=1, bd_cond="D"):
    """Compute the numerical solution of the wave equation with the dual field method

        Parameters:
        n_el: number of elements for the discretization
        n_t: number of time instants
        deg: polynomial degree for finite
        Returns:
        some plots

       """

    def m_form32(v_3, p_3, v_2, u_2):
        m_form = inner(v_3, p_3) * dx + inner(v_2, u_2)

        return m_form

    def m_form10(v_1, u_1, v_0, u_0):
        m_form = inner(v_1, u_1) * dx + inner(v_0, u_0)

        return m_form

    def j_form32(v_3, p_3, v_2, u_2):
        j_form = dot(v_3, div(u_2)) * dx - dot(div(v_2), p_3) * dx

        return j_form

    def j_form10(v_1, u_1, v_0, p_0):
        j_form = dot(v_1, grad(p_0)) * dx - dot(grad(v_0), u_1) * dx

        return j_form

    def bdflow32(v_2, p_0):
        b_form = dot(v_2, n_ver) * p_0 * ds

        return b_form

    def bdflow10(v_0, u_2):
        b_form = v_0 * dot(u_2, n_ver) * ds

        return b_form

    L = 1
    mesh = CubeMesh(n_el, n_el, n_el, L)
    n_ver = FacetNormal(mesh)

    P_0 = FiniteElement("CG", tetrahedron, deg)
    P_1 = FiniteElement("N1curl", tetrahedron, deg, variant='spectral')
    P_2 = FiniteElement("RT", tetrahedron, deg)
    # Integral evaluation on Raviart-Thomas completely freezes interpolation
    # P_2 = FiniteElement("RT", tetrahedron, deg, variant='integral')
    P_3 = FiniteElement("DG", tetrahedron, deg - 1)

    V_3 = FunctionSpace(mesh, P_3)
    V_1 = FunctionSpace(mesh, P_1)

    V_0 = FunctionSpace(mesh, P_0)
    V_2 = FunctionSpace(mesh, P_2)

    V_32 = V_3 * V_2
    V_10 = V_1 * V_0

    v_32 = TestFunction(V_32)
    v_3, v_2 = split(v_32)

    v_10 = TestFunction(V_10)
    v_1, v_0 = split(v_10)

    e_32 = TrialFunction(V_32)
    p_3, u_2 = split(e_32)

    e_10 = TrialFunction(V_10)
    u_1, p_0 = split(e_10)

    dx = Measure('dx')
    ds = Measure('ds')

    x, y, z = SpatialCoordinate(mesh)

    om_x = pi
    om_y = pi
    om_z = pi

    om_t = np.sqrt(om_x ** 2 + om_y ** 2 + om_z ** 2)
    phi_x = 1
    phi_y = 2
    phi_z = 2
    phi_t = 3

    t = Constant(0.0)
    # w_ex = sin(om_x * x + phi_x) * sin(om_y * y + phi_y) * sin(om_z * z + phi_z) * sin(om_t * t + phi_t)

    p_ex = om_t * sin(om_x * x + phi_x) * sin(om_y * y + phi_y) * sin(om_z * z + phi_z) * cos(om_t * t + phi_t)
    u_ex = as_vector([om_x * cos(om_x * x + phi_x) * sin(om_y * y + phi_y) * sin(om_z * z + phi_z) * sin(om_t * t + phi_t),
                        om_y * sin(om_x * x + phi_x) * cos(om_y * y + phi_y) * sin(om_z * z + phi_z) * sin(om_t * t + phi_t),
                        om_z * sin(om_x * x + phi_x) * sin(om_y * y + phi_y) * cos(om_z * z + phi_z) * sin(om_t * t + phi_t)])

    p0_3 = interpolate(p_ex, V_3)
    u0_2 = interpolate(u_ex, V_2)

    u0_1 = interpolate(u_ex, V_1)
    p0_0 = interpolate(p_ex, V_0)

    #
    # e0_32 = Function(V_32)
    # e0_32.sub(0).assign(p0_3)
    # e0_32.sub(1).assign(u0_2)
    #
    # e0_10 = Function(V_10)
    # e0_10.sub(0).assign(u0_1)
    # e0_10.sub(1).assign(p0_0)
    #
    # # e0_32 = project(as_vector([v_ex, sig_ex[0], sig_ex[1], sig_ex[2]]), V_32)
    # # e0_10 = project(as_vector([sig_ex[0], sig_ex[1], sig_ex[2], v_ex]), V_10)
    #
    # p0_3, u0_2 = split(e0_32)
    # u0_2, p0_0 = split(e0_10)

    if bd_cond=="D":
        bc_D = DirichletBC(V_10.sub(1), p_ex, "on_boundary")
        bc_N = None
    elif bd_cond=="N":
        bc_N = DirichletBC(V_32.sub(1), u_ex, "on_boundary")
        bc_D = None
    else:
        bc_D = None
        bc_N = None

    H_32_vec = np.zeros((1 + n_t,))
    H_01_vec = np.zeros((1 + n_t,))

    Hdot_vec = np.zeros((1 + n_t,))
    bdflow_vec = np.zeros((1 + n_t,))

    err_p_3_vec = np.zeros((1 + n_t,))
    err_u_1_vec = np.zeros((1 + n_t,))

    err_p_0_vec = np.zeros((1 + n_t,))
    err_u_2_vec = np.zeros((1 + n_t,))

    err_p_3_vec[0] = np.sqrt(assemble((p0_3 - p_ex)**2 * dx))
    err_u_1_vec[0] = np.sqrt(assemble(dot(u0_1 - u_ex, u0_1 - u_ex) * dx))
    # q_err_Hcurl_vec[0] = np.sqrt(assemble(dot(q0 - sig_ex, q0 - sig_ex) * dx + dot(curl(q0 - sig_ex), curl(q0 - sig_ex))*dx))

    err_p_0_vec[0] = np.sqrt(assemble((p0_0 - p_ex)**2 * dx + dot(grad(p0_0 - p_ex), grad(p0_0 - p_ex))*dx))
    err_u_2_vec[0] = np.sqrt(assemble(dot(u0_2 - u_ex, u0_2 - u_ex) * dx + dot(div(u0_2 - u_ex), div(u0_2 - u_ex)) * dx))

    # Ppoint = (L/7, L/5, L/3)
    #
    # p_P = np.zeros((1+n_t,))
    # p_P[0] = interpolate(v_ex, Vp).at(Ppoint)
    #
    # pd_P = np.zeros((1+n_t, ))
    # pd_P[0] = interpolate(v_ex, Vp_d).at(Ppoint)

    e0_32 = Function(V_32, name="e_32 initial")
    e0_10 = Function(V_10, name="e_10 initial")

    e0_32.sub(0).assign(p0_3)
    e0_32.sub(1).assign(u0_2)

    e0_10.sub(0).assign(u0_1)
    e0_10.sub(1).assign(p0_0)

    H_32_vec[0] = assemble(0.5 * (inner(p0_3, p0_3) * dx + inner(u0_2, u0_2) * dx))
    H_01_vec[0] = assemble(0.5 * (inner(p0_0, p0_0) * dx + inner(u0_1, u0_1) * dx))

    Hdot_vec[0] = assemble(div(u0_2) * p0_0 * dx + inner(grad(p0_0), u0_2) * dx)
    bdflow_vec[0] = assemble(p0_0 * dot(u0_2, n_ver) * ds)

    dt = Constant(t_fin / n_t)

    params = {"mat_type": "aij",
              "snes_type": "ksponly",
              "ksp_type": "preonly",
              "pc_type": "lu"}

    t_vec = np.linspace(0, n_t * float(dt), 1 + n_t)

    # pn = Function(Vp, name="p primal at t_n")
    # pn_d = Function(Vp_d, name="p dual at t_n")
    #
    # qn = Function(Vq, name="q primal at t_n")
    # qn_d = Function(Vq_d, name="q dual at t_n")
    #
    # err_p = Function(Vp, name="p error primal at t_n")
    # err_pd = Function(Vp_d, name="p error dual at t_n")

    enmid_32 = Function(V_32, name="e_32 n+1/2")
    enmid1_32 = Function(V_32, name="e_32 n+3/2")

    en_32 = Function(V_32, name="e_32 n+1/2")
    en_32.assign(e0_32)
    en1_32 = Function(V_32, name="e_32 n+3/2")

    en_10 = Function(V_10, name="e_10 n")
    en_10.assign(e0_10)
    en1_10 = Function(V_10, name="e_10 n+1")

    print("First explicit step")
    print("==============")

    b0_form_32 = m_form32(v_3, p0_3, v_2, u0_2) + dt / 2 * (j_form32(v_3, p0_3, v_2, u0_2) + bdflow32(v_2, p0_0))

    if bc_N == None:
        A0_32 = assemble(m_form32(v_3, p_3, v_2, u_2), mat_type='aij')
        b0_32 = assemble(b0_form_32)
    else:
        A0_32 = assemble(m_form32(v_3, p_3, v_2, u_2), bcs=bc_N, mat_type='aij')
        b0_32 = assemble(b0_form_32, bcs=bc_N)

    solve(A0_32, enmid_32, b0_32, solver_parameters=params)

    pnmid_3, unmid_2 = enmid_32.split()

    ## Settings of intermediate variables and matrices for the 2 linear systems

    pn_3, un_2 = en_32.split()
    un_1, pn_0 = en_10.split()

    Hn_32 = 0.5 * (inner(pn_3, pn_3) * dx + inner(un_2, un_2) * dx)
    Hn_01 = 0.5 * (inner(pn_0, pn_0) * dx + inner(un_1, un_1) * dx)

    Hdot_n = div(un_2) * pn_0 * dx + inner(grad(pn_0), un_2) * dx
    bdflow_n = pn_0 * dot(un_2, n_ver) * ds

    a_form10 = m_form10(v_1, u_1, v_0, p_0) - 0.5*dt*j_form10(v_1, u_1, v_0, p_0)
    a_form32 = m_form32(v_3, p_3, v_2, u_2) - 0.5*dt*j_form32(v_3, p_3, v_2, u_2)

    if bc_D == None:
        A_10 = assemble(a_form10, mat_type='aij')
    else:
        A_10 = assemble(a_form10, bcs=bc_D, mat_type='aij')

    if bc_N == None:
        A_32 = assemble(a_form32, mat_type='aij')
    else:
        A_32 = assemble(a_form32, bcs=bc_N, mat_type='aij')

    print("Computation of the solution")
    print("==============")

    for ii in tqdm(range(n_t)):

        ## Integration of 10 system using unmid_2

        b_form10 = m_form10(v_1, un_1, v_0, pn_0) + dt*(0.5*j_form10(v_1, un_1, v_0, pn_0) + bdflow10(v_0, unmid_2))

        if bc_D == None:
            b_vec10 = assemble(b_form10)
        else:
            b_vec10 = assemble(b_form10, bcs=bc_D)

        solve(A_10, en1_10, b_vec10, solver_parameters=params)



        Hdot_vec[ii+1] = assemble(Hdot)
        bdflow_vec[ii+1] = assemble(bdflow)

        E_L2Hdiv_vec[ii+1] = assemble(E_L2Hdiv)
        E_H1Hrot_vec[ii+1] = assemble(E_H1Hrot)

        # pn.assign(interpolate(p0, Vp))
        # pn_d.assign(interpolate(p0_d, Vp_d))
        #
        # qn.assign(interpolate(q0, Vq))
        # qn_d.assign(interpolate(q0_d, Vq_d))
        #
        # p_P[ii+1] = pn.at(Ppoint)
        # pd_P[ii+1] = pn_d.at(Ppoint)

        t.assign(float(t) + float(dt))
        # print("Primal energy")
        # print("{0:1.1e} {1:5e}".format(float(t), Ep_vec[ii]))
        # print("Dual energy")
        # print("{0:1.1e} {1:5e}".format(float(t), Ed_vec[ii]))
        # print("Scattering energy")
        # print("{0:1.1e} {1:5e}".format(float(t), Es_vec[ii]))

        p_err_L2_vec[ii + 1] = np.sqrt(assemble((p0 - v_ex) ** 2 * dx))
        q_err_Hcurl_vec[ii + 1] = np.sqrt(assemble(dot(q0 - sig_ex, q0 - sig_ex)*dx))
        # q_err_Hcurl_vec[ii + 1] = np.sqrt(assemble(dot(q0 - sig_ex, q0 - sig_ex) * dx + dot(curl(q0 - sig_ex), curl(q0 - sig_ex))*dx))

        pd_err_H1_vec[ii + 1] = np.sqrt(assemble((p0_d - v_ex) ** 2 * dx + dot(grad(p0_d - v_ex), grad(p0_d - v_ex)) * dx))
        qd_err_Hdiv_vec[ii + 1] = np.sqrt(assemble(dot(q0_d - sig_ex, q0_d - sig_ex) * dx + dot(div(q0_d - sig_ex), div(q0_d - sig_ex)) * dx))

    # err_p.assign(pn - interpolate(v_ex, Vp))
    # err_pd.assign(pn_d - interpolate(v_ex, Vp_d))

    # print(r"Initial and final primal energy:")
    # print(r"Inital: ", Ep_vec[0])
    # print(r"Final: ", Ep_vec[-1])
    # print(r"Delta: ", Ep_vec[-1] - Ep_vec[0])
    #
    # print(r"Initial and final dual energy:")
    # print(r"Inital: ", Ed_vec[0])
    # print(r"Final: ", Ed_vec[-1])
    # print(r"Delta: ", Ed_vec[-1] - Ed_vec[0])
    #
    # print(r"Initial and final scattering energy:")
    # print(r"Inital: ", Es_vec[0])
    # print(r"Final: ", Es_vec[-1])
    # print(r"Delta: ", Es_vec[-1] - Es_vec[0])

    # fig = plt.figure()
    # axes = fig.add_subplot(111, projection='3d')
    # contours = trisurf(err_p, axes=axes, cmap="inferno")
    # axes.set_aspect("auto")
    # axes.set_title("Error primal velocity")
    # fig.colorbar(contours)
    #
    # fig = plt.figure()
    # axes = fig.add_subplot(111, projection='3d')
    # contours = trisurf(err_pd, axes=axes, cmap="inferno")
    # axes.set_aspect("auto")
    # axes.set_title("Error dual velocity")
    # fig.colorbar(contours)
    #
    # fig = plt.figure()
    # axes = fig.add_subplot(111, projection='3d')
    # contours = trisurf(pn, axes=axes, cmap="inferno")
    # axes.set_aspect("auto")
    # axes.set_title("Primal velocity")
    # fig.colorbar(contours)
    #
    # fig = plt.figure()
    # axes = fig.add_subplot(111, projection='3d')
    # contours = trisurf(pn_d, axes=axes, cmap="inferno")
    # axes.set_aspect("auto")
    # axes.set_title("Dual velocity")
    # fig.colorbar(contours)

    # plt.figure()
    # plt.plot(t_vec, p_P, 'r-', label=r'primal $p$')
    # plt.plot(t_vec, pd_P, 'b-', label=r'dual $p$')
    # plt.plot(t_vec, om_t * np.sin(om_x * Ppoint[0] + phi_x) * np.sin(om_y * Ppoint[1] + phi_y) * \
    #          np.cos(om_t * t_vec + phi_t), 'g-', label=r'exact $p$')
    # plt.xlabel(r'Time [s]')
    # plt.title(r'$p$ at ' + str(Ppoint))
    # plt.legend()
    #
    # plt.show()

    # p_err_L2 = max(p_err_L2_vec)
    # q_err_Hrot = max(q_err_Hrot_vec)
    #
    # pd_err_H1 = max(pd_err_H1_vec)
    # qd_err_Hdiv = max(qd_err_Hdiv_vec)

    p_err_L2 = p_err_L2_vec[-1]
    q_err_Hrot = q_err_Hcurl_vec[-1]

    pd_err_H1 = pd_err_H1_vec[-1]
    qd_err_Hdiv = qd_err_Hdiv_vec[-1]

    dict_res = {"t_span": t_vec, "energy_L2Hdiv": E_L2Hdiv_vec, "energy_H1Hrot": E_H1Hrot_vec, "power": Hdot_vec, \
                "flow": bdflow_vec, "p_err": p_err_L2, "q_err": q_err_Hrot, "pd_err": pd_err_H1, "qd_err": qd_err_Hdiv}

    return dict_res

# results = compute_err(10, 100, 2, 2)

