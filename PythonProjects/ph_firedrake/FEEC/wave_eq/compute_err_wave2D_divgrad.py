## This is a first test to solve the wave equation in 3d domains using the dual field method
## A staggering method is used for the time discretization

import os
os.environ["OMP_NUM_THREADS"] = "1"

# import numpy as np
from firedrake import *
from irksome import GaussLegendre, Dt, AdaptiveTimeStepper, TimeStepper, LobattoIIIA

import  matplotlib.pyplot as plt
from tools_plotting import setup

from tqdm import tqdm


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
        m_form = inner(v_3, p_3) * dx + inner(v_2, u_2) * dx

        return m_form

    def m_form10(v_1, u_1, v_0, u_0):
        m_form = inner(v_1, u_1) * dx + inner(v_0, u_0) * dx

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
    mesh = RectangleMesh(n_el, n_el, L, L, quadrilateral=False)
    n_ver = FacetNormal(mesh)

    P_0 = FiniteElement("CG", triangle, deg)
    P_1 = FiniteElement("N1curl", triangle, deg, variant='integral')
    # P_2 = FiniteElement("RT", triangle, deg)
    # Integral evaluation on Raviart-Thomas for deg=3 completely freezes interpolation
    P_2 = FiniteElement("RT", triangle, deg, variant='integral')
    P_3 = FiniteElement("DG", triangle, deg - 1)

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

    x, y = SpatialCoordinate(mesh)

    om_x = pi
    om_y = pi

    om_t = np.sqrt(om_x ** 2 + om_y ** 2)
    phi_x = 0
    phi_y = 0
    phi_t = 0

    t = Constant(0.0)
    # w_ex = sin(om_x * x + phi_x) * sin(om_y * y + phi_y) * sin(om_z * z + phi_z) * sin(om_t * t + phi_t)

    p_ex = om_t * sin(om_x * x + phi_x) * sin(om_y * y + phi_y) * cos(om_t * t + phi_t)
    u_ex = as_vector([om_x * cos(om_x * x + phi_x) * sin(om_y * y + phi_y) * sin(om_t * t + phi_t),
                        om_y * sin(om_x * x + phi_x) * cos(om_y * y + phi_y) * sin(om_t * t + phi_t)])

    in_cond_32 = Function(V_32)
    in_cond_32.sub(0).assign(interpolate(p_ex, V_3))
    in_cond_32.sub(1).assign(interpolate(u_ex, V_2))

    in_cond_10 = Function(V_10)
    in_cond_10.sub(0).assign(interpolate(p_ex, V_0))
    in_cond_10.sub(1).assign(interpolate(u_ex, V_1))

    pn_30, q0, p0_d, q0_d = split(in_cond_32)

    if bd_cond=="D":
        bc_D = DirichletBC(V_10.sub(1), p_ex, "on_boundary")
        bc_N = None
    elif bd_cond=="N":
        bc_N = DirichletBC(V_32.sub(1), u_ex, "on_boundary")
        bc_D = None
    else:
        bc_D = None
        bc_N = None

    Ppoint = (L/5, L/5)

    p_0P = np.zeros((1+n_t,))
    p_0P[0] = interpolate(p_ex, V_0).at(Ppoint)

    p_3P = np.zeros((1+n_t, ))
    p_3P[0] = interpolate(p_ex, V_3).at(Ppoint)

    e0_32 = Function(V_32, name="e_32 initial")
    e0_10 = Function(V_10, name="e_10 initial")

    e0_32.sub(0).assign(p0_3)
    e0_32.sub(1).assign(u0_2)

    e0_10.sub(0).assign(u0_1)
    e0_10.sub(1).assign(p0_0)

    dt = Constant(t_fin / n_t)
    butcher_tableau = GaussLegendre(1)

    params = {"mat_type": "aij",
              "snes_type": "ksponly",
              "ksp_type": "preonly",
              "pc_type": "lu"}

    t_vec = np.linspace(0, n_t * float(dt), 1 + n_t)


    Hn_32 = 0.5 * (inner(pn_3, pn_3) * dx + inner(un_2, un_2) * dx)
    Hn_10 = 0.5 * (inner(pn_0, pn_0) * dx + inner(un_1, un_1) * dx)

    Hdot_n = div(un_2) * pn_0 * dx + inner(grad(pn_0), un_2) * dx
    bdflow_n = pn_0 * dot(un_2, n_ver) * ds

    bdflow_ex_n = p_ex * dot(u_ex, n_ver) * ds

    H_32_vec = np.zeros((1 + n_t,))
    H_10_vec = np.zeros((1 + n_t,))

    Hdot_vec = np.zeros((1 + n_t,))
    bdflow_vec = np.zeros((1 + n_t,))
    bdflow_ex_vec = np.zeros((1 + n_t,))

    err_p_3_vec = np.zeros((1 + n_t,))
    err_u_1_vec = np.zeros((1 + n_t,))

    err_p_0_vec = np.zeros((1 + n_t,))
    err_u_2_vec = np.zeros((1 + n_t,))

    err_p30_vec = np.zeros((1 + n_t,))
    err_u12_vec = np.zeros((1 + n_t,))

    H_32_vec[0] = assemble(Hn_32)
    H_10_vec[0] = assemble(Hn_10)

    Hdot_vec[0] = assemble(Hdot_n)
    bdflow_vec[0] = assemble(bdflow_n)
    bdflow_ex_vec[0] = assemble(bdflow_ex_n)

    err_p_3_vec[0] = errornorm(p_ex, p0_3, norm_type="L2")
    err_u_1_vec[0] = errornorm(u_ex, u0_1, norm_type="L2")
    err_p_0_vec[0] = errornorm(p_ex, p0_0, norm_type="H1")
    err_u_2_vec[0] = errornorm(u_ex, u0_2, norm_type="Hdiv")

    diff0_p30 = project(p0_3 - p0_0, V_3)
    diff0_u12 = project(u0_2 - u0_1, V_1)

    err_p30_vec[0] = errornorm(Constant(0), diff0_p30, norm_type="L2")
    err_u12_vec[0] = errornorm(Constant((0.0, 0.0)), diff0_u12, norm_type="L2")

    print("First explicit step")
    print("==============")

    a0_form32 = m_form32(v_3, p_3, v_2, u_2)
    b0_form32 = m_form32(v_3, p0_3, v_2, u0_2) + dt / 2 * (j_form32(v_3, p0_3, v_2, u0_2) + bdflow32(v_2, p0_0))

    A0_32 = assemble(a0_form32, bcs=bc_N, mat_type='aij')
    b0_32 = assemble(b0_form32)

    solve(A0_32, enmid_32, b0_32, solver_parameters=params)

    # print("First implicit step")
    # print("==============")
    # V_3210 = V_32 * V_10
    # w_3210 = TestFunction(V_3210)
    # w_3, w_2, w_1, w_0 = split(w_3210)
    #
    # e_3210 = TrialFunction(V_3210)
    # e_3, e_2, e_1, e_0 = split(e_3210)
    #
    # en1 = Function(V_3210)
    #
    # a0_form = m_form32(w_3, e_3, w_2, e_2) + m_form10(w_1, e_1, w_0, e_0) - 0.5 * dt * (j_form10(w_1, e_1, w_0, e_0) \
    #             + j_form32(w_3, e_3, w_2, e_2)  + bdflow10(w_0, e_2) + bdflow32(w_2, e_0))
    # b0_form = m_form10(w_1, un_1, w_0, pn_0) + m_form32(w_3, pn_3, w_2, un_2) + 0.5*dt*(j_form10(w_1, un_1, w_0, pn_0)\
    #             + j_form32(w_3, pn_3, w_2, un_2) + bdflow10(w_0, un_2) + bdflow32(w_2, pn_0))
    #
    # if bd_cond=="D":
    #     bc_D_first = DirichletBC(V_3210.sub(3), p_ex, "on_boundary")
    #     A0 = assemble(a0_form, bcs=bc_D_first, mat_type='aij')
    # elif bd_cond=="N":
    #     bc_N_first = DirichletBC(V_3210.sub(1), u_ex, "on_boundary")
    #     A0 = assemble(a0_form, bcs=bc_N_first, mat_type='aij')
    #
    # b0 = assemble(b0_form)
    #
    # solve(A0, en1, b0, solver_parameters=params)
    #
    # en1_32.sub(0).assign(en1.split()[0])
    # en1_32.sub(1).assign(en1.split()[1])
    #
    # enmid_32.assign(0.5*(en_32 + en1_32))

    ## Settings of intermediate variables and matrices for the 2 linear systems

    print("Assemble of the A matrices")
    print("==============")

    a_form10 = m_form10(v_1, u_1, v_0, p_0) - 0.5*dt*j_form10(v_1, u_1, v_0, p_0)
    a_form32 = m_form32(v_3, p_3, v_2, u_2) - 0.5*dt*j_form32(v_3, p_3, v_2, u_2)

    A_10 = assemble(a_form10, bcs=bc_D, mat_type='aij')
    A_32 = assemble(a_form32, bcs=bc_N, mat_type='aij')

    print("Computation of the solution")
    print("==============")

    for ii in tqdm(range(n_t)):

        ## Integration of 10 system using unmid_2

        pnmid_3, unmid_2 = enmid_32.split()
        b_form10 = m_form10(v_1, un_1, v_0, pn_0) + dt*(0.5*j_form10(v_1, un_1, v_0, pn_0) + bdflow10(v_0, unmid_2))

        b_vec10 = assemble(b_form10)

        solve(A_10, en1_10, b_vec10, solver_parameters=params)

        un1_1, pn1_0 = en1_10.split()

        ## Integration of 32 system using pn1_0

        b_form32 = m_form32(v_3, pnmid_3, v_2, unmid_2) + dt*(0.5*j_form32(v_3, pnmid_3, v_2, unmid_2) \
                                                              + bdflow32(v_2, pn1_0))
        b_vec32 = assemble(b_form32)

        solve(A_32, enmid1_32, b_vec32, solver_parameters=params)

        # If it does not work split and then assign
        pnmid_3, unmid_2 = enmid_32.split()
        pnmid1_3, unmid1_2 = enmid1_32.split()

        en1_32.sub(0).assign(0.5*(pnmid_3 + pnmid1_3))
        en1_32.sub(1).assign(0.5*(unmid_2 + unmid1_2))

        en_32.assign(en1_32)

        # en_32.assign(0.5 * (enmid_32 + enmid1_32))

        en_10.assign(en1_10)
        enmid_32.assign(enmid1_32)

        un_1, pn_0 = en_10.split()
        pn_3, un_2 = en_32.split()

        Hdot_vec[ii+1] = assemble(Hdot_n)
        bdflow_vec[ii+1] = assemble(bdflow_n)

        H_32_vec[ii+1] = assemble(Hn_32)
        H_10_vec[ii+1] = assemble(Hn_10)

        t.assign(float(t) + float(dt))

        bdflow_ex_vec[ii+1] = assemble(bdflow_ex_n)
        # print(bdflow_ex_vec[ii+1])
        err_p_3_vec[ii+1] = errornorm(p_ex, pn_3, norm_type="L2")
        err_u_1_vec[ii+1] = errornorm(u_ex, un_1, norm_type="L2")
        err_p_0_vec[ii+1] = errornorm(p_ex, pn_0, norm_type="H1")
        err_u_2_vec[ii+1] = errornorm(u_ex, un_2, norm_type="Hdiv")

        diffn_p30 = project(pn_3 - pn_0, V_3)
        diffn_u12 = project(un_2 - un_1, V_1)

        err_p30_vec[ii+1] = errornorm(Constant(0), diffn_p30, norm_type="L2")
        err_u12_vec[ii+1] = errornorm(Constant((0.0, 0.0)), diffn_u12, norm_type="L2")

    #     p_3P[ii + 1] = pn_3.at(Ppoint)
    #     p_0P[ii + 1] = pn_0.at(Ppoint)
    #
    # err_p3.assign(pn_3 - interpolate(p_ex, V_3))
    # err_p0.assign(pn_0 - interpolate(p_ex, V_0))

    # fig = plt.figure()
    # axes = fig.add_subplot(111, projection='3d')
    # contours = trisurf(err_p3, axes=axes, cmap="inferno")
    # axes.set_aspect("auto")
    # axes.set_title("Error $p_3$")
    # fig.colorbar(contours)
    #
    # fig = plt.figure()
    # axes = fig.add_subplot(111, projection='3d')
    # contours = trisurf(err_p0, axes=axes, cmap="inferno")
    # axes.set_aspect("auto")
    # axes.set_title("Error $p_0$")
    # fig.colorbar(contours)

    # fig = plt.figure()
    # axes = fig.add_subplot(111, projection='3d')
    # contours = trisurf(interpolate(p_ex, V_3), axes=axes, cmap="inferno")
    # axes.set_aspect("auto")
    # axes.set_title("$p_3$ Exact")
    # fig.colorbar(contours)
    #
    # fig = plt.figure()
    # axes = fig.add_subplot(111, projection='3d')
    # contours = trisurf(interpolate(p_ex, V_0), axes=axes, cmap="inferno")
    # axes.set_aspect("auto")
    # axes.set_title("$p_0$ Exact")
    # fig.colorbar(contours)
    #
    # fig = plt.figure()
    # axes = fig.add_subplot(111, projection='3d')
    # contours = trisurf(pn_3, axes=axes, cmap="inferno")
    # axes.set_aspect("auto")
    # axes.set_title("$P_3$")
    # fig.colorbar(contours)
    #
    # fig = plt.figure()
    # axes = fig.add_subplot(111, projection='3d')
    # contours = trisurf(pn_0, axes=axes, cmap="inferno")
    # axes.set_aspect("auto")
    # axes.set_title("$P_0$")
    # fig.colorbar(contours)

    # print(r"Initial and final 32 energy:")
    # print(r"Inital: ", H_32_vec[0])
    # print(r"Final: ", H_32_vec[-1])
    # print(r"Delta: ", H_32_vec[-1] - H_32_vec[0])
    #
    # print(r"Initial and final 10 energy:")
    # print(r"Inital: ", H_10_vec[0])
    # print(r"Final: ", H_10_vec[-1])
    # print(r"Delta: ", H_10_vec[-1] - H_10_vec[0])
    #
    # plt.figure()
    # plt.plot(t_vec, p_3P, 'r-', label=r'$p_3$')
    # plt.plot(t_vec, p_0P, 'b-', label=r'$p_0$')
    # plt.plot(t_vec, om_t * np.sin(om_x * Ppoint[0] + phi_x) * np.sin(om_y * Ppoint[1] + phi_y) \
    #          * np.cos(om_t * t_vec + phi_t), 'g-', label=r'exact $p$')
    # plt.xlabel(r'Time [s]')
    # plt.title(r'$p$ at ' + str(Ppoint))
    # plt.legend()

    # err_p_3 = np.sqrt(np.sum(float(dt) * np.power(err_p_3_vec, 2)))
    # err_u_1 = np.sqrt(np.sum(float(dt) * np.power(err_u_1_vec, 2)))
    # err_p_0 = np.sqrt(np.sum(float(dt) * np.power(err_p_0_vec, 2)))
    # err_u_2 = np.sqrt(np.sum(float(dt) * np.power(err_u_2_vec, 2)))

    # err_p_3 = max(err_p_3_vec)
    # err_u_1 = max(err_u_1_vec)
    #
    # err_p_0 = max(err_p_0_vec)
    # err_u_2 = max(err_u_2_vec)
    #
    # err_p30 = max(err_p30_vec)
    # err_u12 = max(err_u12_vec)

    err_p_3 = err_p_3_vec[-1]
    err_u_1 = err_u_1_vec[-1]

    err_p_0 = err_p_0_vec[-1]
    err_u_2 = err_u_2_vec[-1]

    err_p30 = err_p30_vec[-1]
    err_u12 = err_u12_vec[-1]

    dict_res = {"t_span": t_vec, "energy_32": H_32_vec, "energy_10": H_10_vec, "power": Hdot_vec, \
                "flow": bdflow_vec, "flow_ex": bdflow_ex_vec, "err_p3": err_p_3, "err_u1": err_u_1, "err_p0": err_p_0, "err_u2": err_u_2, \
                "err_p30": err_p30, "err_u12": err_u12}

    return dict_res

# n_elem = 10
# pol_deg = 2
#
# n_time = 1000
# t_fin = 1
#
# results = compute_err(n_elem, n_time, pol_deg, t_fin)
#
# t_vec = results["t_span"]
# Hdot_vec = results["power"]
# bdflow_vec = results["flow"]
# bdflow_ex_vec = results["flow_ex"]
#
# H_32 = results["energy_32"]
# H_10 = results["energy_10"]
#
# plt.figure()
# plt.plot(t_vec, H_32, 'r', label=r'$H_{32}$')
# plt.plot(t_vec, H_10, 'b', label=r'$H_{10}$')
# plt.xlabel(r'Time [s]')
# plt.title(r' Mixed energy')
# plt.legend()
#
# plt.figure()
# plt.plot(t_vec, bdflow_vec, 'r', label=r'bd flow')
# plt.plot(t_vec, bdflow_ex_vec, 'b', label=r'bd flow ex')
# plt.xlabel(r'Time [s]')
# plt.title(r'Boundary flow')
# plt.legend()
#
#
# plt.figure()
# plt.plot(t_vec, Hdot_vec - bdflow_vec, 'r--', label=r'Energy residual')
# plt.xlabel(r'Time [s]')
# plt.title(r'Energy residual')
# plt.legend()
#
# diffH_L2Hdiv = np.diff(H_32)
# diffH_H1Hcurl = np.diff(H_10)
# Delta_t = np.diff(t_vec)
# int_bdflow = np.zeros((n_time, ))
#
# for i in range(n_time):
#     int_bdflow[i] = 0.5*Delta_t[i]*(bdflow_vec[i+1] + bdflow_vec[i])
#
# plt.figure()
# plt.plot(t_vec[1:], diffH_L2Hdiv, 'ro', label=r'$\Delta H_{32}$')
# plt.plot(t_vec[1:], diffH_H1Hcurl, 'b--', label=r'$\Delta H_{10}$')
# plt.plot(t_vec[1:], int_bdflow, '*-', label=r'Bd flow int')
# plt.xlabel(r'Time [s]')
# plt.title(r'Energy balance')
# plt.legend()
#
# plt.show()
