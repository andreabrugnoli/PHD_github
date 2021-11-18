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
    """Compute the numerical solution of the wave equation with mixed formulations

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
    mesh = RectangleMesh(n_el, n_el, 1, 1/2, quadrilateral=False)
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

    om_x = 1
    om_y = 1

    om_t = np.sqrt(om_x ** 2 + om_y ** 2)
    phi_x = 0
    phi_y = 0
    phi_t = 0

    t = Constant(0.0)
    t_1 = Constant(0.0)

    ft = 2 * sin(om_t * t + phi_t) + 3 * cos(om_t * t + phi_t)
    dft_t = om_t * (2 * cos(om_t * t + phi_t) - 3 * sin(om_t * t + phi_t))  # diff(dft_t, t)

    ft1 = 2 * sin(om_t * t_1 + phi_t) + 3 * cos(om_t * t_1 + phi_t)
    dft1_t1 = om_t * (2 * cos(om_t * t_1 + phi_t) - 3 * sin(om_t * t_1 + phi_t))  # diff(dft_t, t)

    gxy = cos(om_x * x + phi_x) * sin(om_y * y + phi_y)

    dgxy_x = - om_x * sin(om_x * x + phi_x) * sin(om_y * y + phi_y)
    dgxy_y = om_y * cos(om_x * x + phi_x) * cos(om_y * y + phi_y)

    # w_ex = gxy * ft

    p_ex = gxy * dft_t
    p_ex1 = gxy * dft1_t1

    u_ex = as_vector([dgxy_x * ft,
                      dgxy_y * ft])  # grad(gxy)

    u_ex1 = as_vector([dgxy_x * ft1,
                       dgxy_y * ft1])

    p0_3 = interpolate(p_ex, V_3)
    u0_2 = interpolate(u_ex, V_2)

    u0_1 = interpolate(u_ex, V_1)
    p0_0 = interpolate(p_ex, V_0)

    in_cond_32 = Function(V_32)
    in_cond_32.sub(0).assign(p0_3)
    in_cond_32.sub(1).assign(u0_2)

    in_cond_10 = Function(V_10)
    in_cond_10.sub(0).assign(u0_1)
    in_cond_10.sub(1).assign(p0_0)


    if bd_cond=="D":
        bc_D = DirichletBC(V_10.sub(1), p_ex1, "on_boundary")
        bc_N = None
    elif bd_cond=="N":
        bc_N = DirichletBC(V_32.sub(1), u_ex1, "on_boundary")
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

    en_32 = Function(V_32, name="e_32 n")
    en_32.assign(e0_32)
    en1_32 = Function(V_32, name="e_32 n+1")

    en_10 = Function(V_10, name="e_10 n")
    en_10.assign(e0_10)
    en1_10 = Function(V_10, name="e_10 n+1")

    err_p0 = Function(V_0, name="p_0")
    err_p3 = Function(V_3, name="p_3")

    pn_3, un_2 = en_32.split()
    un_1, pn_0 = en_10.split()

    Hn_32 = 0.5 * (inner(pn_3, pn_3) * dx + inner(un_2, un_2) * dx)
    Hn_10 = 0.5 * (inner(pn_0, pn_0) * dx + inner(un_1, un_1) * dx)

    Hn_ex1 = 0.5 * (inner(p_ex, p_ex) * dx(domain=mesh) + inner(u_ex, u_ex) * dx(domain=mesh))

    # dHn_t_ex = p_ex*diff(p_ex, t) * dx(domain=mesh) # + diff(inner(u_ex, u_ex), t) * dx(domain=mesh)

    a1 = (0.5 - np.sin(0.5) * np.cos(0.5)) * (1 + np.sin(1) * np.cos(1))
    a2 = ((0.5 + np.sin(0.5) * np.cos(0.5)) * (1 + np.sin(1) * np.cos(1)) + \
          (0.5 - np.sin(0.5) * np.cos(0.5)) * (1 - np.sin(1) * np.cos(1)))
    Hn_ex2 = 1 / 8 * (dft_t) ** 2 * a1 + 1 / 8 * (ft) ** 2 * a2
    dHn_t_ex2 = diff(Hn_ex2, t)

    Hdot_n = div(un_2) * pn_0 * dx + inner(grad(pn_0), un_2) * dx
    bdflow_n = pn_0 * dot(un_2, n_ver) * ds

    bdflow_ex_n = p_ex * dot(u_ex, n_ver) * ds(domain=mesh)

    H_32_vec = np.zeros((1 + n_t,))
    H_10_vec = np.zeros((1 + n_t,))
    H_ex_vec1 = np.zeros((1 + n_t,))
    H_ex_vec2 = np.zeros((1 + n_t,))

    bdflow_vec = np.zeros((1 + n_t,))
    bdflow_ex_vec = np.zeros((1 + n_t,))

    Hdot_vec = np.zeros((1 + n_t,))
    Hdot_ex_vec2 = np.zeros((1 + n_t,))

    errL2_p_3_vec = np.zeros((1 + n_t,))
    errL2_u_1_vec = np.zeros((1 + n_t,))

    errL2_p_0_vec = np.zeros((1 + n_t,))
    errL2_u_2_vec = np.zeros((1 + n_t,))

    errHcurl_u_1_vec = np.zeros((1 + n_t,))
    errH1_p_0_vec = np.zeros((1 + n_t,))
    errHdiv_u_2_vec = np.zeros((1 + n_t,))

    err_p30_vec = np.zeros((1 + n_t,))
    err_u12_vec = np.zeros((1 + n_t,))

    H_32_vec[0] = assemble(Hn_32)
    H_10_vec[0] = assemble(Hn_10)
    H_ex_vec1[0] = assemble(Hn_ex1)
    H_ex_vec2[0] = Hn_ex2

    Hdot_vec[0] = assemble(Hdot_n)
    bdflow_vec[0] = assemble(bdflow_n)
    bdflow_ex_vec[0] = assemble(bdflow_ex_n)
    Hdot_ex_vec2[0] = dHn_t_ex2

    errL2_p_3_vec[0] = errornorm(p_ex, p0_3, norm_type="L2")
    errL2_u_1_vec[0] = errornorm(u_ex, u0_1, norm_type="L2")
    errL2_p_0_vec[0] = errornorm(p_ex, p0_0, norm_type="L2")
    errL2_u_2_vec[0] = errornorm(u_ex, u0_2, norm_type="L2")

    errHcurl_u_1_vec[0] = errornorm(u_ex, u0_1, norm_type="Hcurl")
    errH1_p_0_vec[0] = errornorm(p_ex, p0_0, norm_type="H1")
    errHdiv_u_2_vec[0] = errornorm(u_ex, u0_2, norm_type="Hdiv")

    err_p30_vec[0] = np.sqrt(assemble(inner(p0_3 - p0_0, p0_3 - p0_0) * dx))
    err_u12_vec[0] = np.sqrt(assemble(inner(u0_2 - u0_1, u0_2 - u0_1) * dx))

    # diff0_p30 = project(p0_3 - p0_0, V_3)
    # diff0_u12 = project(u0_2 - u0_1, V_1)
    #
    # err_p30_vec[0] = errornorm(Constant(0), diff0_p30, norm_type="L2")
    # err_u12_vec[0] = errornorm(Constant((0.0, 0.0)), diff0_u12, norm_type="L2")

    ## Settings of intermediate variables and matrices for the 2 linear systems

    print("Assemble of the A matrices")
    print("==============")

    a_form10 = m_form10(v_1, u_1, v_0, p_0) - 0.5*dt*j_form10(v_1, u_1, v_0, p_0)
    a_form32 = m_form32(v_3, p_3, v_2, u_2) - 0.5*dt*j_form32(v_3, p_3, v_2, u_2)



    print("Computation of the solution")
    print("==============")

    for ii in tqdm(range(n_t)):

        ## Integration of 10 system
        t_1.assign(float(t) + float(dt))
        u_ex_mid = 0.5*(u_ex + u_ex1)

        A_10 = assemble(a_form10, bcs=bc_D, mat_type='aij')
        A_32 = assemble(a_form32, bcs=bc_N, mat_type='aij')

        b_form10 = m_form10(v_1, un_1, v_0, pn_0) + dt*(0.5*j_form10(v_1, un_1, v_0, pn_0) + bdflow10(v_0, u_ex_mid))
        # b_form10 = m_form10(v_1, un_1, v_0, pn_0) + dt*(0.5*j_form10(v_1, un_1, v_0, pn_0))
        b_vec10 = assemble(b_form10)

        solve(A_10, en1_10, b_vec10, solver_parameters=params)

        ## Integration of 32 system

        p_ex_mid = 0.5*(p_ex + p_ex1)

        b_form32 = m_form32(v_3, pn_3, v_2, un_2) + dt*(0.5*j_form32(v_3, pn_3, v_2, un_2) + bdflow32(v_2, p_ex_mid))
        # b_form32 = m_form32(v_3, pn_3, v_2, un_2) + dt*(0.5*j_form32(v_3, pn_3, v_2, un_2))
        b_vec32 = assemble(b_form32)

        solve(A_32, en1_32, b_vec32, solver_parameters=params)

        en_32.assign(en1_32)
        en_10.assign(en1_10)

        un_1, pn_0 = en_10.split()
        pn_3, un_2 = en_32.split()

        Hdot_vec[ii + 1] = assemble(Hdot_n)
        bdflow_vec[ii + 1] = assemble(bdflow_n)

        H_32_vec[ii + 1] = assemble(Hn_32)
        H_10_vec[ii + 1] = assemble(Hn_10)

        t.assign(float(t) + float(dt))

        H_ex_vec1[ii + 1] = assemble(Hn_ex1)
        H_ex_vec2[ii + 1] = Hn_ex2

        bdflow_ex_vec[ii + 1] = assemble(bdflow_ex_n)
        Hdot_ex_vec2[ii + 1] = dHn_t_ex2

        # print(bdflow_ex_vec[ii+1])
        errL2_p_3_vec[ii + 1] = errornorm(p_ex, pn_3, norm_type="L2")
        errL2_u_1_vec[ii + 1] = errornorm(u_ex, un_1, norm_type="L2")
        errL2_p_0_vec[ii + 1] = errornorm(p_ex, pn_0, norm_type="L2")
        errL2_u_2_vec[ii + 1] = errornorm(u_ex, un_2, norm_type="L2")

        errHcurl_u_1_vec[ii + 1] = errornorm(u_ex, un_1, norm_type="Hcurl")
        errH1_p_0_vec[ii + 1] = errornorm(p_ex, pn_0, norm_type="H1")
        errHdiv_u_2_vec[ii + 1] = errornorm(u_ex, un_2, norm_type="Hdiv")

        err_p30_vec[ii + 1] = np.sqrt(assemble(inner(pn_3 - pn_0, pn_3 - pn_0) * dx))
        err_u12_vec[ii + 1] = np.sqrt(assemble(inner(un_2 - un_1, un_2 - un_1) * dx))

        # diffn_p30 = project(pn_3 - pn_0, V_3)
        # diffn_u12 = project(un_2 - un_1, V_1)
        #
        # err_p30_vec[ii+1] = errornorm(Constant(0), diffn_p30, norm_type="L2")
        # err_u12_vec[ii+1] = errornorm(Constant((0.0, 0.0)), diffn_u12, norm_type="L2")
    #
    #     p_3P[ii + 1] = pn_3.at(Ppoint)
    #     p_0P[ii + 1] = pn_0.at(Ppoint)
    #
    # err_p3.assign(pn_3 - interpolate(p_ex, V_3))
    # err_p0.assign(pn_0 - interpolate(p_ex, V_0))
    #
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
    #
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
    #
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
    # plt.plot(t_vec, om_t * np.cos(om_x * Ppoint[0] + phi_x) * np.sin(om_y * Ppoint[1] + phi_y) \
    #          * np.cos(om_t * t_vec + phi_t), 'g-', label=r'exact $p$')
    # plt.xlabel(r'Time [s]')
    # plt.title(r'$p$ at ' + str(Ppoint))
    # plt.legend()

    # err_p_3 = np.sqrt(np.sum(float(dt) * np.power(err_p_3_vec, 2)))
    # err_u_1 = np.sqrt(np.sum(float(dt) * np.power(err_u_1_vec, 2)))
    # err_p_0 = np.sqrt(np.sum(float(dt) * np.power(err_p_0_vec, 2)))
    # err_u_2 = np.sqrt(np.sum(float(dt) * np.power(err_u_2_vec, 2)))
    #
    # err_p_3 = max(err_p_3_vec)
    # err_u_1 = max(err_u_1_vec)
    #
    # err_p_0 = max(err_p_0_vec)
    # err_u_2 = max(err_u_2_vec)
    #
    # err_p30 = max(err_p30_vec)
    # err_u12 = max(err_u12_vec)

    errL2_p_3 = errL2_p_3_vec[-1]
    errL2_u_1 = errL2_u_1_vec[-1]

    errL2_p_0 = errL2_p_0_vec[-1]
    errL2_u_2 = errL2_u_2_vec[-1]

    errHcurl_u_1 = errHcurl_u_1_vec[-1]

    errH1_p_0 = errH1_p_0_vec[-1]
    errHdiv_u_2 = errHdiv_u_2_vec[-1]

    err_p30 = err_p30_vec[-1]
    err_u12 = err_u12_vec[-1]

    dict_res = {"t_span": t_vec, "energy_32": H_32_vec, "energy_ex1": H_ex_vec1, "energy_ex2": H_ex_vec2,
                "energy_10": H_10_vec, "power": Hdot_vec, "power_ex": Hdot_ex_vec2, "flow": bdflow_vec, \
                "flow_ex": bdflow_ex_vec, "err_p3": errL2_p_3, \
                "err_u1": [errL2_u_1, errHcurl_u_1], "err_p0": [errL2_p_0, errH1_p_0], \
                "err_u2": [errL2_u_2, errHdiv_u_2], "err_p30": err_p30, "err_u12": err_u12}

    return dict_res
#
# n_elem = 5
# pol_deg = 1
#
# n_time = 100
# t_fin = 1
#
# results = compute_err(n_elem, n_time, pol_deg, t_fin)
#
# t_vec = results["t_span"]
# Hdot_vec = results["power"]
# Hdot_ex_vec = results["power_ex"]
#
# bdflow_vec = results["flow"]
# bdflow_ex_vec = results["flow_ex"]
#
# H_32 = results["energy_32"]
# H_10 = results["energy_10"]
# H_ex1 = results["energy_ex1"]
# H_ex2 = results["energy_ex2"]
#
# errL2_p3 = results["err_p3"]
# errL2_u1, errHcurl_u1 = results["err_u1"]
# errL2_p0, errH1_p0 = results["err_p0"]
# errL2_u2, errHdiv_u2 = results["err_u2"]
#
# print("Error L2 p3: " + str(errL2_p3))
#
# print("Error L2 u1: " + str(errL2_u1))
# print("Error Hcurl u1: " + str(errHcurl_u1))
#
# print("Error L2 p0: " + str(errL2_p0))
# print("Error H1 p0: " + str(errH1_p0))
#
# print("Error L2 u2: " + str(errL2_u2))
# print("Error Hdiv u2: " + str(errHdiv_u2))
# plt.figure()
# plt.plot(t_vec, H_32, 'r-.', label=r'$H_{32}$')
# plt.plot(t_vec, H_10, 'b--', label=r'$H_{10}$')
# plt.plot(t_vec, H_ex1, '*', label=r'H Exact1')
# plt.plot(t_vec, H_ex2, 'o', label=r'H Exact2')
# plt.xlabel(r'Time [s]')
# plt.title(r'Energies')
# plt.legend()
#
# plt.figure()
# plt.plot(t_vec, bdflow_vec, 'r-.', label=r'bd flow')
# plt.plot(t_vec, bdflow_ex_vec, 'b--', label=r'bd flow ex')
# plt.plot(t_vec, Hdot_vec, 'o', label=r'Hdot')
# plt.plot(t_vec, Hdot_ex_vec, '+', label=r'Hdot ex')
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
# plt.show()
#
# # diffH_L2Hdiv = np.diff(H_32)
# # diffH_H1Hcurl = np.diff(H_10)
# # Delta_t = np.diff(t_vec)
# # int_bdflow = np.zeros((n_time, ))
# #
# # for i in range(n_time):
# #     int_bdflow[i] = 0.5*Delta_t[i]*(bdflow_vec[i+1] + bdflow_vec[i])
# #
# # plt.figure()
# # plt.plot(t_vec[1:], diffH_L2Hdiv, 'ro.', label=r'$\Delta H_{32}$')
# # plt.plot(t_vec[1:], diffH_H1Hcurl, 'b--', label=r'$\Delta H_{10}$')
# # plt.plot(t_vec[1:], int_bdflow, '*-.', label=r'Bd flow int')
# # plt.xlabel(r'Time [s]')
# # plt.title(r'Energy balance')
# # plt.legend()
# #
# # plt.show()