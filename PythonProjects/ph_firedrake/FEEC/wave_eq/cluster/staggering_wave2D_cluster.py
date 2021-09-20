## This is a first test to solve the wave equation in 3d domains using the dual field method
## A staggering method is used for the time discretization

import os
os.environ["OMP_NUM_THREADS"] = "1"

# import numpy as np
from firedrake import *
import matplotlib.pyplot as plt
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

    L = 1/2
    mesh = RectangleMesh(n_el, n_el, 1, 1/2, quadrilateral=False)
    n_ver = FacetNormal(mesh)

    # P_0 = FiniteElement("CG", quadrilateral, deg)
    # P_1 = FiniteElement("RTCE", quadrilateral, deg)
    # P_2 = FiniteElement("RTCF", quadrilateral, deg)
    # P_3 = FiniteElement("DG", quadrilateral, deg - 1)

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

    dt = Constant(t_fin / n_t)

    params = {"mat_type": "aij",
              "snes_type": "ksponly",
              "ksp_type": "preonly",
              "pc_type": "lu"}

    t_vec = np.linspace(0, n_t * float(dt), 1 + n_t)

    t = Constant(0.0)
    t_10 = Constant(0)
    t_32 = Constant(dt/2)

    ft = 2 * sin(om_t * t + phi_t) + 3 * cos(om_t * t + phi_t)
    dft_t = om_t * (2 * cos(om_t * t + phi_t) - 3 * sin(om_t * t + phi_t))  # diff(dft_t, t)

    ft10 = 2 * sin(om_t * t_10 + phi_t) + 3 * cos(om_t * t_10 + phi_t)
    dft_t10 = om_t * (2 * cos(om_t * t_10 + phi_t) - 3 * sin(om_t * t_10 + phi_t))  # diff(dft_t, t)

    ft32 = 2 * sin(om_t * t_32 + phi_t) + 3 * cos(om_t * t_32 + phi_t)
    dft_t32 = om_t * (2 * cos(om_t * t_32 + phi_t) - 3 * sin(om_t * t_32 + phi_t))  # diff(dft_t, t)

    gxy = cos(om_x * x + phi_x) * sin(om_y * y + phi_y)

    dgxy_x = - om_x * sin(om_x * x + phi_x) * sin(om_y * y + phi_y)
    dgxy_y = om_y * cos(om_x * x + phi_x) * cos(om_y * y + phi_y)

    # w_ex = gxy * ft

    p_ex = gxy * dft_t
    u_ex = as_vector([dgxy_x * ft,
                      dgxy_y * ft])  # grad(gxy)

    p_ex10 = gxy * dft_t10
    u_ex32 = as_vector([dgxy_x * ft32,
                        dgxy_y * ft32])

    p0_3 = interpolate(p_ex, V_3)
    u0_2 = interpolate(u_ex, V_2)
    # u0_2 = project(u_ex, V_2)

    u0_1 = interpolate(u_ex, V_1)
    # u0_1 = project(u_ex, V_1)
    p0_0 = interpolate(p_ex, V_0)

    if bd_cond=="D":
        bc_D= DirichletBC(V_10.sub(1), p_ex10, "on_boundary")
        bc_N = None
    elif bd_cond=="N":
        bc_N = DirichletBC(V_32.sub(1), u_ex32, "on_boundary")

        bc_D = None
    else:
        bc_D = [DirichletBC(V_10.sub(1), p_ex10, 1), \
                DirichletBC(V_10.sub(1), p_ex10, 3)]

        bc_N = [DirichletBC(V_32.sub(1), u_ex32, 2), \
                DirichletBC(V_32.sub(1), u_ex32, 4)]

    e0_32 = Function(V_32, name="e_32 initial")
    e0_10 = Function(V_10, name="e_10 initial")

    e0_32.sub(0).assign(p0_3)
    e0_32.sub(1).assign(u0_2)

    e0_10.sub(0).assign(u0_1)
    e0_10.sub(1).assign(p0_0)

    enmid_32 = Function(V_32, name="e_32 n+1/2")
    enmid1_32 = Function(V_32, name="e_32 n+3/2")

    en_32 = Function(V_32, name="e_32 n")
    en_32.assign(e0_32)
    en1_32 = Function(V_32, name="e_32 n+1")

    en_10 = Function(V_10, name="e_10 n")
    en_10.assign(e0_10)
    en1_10 = Function(V_10, name="e_10 n+1")

    pn_3, un_2 = en_32.split()
    un_1, pn_0 = en_10.split()


    print("First explicit step")
    print("==============")

    a0_form32 = m_form32(v_3, p_3, v_2, u_2)
    b0_form32 = m_form32(v_3, p0_3, v_2, u0_2) + dt / 2 * (j_form32(v_3, p0_3, v_2, u0_2) + bdflow32(v_2, p0_0))
    # b0_form32 = m_form32(v_3, p0_3, v_2, u0_2) + dt / 2 * (j_form32(v_3, p0_3, v_2, u0_2))
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
    # else:
    #     bc_ND_first =  [DirichletBC(V_3210.sub(3), p_ex, 1), \
    #                     DirichletBC(V_3210.sub(3), p_ex, 2), \
    #                     DirichletBC(V_3210.sub(1), u_ex, 3), \
    #                     DirichletBC(V_3210.sub(1), u_ex, 4)]
    #     A0 = assemble(a0_form, bcs=bc_ND_first, mat_type='aij')
    #
    # b0 = assemble(b0_form)
    #
    # solve(A0, en1, b0, solver_parameters=params)
    # en1_32.sub(0).assign(en1.split()[0])
    # en1_32.sub(1).assign(en1.split()[1])
    # enmid_32.assign(0.5*(en_32 + en1_32))

    ## Settings of intermediate variables and matrices for the 2 linear systems

    a_form10 = m_form10(v_1, u_1, v_0, p_0) - 0.5*dt*j_form10(v_1, u_1, v_0, p_0)
    a_form32 = m_form32(v_3, p_3, v_2, u_2) - 0.5*dt*j_form32(v_3, p_3, v_2, u_2)

    print("Computation of the solution with n elem " + str(n_el) + " n time " + str(n_t) + " deg " + str(deg))
    print("==============")

    for ii in tqdm(range(n_t)):

        ## Integration of 10 system using unmid_2

        t_10.assign(float(t) + float(dt))
        A_10 = assemble(a_form10, bcs=bc_D, mat_type='aij')

        pnmid_3, unmid_2 = enmid_32.split()
        b_form10 = m_form10(v_1, un_1, v_0, pn_0) + dt*(0.5*j_form10(v_1, un_1, v_0, pn_0) + bdflow10(v_0, unmid_2))
        # b_form10 = m_form10(v_1, un_1, v_0, pn_0) + dt*0.5*j_form10(v_1, un_1, v_0, pn_0)

        b_vec10 = assemble(b_form10)

        solve(A_10, en1_10, b_vec10, solver_parameters=params)

        un1_1, pn1_0 = en1_10.split()

        ## Integration of 32 system using pn1_0

        t_32.assign(float(t) + 3/2*float(dt))
        A_32 = assemble(a_form32, bcs=bc_N, mat_type='aij')

        b_form32 = m_form32(v_3, pnmid_3, v_2, unmid_2) + dt*(0.5*j_form32(v_3, pnmid_3, v_2, unmid_2) \
                                                              + bdflow32(v_2, pn1_0))
        # b_form32 = m_form32(v_3, pnmid_3, v_2, unmid_2) + dt * 0.5 * j_form32(v_3, pnmid_3, v_2, unmid_2)
        b_vec32 = assemble(b_form32)

        solve(A_32, enmid1_32, b_vec32, solver_parameters=params)

        # If it does not work split and then assign
        # pnmid_3, unmid_2 = enmid_32.split()
        # pnmid1_3, unmid1_2 = enmid1_32.split()
        #
        # en1_32.sub(0).assign(0.5*(pnmid_3 + pnmid1_3))
        # en1_32.sub(1).assign(0.5*(unmid_2 + unmid1_2))
        #
        # en_32.assign(en1_32)

        en_32.assign(0.5 * (enmid_32 + enmid1_32))

        en_10.assign(en1_10)
        enmid_32.assign(enmid1_32)

        un_1, pn_0 = en_10.split()
        pn_3, un_2 = en_32.split()

        t.assign(float(t) + float(dt))

    errL2_p_3 = errornorm(p_ex, pn_3, norm_type="L2")
    errL2_u_1 = errornorm(u_ex, un_1, norm_type="L2")
    errL2_p_0 = errornorm(p_ex, pn_0, norm_type="L2")
    errL2_u_2 = errornorm(u_ex, un_2, norm_type="L2")

    errHcurl_u_1 = errornorm(u_ex, un_1, norm_type="Hcurl")
    errH1_p_0 = errornorm(p_ex, pn_0, norm_type="H1")
    errHdiv_u_2 = errornorm(u_ex, un_2, norm_type="Hdiv")

    err_p30 = np.sqrt(assemble(inner(pn_3 - pn_0, pn_3 - pn_0) * dx))
    err_u12 = np.sqrt(assemble(inner(un_2 - un_1, un_2 - un_1) * dx))

    dict_res = {"err_p3": errL2_p_3, "err_u1": [errL2_u_1, errHcurl_u_1], "err_p0": [errL2_p_0, errH1_p_0], \
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
