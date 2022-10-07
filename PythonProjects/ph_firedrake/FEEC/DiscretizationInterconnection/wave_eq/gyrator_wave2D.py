## This is a first test to solve the wave equation in 3d domains using the dual field method
## A staggering method is used for the time discretization

import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from firedrake import *
import matplotlib.pyplot as plt
from tools_plotting import setup
from tqdm import tqdm
# from time import sleep
from matplotlib.ticker import FormatStrFormatter
import pickle


bc_case = "_DN"
geo_case = "_3D"


def compute_err(n_el, n_t, deg=1, t_fin=1, bd_cond="D"):
    """Compute the numerical solution of the wave equation with a DG method based on interconnection

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

    def m_form10(v_1, u_1, v_0, p_0):
        m_form = inner(v_1, u_1) * dx + inner(v_0, p_0) * dx

        return m_form

    def m_form(v_0, p_0, v_1, u_1, v_3, p_3, v_2, u_2):

        return m_form10(v_1, u_1, v_0, p_0) + m_form32(v_3, p_3, v_2, u_2)

    def j_form32(v_3, p_3, v_2, u_2):
        j_form = dot(v_3, div(u_2)) * dx - dot(div(v_2), p_3) * dx

        return j_form

    def j_form10(v_1, u_1, v_0, p_0):
        j_form = dot(v_1, grad(p_0)) * dx - dot(grad(v_0), u_1) * dx

        return j_form

    def j_int(v2_b, p0_b, v0_b, u2_b):
        j_int_20 = (dot(v2_b('+'), n_ver('+')) * p0_b('-') + dot(v2_b('-'), n_ver('-')) * p0_b('+')) * dS
        j_int_02 = - (v0_b('+') * dot(u2_b('-'), n_ver('-')) + v0_b('-') * dot(u2_b('+'), n_ver('+'))) * dS

        return j_int_20 + j_int_02

    def j_form(v_0, p_0, v_1, u_1, v_3, p_3, v_2, u_2):

        return j_form10(v_1, u_1, v_0, p_0) + j_form32(v_3, p_3, v_2, u_2) + j_int(v_2, p_0, v_0, u_2)

    # def j_int_p(v2_b, p0_b, v0_b, u2_b):
    #     j_int_p = (dot(v2_b('+'), n_ver('+')) * p0_b('-') - v0_b('-') * dot(u2_b('+'), n_ver('+'))) * dS
    #
    #     return j_int_p
    #
    # def j_int_d(v2_b, p0_b, v0_b, u2_b):
    #     j_int_d = (-v0_b('+') * dot(u2_b('-'), n_ver('-')) + dot(v2_b('-'), n_ver('-')) * p0_b('+')) * dS
    #
    #     return j_int_d

    def bdflow0(v_2, p_0):
        b_form = dot(v_2, n_ver) * p_0 * ds

        return b_form

    def bdflow2(v_0, u_2):
        b_form = v_0 * dot(u_2, n_ver) * ds

        return b_form

    def bdflow(v_2, p_0, v_0, u_2):

        return bdflow0(v_2, p_0) + bdflow2(v_0, u_2)


    L = 1/2

    mesh = BoxMesh(n_el, n_el, n_el, 1, 1/2, 1/2)
    n_ver = FacetNormal(mesh)

    P0 = FiniteElement("CG", tetrahedron, deg)
    P1 = FiniteElement("N1curl", tetrahedron, deg)
    # P1 = FiniteElement("N1curl", tetrahedron, deg)
    P2 = FiniteElement("RT", tetrahedron, deg)
    # Integral evaluation on Raviart-Thomas and NED for deg=3 completely freezes interpolation
    # P2 = FiniteElement("RT", tetrahedron, deg, variant='integral')
    P3 = FiniteElement("DG", tetrahedron, deg - 1)

    P0_b = BrokenElement(P0)
    P1_b = BrokenElement(P1)
    P2_b = BrokenElement(P2)
    P3_b = BrokenElement(P3)

    V0_b = FunctionSpace(mesh, P0_b)
    V1_b = FunctionSpace(mesh, P1_b)
    V2_b = FunctionSpace(mesh, P2_b)
    V3_b = FunctionSpace(mesh, P3_b)

    V0132_b = V0_b * V1_b * V3_b * V2_b

    # print(V0_b.dim())
    # print(V1_b.dim())
    # print(V2_b.dim())
    # print(V3_b.dim())

    v0132_b = TestFunction(V0132_b)
    v0_b, v1_b, v3_b, v2_b = split(v0132_b)

    e0132_b = TrialFunction(V0132_b)
    p0_b, u1_b, p3_b, u2_b = split(e0132_b)

    dx = Measure('dx')
    ds = Measure('ds')
    dS = Measure('dS')

    x, y, z = SpatialCoordinate(mesh)

    om_x = 1
    om_y = 1
    om_z = 1

    om_t = np.sqrt(om_x ** 2 + om_y ** 2 + om_z ** 2)
    phi_x = 0
    phi_y = 0
    phi_z = 0
    phi_t = 0

    dt = Constant(t_fin / n_t)

    params = {"mat_type": "aij",
              "snes_type": "ksponly",
              "ksp_type": "preonly",
              "pc_type": "lu"}

    t_vec = np.linspace(0, n_t * float(dt), 1 + n_t)

    t = Constant(0.0)
    t_1 = Constant(dt)

    ft = 2 * sin(om_t * t + phi_t) + 3 * cos(om_t * t + phi_t)
    dft = om_t * (2 * cos(om_t * t + phi_t) - 3 * sin(om_t * t + phi_t))  # diff(dft_t, t)

    ft_1 = 2 * sin(om_t * t_1 + phi_t) + 3 * cos(om_t * t_1 + phi_t)
    dft_1 = om_t * (2 * cos(om_t * t_1 + phi_t) - 3 * sin(om_t * t_1 + phi_t))  # diff(dft_t, t)

    gxyz = cos(om_x * x + phi_x) * sin(om_y * y + phi_y) * sin(om_z * z + phi_z)

    dgxyz_x = - om_x * sin(om_x * x + phi_x) * sin(om_y * y + phi_y) * sin(om_z * z + phi_z)
    dgxyz_y = om_y * cos(om_x * x + phi_x) * cos(om_y * y + phi_y) * sin(om_z * z + phi_z)
    dgxyz_z = om_z * cos(om_x * x + phi_x) * sin(om_y * y + phi_y) * cos(om_z * z + phi_z)

    grad_gxyz = as_vector([dgxyz_x,
                           dgxyz_y,
                           dgxyz_z]) # grad(gxyz)

    p_ex = gxyz * dft
    u_ex = grad_gxyz * ft

    p_ex_1 = gxyz * dft_1
    u_ex_1 = grad_gxyz * ft_1

    u_ex_mid = 0.5 * (u_ex + u_ex_1)
    p_ex_mid = 0.5 * (p_ex + p_ex_1)

    p0_3_b = project(p_ex, V3_b)
    u0_2_b = project(u_ex, V2_b)
    u0_1_b = project(u_ex, V1_b)
    p0_0_b = project(p_ex, V0_b)


    if bd_cond == "D":
        bc_D = [DirichletBC(V0132_b.sub(0), p_ex_1, "on_boundary")]
        bc_D_nat = []

        bc_N = []
        bc_N_nat = [DirichletBC(V0132_b.sub(3), u_ex_1, "on_boundary")]

    elif bd_cond == "N":
        bc_N = [DirichletBC(V0132_b.sub(3), u_ex_1, "on_boundary")]
        bc_N_nat = []

        bc_D = []
        bc_D_nat = [DirichletBC(V0132_b.sub(0), p_ex_1, "on_boundary")]
    else:
        bc_D = [DirichletBC(V0132_b.sub(0), p_ex_1, 1), \
                DirichletBC(V0132_b.sub(0), p_ex_1, 3),
                DirichletBC(V0132_b.sub(0), p_ex_1, 5)]

        bc_D_nat = [DirichletBC(V0132_b.sub(0), p_ex_1, 2), \
                    DirichletBC(V0132_b.sub(0), p_ex_1, 4), \
                    DirichletBC(V0132_b.sub(0), p_ex_1, 6)]

        bc_N = [DirichletBC(V0132_b.sub(3), u_ex_1, 2), \
                DirichletBC(V0132_b.sub(3), u_ex_1, 4),
                DirichletBC(V0132_b.sub(3), u_ex_1, 6)]

        bc_N_nat = [DirichletBC(V0132_b.sub(3), u_ex_1, 1), \
                    DirichletBC(V0132_b.sub(3), u_ex_1, 3), \
                    DirichletBC(V0132_b.sub(3), u_ex_1, 5)]

    bcs = bc_D + bc_N

    dofsV0_D = []
    dofsV2_D = []

    if bc_D is not None:
        for ii in range(len(bc_D)):
            nodesV0_D = bc_D[ii].nodes
            nodesV2_D = V0_b.dim() + V1_b.dim() + V3_b.dim() + bc_N_nat[ii].nodes

            dofsV0_D = dofsV0_D + list(nodesV0_D)
            dofsV2_D = dofsV2_D + list(nodesV2_D)

    dofsV0_D = list(set(dofsV0_D))
    dofsV2_D = list(set(dofsV2_D))

    # print("dofs on Gamma_D for 10")
    # print(dofs10_D)
    # print("dofs on Gamma_D for 32")
    # print(dofs32_D)

    dofsV0_N = []
    dofsV2_N = []

    if bc_N is not None:
        for ii in range(len(bc_N)):
            nodesV2_N = V0_b.dim() + V1_b.dim() + V3_b.dim() + bc_N[ii].nodes
            nodesV0_N = bc_D_nat[ii].nodes

            dofsV2_N = dofsV2_N + list(nodesV2_N)
            dofsV0_N = dofsV0_N + list(nodesV0_N)

    dofsV2_N = list(set(dofsV2_N))
    dofsV0_N = list(set(dofsV0_N))

    for element in dofsV0_D:
        if element in dofsV0_N:
            dofsV0_N.remove(element)

    for element in dofsV2_N:
        if element in dofsV2_D:
            dofsV2_D.remove(element)

    # print("dofs on Gamma_N for 10")
    # print(dofs10_N)
    # print("dofs on Gamma_N for 32")
    # print(dofs32_N)

    Ppoint = (L/5, L/5, L/5)

    p_0P = np.zeros((1+n_t,))
    p_0P[0] = project(p_ex, V0_b).at(Ppoint)

    p_3P = np.zeros((1+n_t, ))
    p_3P[0] = project(p_ex, V3_b).at(Ppoint)

    e0_0132_b = Function(V0132_b, name="e at t=0")

    e0_0132_b.sub(0).assign(p0_0_b)
    e0_0132_b.sub(1).assign(u0_1_b)
    e0_0132_b.sub(2).assign(p0_3_b)
    e0_0132_b.sub(3).assign(u0_2_b)

    en_0132_b = Function(V0132_b, name="e at t=n")
    en_0132_b.assign(e0_0132_b)

    enmid_0132_b = Function(V0132_b, name="e at t=n+1/2")
    en1_0132_b = Function(V0132_b, name="e at t=n+1")

    pn_0_b, un_1_b, pn_3_b, un_2_b = en_0132_b.split()

    pnmid_0_b, unmid_1_b, pnmid_3_b, unmid_2_b = enmid_0132_b.split()
    pn1_0_b, un1_1_b, pn1_3_b, un1_2_b = en1_0132_b.split()

    Hn_32 = 0.5 * (inner(pn_3_b, pn_3_b) * dx + inner(un_2_b, un_2_b) * dx)
    Hn_10 = 0.5 * (inner(pn_0_b, pn_0_b) * dx + inner(un_1_b, un_1_b) * dx)

    Hn_31 = 0.5 * (inner(pn_3_b, pn_3_b) * dx + inner(un_1_b, un_1_b) * dx)
    Hn_02 = 0.5 * (inner(pn_0_b, pn_0_b) * dx + inner(un_2_b, un_2_b) * dx)

    Hn_3210 = 0.5 * (dot(pn_0_b, pn_3_b) * dx + dot(un_2_b, un_1_b) * dx)

    Hn_ex = 0.5 * (inner(p_ex, p_ex) * dx(domain=mesh) + inner(u_ex, u_ex) * dx(domain=mesh))

    Hdot_n = 1/dt*(dot(pnmid_0_b, pn1_3_b - pn_3_b) * dx(domain=mesh) \
                   + dot(unmid_2_b, un1_1_b - un_1_b) * dx(domain=mesh))

    bdflow_midn = pnmid_0_b * dot(unmid_2_b, n_ver) * ds(domain=mesh)

    y_nmid_ess = 1 / dt * m_form(v0_b, pn1_0_b - pn_0_b, v1_b, un1_1_b - un_1_b, v3_b, pn1_3_b - pn_3_b, v2_b, un1_2_b - un_2_b) \
                   - j_form(v0_b, pnmid_0_b, v1_b, unmid_1_b, v3_b, pnmid_3_b, v2_b, unmid_2_b)

    u_nmid_nat2 = bdflow2(v0_b, unmid_2_b)
    u_nmid_nat0 = bdflow0(v2_b, pnmid_0_b)

    bdflow_n = pn_0_b * dot(un_2_b, n_ver) * ds(domain=mesh)
    bdflow_ex_n = p_ex * dot(u_ex, n_ver) * ds(domain=mesh)

    H_32_vec = np.zeros((1 + n_t,))
    H_10_vec = np.zeros((1 + n_t,))

    H_31_vec = np.zeros((1 + n_t,))
    H_02_vec = np.zeros((1 + n_t,))

    H_3210_vec = np.zeros((1 + n_t,))

    Hdot_vec = np.zeros((n_t,))

    bdflow_mid_vec = np.zeros((n_t,))

    bdflowV0_mid_vec = np.zeros((n_t,))
    bdflowV2_mid_vec = np.zeros((n_t,))

    bdflow_vec = np.zeros((1 + n_t,))
    bdflow_ex_vec = np.zeros((1 + n_t,))

    H_ex_vec = np.zeros((1 + n_t,))

    errL2_p_3_vec = np.zeros((1 + n_t,))
    errL2_u_1_vec = np.zeros((1 + n_t,))

    errL2_p_0_vec = np.zeros((1 + n_t,))
    errL2_u_2_vec = np.zeros((1 + n_t,))

    errHcurl_u_1_vec = np.zeros((1 + n_t,))
    errH1_p_0_vec = np.zeros((1 + n_t,))
    errHdiv_u_2_vec = np.zeros((1 + n_t,))

    err_p30_vec = np.zeros((1 + n_t,))
    err_u12_vec = np.zeros((1 + n_t,))

    errH_32_vec = np.zeros((1 + n_t,))
    errH_10_vec = np.zeros((1 + n_t,))
    errH_3210_vec = np.zeros((1 + n_t,))

    H_32_vec[0] = assemble(Hn_32)
    H_10_vec[0] = assemble(Hn_10)

    H_31_vec[0] = assemble(Hn_31)
    H_02_vec[0] = assemble(Hn_02)

    H_3210_vec[0] = assemble(Hn_3210)

    H_ex_vec[0] = assemble(Hn_ex)

    errH_32_vec[0] = np.abs(H_32_vec[0] - H_ex_vec[0])
    errH_10_vec[0] = np.abs(H_10_vec[0] - H_ex_vec[0])
    errH_3210_vec[0] = np.abs(H_3210_vec[0] - H_ex_vec[0])

    Hdot_vec[0] = assemble(Hdot_n)
    bdflow_vec[0] = assemble(bdflow_n)
    bdflow_ex_vec[0] = assemble(bdflow_ex_n)

    errL2_p_3_vec[0] = errornorm(p_ex, p0_3_b, norm_type="L2")
    errL2_u_1_vec[0] = errornorm(u_ex, u0_1_b, norm_type="L2")
    errL2_p_0_vec[0] = errornorm(p_ex, p0_0_b, norm_type="L2")
    errL2_u_2_vec[0] = errornorm(u_ex, u0_2_b, norm_type="L2")

    errHcurl_u_1_vec[0] = errornorm(u_ex, u0_1_b, norm_type="Hcurl")
    errH1_p_0_vec[0] = errornorm(p_ex, p0_0_b, norm_type="H1")
    errHdiv_u_2_vec[0] = errornorm(u_ex, u0_2_b, norm_type="Hdiv")

    err_p30_vec[0] = np.sqrt(assemble(inner(p0_3_b - p0_0_b, p0_3_b - p0_0_b) * dx))
    err_u12_vec[0] = np.sqrt(assemble(inner(u0_2_b - u0_1_b, u0_2_b - u0_1_b) * dx))

    ## Settings of intermediate variables and matrices for the 2 linear systems

    a_form = m_form(v0_b, p0_b, v1_b, u1_b, v3_b, p3_b, v2_b, u2_b) \
             - 0.5*dt*j_form(v0_b, p0_b, v1_b, u1_b, v3_b, p3_b, v2_b, u2_b)

    print("Computation of the solution with n elem " + str(n_el) + " n time " + str(n_t) + " deg " + str(deg))
    print("==============")

    for ii in tqdm(range(n_t)):

        input_2 = project(u_ex_mid, V2_b)
        input_0 = project(p_ex_mid, V0_b)

        ## Integration of the coupled system

        A_mat = assemble(a_form, bcs=bcs, mat_type='aij')

        b_form = m_form(v0_b, pn_0_b, v1_b, un_1_b, v3_b, pn_3_b, v2_b, un_2_b) \
                   + dt*(0.5*j_form(v0_b, pn_0_b, v1_b, un_1_b, v3_b, pn_3_b, v2_b, un_2_b) + bdflow(v2_b, input_0, v0_b, input_2))
        b_vec = assemble(b_form)

        solve(A_mat, en1_0132_b, b_vec, solver_parameters=params)

        # Computation of energy rate and fluxes

        enmid_0132_b.assign(0.5 * (en_0132_b + en1_0132_b))

        Hdot_vec[ii] = assemble(Hdot_n)

        bdflow_mid_vec[ii] = assemble(bdflow_midn)

        yhat_V0 = assemble(y_nmid_ess).vector().get_local()[dofsV0_D]
        u_midn_V0 = enmid_0132_b.vector().get_local()[dofsV0_D]

        uhat_V0 = assemble(u_nmid_nat0).vector().get_local()[dofsV0_N]
        y_midn_V0 = enmid_0132_b.vector().get_local()[dofsV0_N]

        bdflowV0_nat = np.dot(uhat_V0, y_midn_V0)
        bdflowV0_ess = np.dot(yhat_V0, u_midn_V0)
        bdflowV0_mid_vec[ii] = bdflowV0_nat + bdflowV0_ess

        yhat_V2 = assemble(y_nmid_ess).vector().get_local()[dofsV2_N]
        u_midn_V2 = enmid_0132_b.vector().get_local()[dofsV2_N]

        uhat_V2 = assemble(u_nmid_nat2).vector().get_local()[dofsV2_D]
        y_midn_V2 = enmid_0132_b.vector().get_local()[dofsV2_D]

        bdflowV2_nat = np.dot(uhat_V2, y_midn_V2)
        bdflowV2_ess = np.dot(yhat_V2, u_midn_V2)
        bdflowV2_mid_vec[ii] = bdflowV2_nat + bdflowV2_ess

        # New assign

        en_0132_b.assign(en1_0132_b)

        pn_0_b, un_1_b, pn_3_b, un_2_b = en_0132_b.split()

        bdflow_vec[ii+1] = assemble(bdflow_n)

        H_32_vec[ii+1] = assemble(Hn_32)
        H_10_vec[ii+1] = assemble(Hn_10)

        H_31_vec[ii+1] = assemble(Hn_31)
        H_02_vec[ii+1] = assemble(Hn_02)

        H_3210_vec[ii+1] = assemble(Hn_3210)

        p_3P[ii+1] = pn_3_b.at(Ppoint)
        p_0P[ii+1] = pn_0_b.at(Ppoint)

        t.assign(float(t) + float(dt))
        t_1.assign(float(t_1) + float(dt))

        H_ex_vec[ii + 1] = assemble(Hn_ex)

        bdflow_ex_vec[ii + 1] = assemble(bdflow_ex_n)

        # print(bdflow_ex_vec[ii+1])
        errH_32_vec[ii + 1] = np.abs(H_32_vec[ii + 1] - H_ex_vec[ii + 1])
        errH_10_vec[ii + 1] = np.abs(H_10_vec[ii + 1] - H_ex_vec[ii + 1])
        errH_3210_vec[ii + 1] = np.abs(H_3210_vec[ii + 1] - H_ex_vec[ii + 1])

        errL2_p_3_vec[ii + 1] = errornorm(p_ex, pn_3_b, norm_type="L2")
        errL2_u_1_vec[ii + 1] = errornorm(u_ex, un_1_b, norm_type="L2")
        errL2_p_0_vec[ii + 1] = errornorm(p_ex, pn_0_b, norm_type="L2")
        errL2_u_2_vec[ii + 1] = errornorm(u_ex, un_2_b, norm_type="L2")

        errHcurl_u_1_vec[ii + 1] = errornorm(u_ex, un_1_b, norm_type="Hcurl")
        errH1_p_0_vec[ii + 1] = errornorm(p_ex, pn_0_b, norm_type="H1")
        errHdiv_u_2_vec[ii + 1] = errornorm(u_ex, un_2_b, norm_type="Hdiv")

        err_p30_vec[ii + 1] = np.sqrt(assemble(inner(pn_3_b - pn_0_b, pn_3_b - pn_0_b) * dx))
        err_u12_vec[ii + 1] = np.sqrt(assemble(inner(un_2_b - un_1_b, un_2_b - un_1_b) * dx))

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
    # contours = trisurf(project(p_ex, V3_b), axes=axes, cmap="inferno")
    # axes.set_aspect("auto")
    # axes.set_title("$p_3$ Exact")
    # fig.colorbar(contours)
    #
    # fig = plt.figure()
    # axes = fig.add_subplot(111, projection='3d')
    # contours = trisurf(interpolate(p_ex, V0_b), axes=axes, cmap="inferno")
    # axes.set_aspect("auto")
    # axes.set_title("$p_0$ Exact")
    # fig.colorbar(contours)
    #
    # fig = plt.figure()
    # axes = fig.add_subplot(111, projection='3d')
    # contours = trisurf(pn_3_b, axes=axes, cmap="inferno")
    # axes.set_aspect("auto")
    # axes.set_title("$P_3$")
    # fig.colorbar(contours)
    #
    # fig = plt.figure()
    # axes = fig.add_subplot(111, projection='3d')
    # contours = trisurf(pn_0_b, axes=axes, cmap="inferno")
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
    plt.figure()
    plt.plot(t_vec, p_3P, 'r-', label=r'$p_3$')
    plt.plot(t_vec, p_0P, 'b-', label=r'$p_0$')
    plt.plot(t_vec, om_t * np.cos(om_x * Ppoint[0] + phi_x) * np.sin(om_y * Ppoint[1] + phi_y) \
             * np.sin(om_z * Ppoint[2] + phi_z) * (2 * np.cos(om_t * t_vec + phi_t) - 3 * np.sin(om_t * t_vec + phi_t)), \
             'g-', label=r'exact $p$')
    plt.xlabel(r'Time [s]')
    plt.title(r'$p$ at ' + str(Ppoint))
    plt.legend()
    plt.show()
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

    errH_3210 = errH_3210_vec[-1]
    errH_10 = errH_10_vec[-1]
    errH_32 = errH_32_vec[-1]

    int_bd_flow = np.zeros((1 + n_t,))

    for i in range(n_t):
        int_bd_flow[i+1] = int_bd_flow[i] + dt*bdflow_mid_vec[i]

    H_df_vec = H_3210_vec[0] + int_bd_flow

    dict_res = {"t_span": t_vec, "energy_ex": H_ex_vec, "energy_df": H_df_vec, "energy_3210": H_3210_vec,\
                "energy_32": H_32_vec, "energy_01": H_10_vec, "energy_31": H_31_vec, "energy_02": H_02_vec, \
                "power": Hdot_vec, "flow": bdflow_vec, "flow_ex": bdflow_ex_vec, "int_flow": int_bd_flow, \
                "flow_mid": bdflow_mid_vec, "flow10_mid": bdflowV0_mid_vec, "flow32_mid": bdflowV2_mid_vec,\
                "err_p3": errL2_p_3, "err_u1": [errL2_u_1, errHcurl_u_1], \
                "err_p0": [errL2_p_0, errH1_p_0], "err_u2": [errL2_u_2, errHdiv_u_2], "err_p30": err_p30, \
                "err_u12": err_u12, "err_H": [errH_3210, errH_10, errH_32]}

    return dict_res


bd_cond = input("Enter bc: ")

n_elem = 4
pol_deg = 2

n_time = 50
t_fin = 5

dt = t_fin / n_time

results = compute_err(n_elem, n_time, pol_deg, t_fin, bd_cond=bd_cond)


dictres_file = open("results_wave.pkl", "wb")
pickle.dump(results, dictres_file)
dictres_file.close()
