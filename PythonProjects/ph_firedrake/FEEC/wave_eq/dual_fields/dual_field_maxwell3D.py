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

path_fig = "/home/andrea/Pictures/PythonPlots/DualField_maxwell3D/"
bc_case = "_DN"
geo_case = "_3D"


def compute_err(n_el, n_t, deg=1, t_fin=1, bd_cond="D"):
    """Compute the numerical solution of the Maxwell equations with the dual field method

        Parameters:
        n_el: number of elements for the discretization
        n_t: number of time instants
        deg: polynomial degree for finite
        Returns:
        some plots

       """

    c_0 = 299792458
    mu_0 = 1.25663706212 * 10 ** (-6)
    eps_0 = 8.85418781762 * 10 ** (-12)

    def m_formE2H1(vE_2, E_2, vH_1, H_1):
        m_form = inner(vE_2, eps_0*E_2) * dx + inner(vH_1, mu_0*H_1) * dx

        return m_form

    def m_formH2E1(vH_2, H_2, vE_1, E_1):
        m_form = inner(vH_2, mu_0*H_2) * dx + inner(vE_1, eps_0*E_1) * dx

        return m_form

    def j_formE2H1(vE_2, E_2, vH_1, H_1):
        j_form = dot(vE_2, curl(H_1)) * dx - dot(curl(vH_1), E_2) * dx

        return j_form

    def j_formH2E1(vH_2, H_2, vE_1, E_1):
        j_form = -dot(vH_2, curl(E_1)) * dx + dot(curl(vE_1), H_2) * dx

        return j_form

    def bdflowE2H1(vH_1, E_1):
        b_form = -dot(cross(vH_1, n_ver), E_1) * ds
        # b_form = -dot(vH_1, cross(n_ver, E_1)) * ds

        return b_form

    def bdflowH2E1(vE_1, H_1):
        b_form = dot(cross(vE_1, n_ver), H_1) * ds
        # b_form = dot(vE_1, cross(n_ver, H_1)) * ds

        return b_form

    L = 1
    mesh = BoxMesh(n_el, n_el, n_el, 1, 1, 1)
    n_ver = FacetNormal(mesh)

    PE_2 = FiniteElement("RT", tetrahedron, deg)
    PE_1 = FiniteElement("N1curl", tetrahedron, deg)

    PH_2 = FiniteElement("RT", tetrahedron, deg)
    PH_1 = FiniteElement("N1curl", tetrahedron, deg)

    VE_2 = FunctionSpace(mesh, PE_2)
    VE_1 = FunctionSpace(mesh, PE_1)

    VH_2 = FunctionSpace(mesh, PH_2)
    VH_1 = FunctionSpace(mesh, PH_1)

    V_E2_H1 = VE_2 * VH_1
    V_H2_E1 = VH_2 * VE_1

    print(VE_2.dim())
    print(VH_1.dim())

    v_E2_H1 = TestFunction(V_E2_H1)
    v_E2, v_H1 = split(v_E2_H1)

    v_H2_E1 = TestFunction(V_H2_E1)
    v_H2, v_E1 = split(v_H2_E1)

    e_E2_H1 = TrialFunction(V_E2_H1)
    E_2, H_1 = split(e_E2_H1)

    e_H2_E1 = TrialFunction(V_H2_E1)
    H_2, E_1 = split(e_H2_E1)

    dx = Measure('dx')
    ds = Measure('ds')

    x, y, z = SpatialCoordinate(mesh)

    om_x = pi
    om_y = pi
    om_z = pi

    om_t = np.sqrt(om_x ** 2 + om_y ** 2 + om_z ** 2)*c_0
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

    ft = sin(om_t * t)/om_t
    dft = cos(om_t * t)

    ft_1 = sin(om_t * t_1) / om_t
    dft_1 = cos(om_t * t_1)

    g_x = -cos(om_x * x + phi_x) * sin(om_y * y + phi_y) * sin(om_z * z + phi_z)
    g_y = Constant(0.0)
    g_z = sin(om_x * x + phi_x) * sin(om_y * y + phi_y) * cos(om_z * z + phi_z)

    g_fun = as_vector([g_x,
                       g_y,
                       g_z])

    curl_g = as_vector([pi*sin(om_x*x+phi_x)*cos(om_y * y + phi_y)*cos(om_z * z + phi_z),
                       -2*pi*cos(om_x*x+phi_x)*sin(om_y * y + phi_y)*cos(om_z * z + phi_z),
                        pi*cos(om_x*x+phi_x)*cos(om_y * y + phi_y)*sin(om_z * z + phi_z)]) # curl(g_fun)

    E_ex = mu_0*g_fun * dft
    H_ex = -curl_g * ft

    E_ex_1 = mu_0*g_fun * dft_1
    H_ex_1 = -curl_g * ft_1

    print(c_0, (np.sqrt(eps_0*mu_0))**(-1))

    E_ex_mid = 0.5 * (E_ex + E_ex_1)
    H_ex_mid = 0.5 * (H_ex + H_ex_1)

    E0_2 = interpolate(E_ex, VE_2)
    E0_1 = interpolate(E_ex, VE_1)

    H0_2 = interpolate(H_ex, VH_2)
    H0_1 = interpolate(H_ex, VH_1)

    if bd_cond == "H":
        bc_H = [DirichletBC(V_E2_H1.sub(1), H_ex_1, "on_boundary")]
        bc_H_nat = None

        bc_E = None
        bc_E_nat = [DirichletBC(V_H2_E1.sub(1), E_ex_1, "on_boundary")]

    elif bd_cond == "E":
        bc_E = [DirichletBC(V_H2_E1.sub(1), E_ex_1, "on_boundary")]
        bc_E_nat = None

        bc_H = None
        bc_H_nat = [DirichletBC(V_E2_H1.sub(1), H_ex_1, "on_boundary")]
    else:
        bc_H = [DirichletBC(V_E2_H1.sub(1), H_ex_1, 1), \
                DirichletBC(V_E2_H1.sub(1), H_ex_1, 3),
                DirichletBC(V_E2_H1.sub(1), H_ex_1, 5)]

        bc_H_nat = [DirichletBC(V_E2_H1.sub(1), H_ex_1, 2), \
                    DirichletBC(V_E2_H1.sub(1), H_ex_1, 4), \
                    DirichletBC(V_E2_H1.sub(1), H_ex_1, 6)]

        bc_E = [DirichletBC(V_H2_E1.sub(1), E_ex_1, 2), \
                DirichletBC(V_H2_E1.sub(1), E_ex_1, 4),
                DirichletBC(V_H2_E1.sub(1), E_ex_1, 6)]

        bc_E_nat = [DirichletBC(V_H2_E1.sub(1), E_ex_1, 1), \
                    DirichletBC(V_H2_E1.sub(1), E_ex_1, 3), \
                    DirichletBC(V_H2_E1.sub(1), E_ex_1, 5)]

    dofsE2_H1_H = []
    dofsH2_E1_H = []

    if bc_H is not None:
        for ii in range(len(bc_H)):
            nodesE2_H1_H = VE_2.dim() + bc_H[ii].nodes
            nodesH2_E1_H = VH_2.dim() + bc_E_nat[ii].nodes

            dofsE2_H1_H = dofsE2_H1_H + list(nodesE2_H1_H)
            dofsH2_E1_H = dofsH2_E1_H + list(nodesH2_E1_H)


    dofsE2_H1_H = list(set(dofsE2_H1_H))
    dofsH2_E1_H = list(set(dofsH2_E1_H))

    dofsE2_H1_E = []
    dofsH2_E1_E = []

    if bc_E is not None:
        for ii in range(len(bc_E)):
            nodesH2_E1_E = VH_2.dim() + bc_E[ii].nodes
            nodesE2_H1_E = VE_2.dim() + bc_H_nat[ii].nodes

            dofsH2_E1_E = dofsH2_E1_E + list(nodesH2_E1_E)
            dofsE2_H1_E = dofsE2_H1_E + list(nodesE2_H1_E)

    dofsE2_H1_E = list(set(dofsE2_H1_E))
    dofsH2_E1_E = list(set(dofsH2_E1_E))

    for element in dofsE2_H1_H:
        if element in dofsE2_H1_E:
            dofsE2_H1_E.remove(element)


    for element in dofsH2_E1_E:
        if element in dofsH2_E1_H:
            dofsH2_E1_H.remove(element)

    print("dofs on Gamma_H for E21")
    print(dofsE2_H1_H)
    print("dofs on Gamma_E for E21")
    print(dofsE2_H1_E)

    print("dofs on Gamma_E for H21")
    print(dofsH2_E1_E)
    print("dofs on Gamma_H for H21")
    print(dofsH2_E1_H)


    e0_E2_H1 = Function(V_E2_H1, name="e_E2H1 initial")
    e0_H2_E1 = Function(V_H2_E1, name="e_H2E1 initial")

    e0_E2_H1.sub(0).assign(E0_2)
    e0_E2_H1.sub(1).assign(H0_1)

    e0_H2_E1.sub(0).assign(H0_2)
    e0_H2_E1.sub(1).assign(E0_1)

    en_E2_H1 = Function(V_E2_H1, name="e_E2H1 n")
    en_H2_E1 = Function(V_H2_E1, name="e_H2E1 n")

    en_E2_H1.assign(e0_E2_H1)
    en_H2_E1.assign(e0_H2_E1)

    enmid_E2_H1 = Function(V_E2_H1, name="e_E2H1 n+1/2")
    enmid_H2_E1 = Function(V_H2_E1, name="e_H2E1 n+1/2")

    en1_E2_H1 = Function(V_E2_H1, name="e_E2H1 n+1")
    en1_H2_E1 = Function(V_H2_E1, name="e_H2E1 n+1")

    En_2, Hn_1 = en_E2_H1.split()
    Hn_2, En_1 = en_H2_E1.split()

    Enmid_2, Hnmid_1 = enmid_E2_H1.split()
    Hnmid_2, Enmid_1 = enmid_H2_E1.split()

    En1_2, Hn1_1 = en1_E2_H1.split()
    Hn1_2, En1_1 = en1_H2_E1.split()

    Hn_E2H1 = 0.5 * (inner(En_2, En_2) * dx + inner(Hn_1, Hn_1) * dx)
    Hn_H2E1 = 0.5 * (inner(Hn_2, Hn_2) * dx + inner(En_1, En_1) * dx)

    Hn_E2H2 = 0.5 * (inner(En_2, En_2) * dx + inner(Hn_2, Hn_2) * dx)
    Hn_E1H1 = 0.5 * (inner(En_1, En_1) * dx + inner(Hn_1, Hn_1) * dx)

    Hn_dual = 0.5 * (dot(En_1, En_2) * dx + dot(Hn_1, Hn_2) * dx)

    Hn_ex = 0.5 * (inner(E_ex, E_ex) * dx(domain=mesh) + inner(H_ex, H_ex) * dx(domain=mesh))

    Hdot_n = 1/dt*(dot(Enmid_1, En1_1 - En_1) * dx(domain=mesh) + dot(Hnmid_1, Hn1_1 - Hn_1) * dx(domain=mesh))

    bdflow_midn = dot(cross(Enmid_1, n_ver), Hnmid_1) * ds(domain=mesh)

    y_nmid_essE2H1 = 1 / dt * m_formE2H1(v_E2, En1_2 - En_2, v_H1, Hn1_1 - Hn_1)\
                   - j_formE2H1(v_E2, Enmid_2, v_H1, Hnmid_1)
    u_nmid_natE2H1 = bdflowE2H1(v_H2, Enmid_1)


    y_nmid_essH2E1 = 1 / dt * m_formH2E1(v_H2, Hn1_2-Hn_2, v_E1, En1_1 - En_1) \
              - j_formH2E1(v_H2, Hnmid_2, v_E1, Enmid_1)
    u_nmid_natH2E1 = bdflowH2E1(VE_1, Hnmid_1)

    bdflow_n = dot(cross(En_1, n_ver), Hn_1) * ds(domain=mesh)
    bdflow_ex_n = dot(cross(E_ex, n_ver), H_ex) * ds(domain=mesh)

    H_E2H1_vec = np.zeros((1 + n_t,))
    H_H2E1_vec = np.zeros((1 + n_t,))

    H_E2H2_vec = np.zeros((1 + n_t,))
    H_E1H1_vec = np.zeros((1 + n_t,))

    H_dual_vec = np.zeros((1 + n_t,))

    Hdot_vec = np.zeros((n_t,))

    bdflow_mid_vec = np.zeros((n_t,))

    bdflowE2H1_mid_vec = np.zeros((n_t,))
    bdflowH2E1_mid_vec = np.zeros((n_t,))

    bdflow_vec = np.zeros((1 + n_t,))
    bdflow_ex_vec = np.zeros((1 + n_t,))

    H_ex_vec = np.zeros((1 + n_t,))

    errL2_E_2_vec = np.zeros((1 + n_t,))
    errL2_H_1_vec = np.zeros((1 + n_t,))

    errL2_H_2_vec = np.zeros((1 + n_t,))
    errL2_E_1_vec = np.zeros((1 + n_t,))

    errHcurl_E_1_vec = np.zeros((1 + n_t,))
    errHdiv_E_2_vec = np.zeros((1 + n_t,))

    errHcurl_H_1_vec = np.zeros((1 + n_t,))
    errHdiv_H_2_vec = np.zeros((1 + n_t,))

    err_E21_vec = np.zeros((1 + n_t,))
    err_H21_vec = np.zeros((1 + n_t,))

    errH_E2H1_vec = np.zeros((1 + n_t,))
    errH_H2E1_vec = np.zeros((1 + n_t,))
    errH_dual_vec = np.zeros((1 + n_t,))

    H_E2H1_vec[0] = assemble(Hn_E2H1)
    H_H2E1_vec[0] = assemble(Hn_H2E1)

    H_E2H2_vec[0] = assemble(Hn_E2H2)
    H_E1H1_vec[0] = assemble(Hn_E1H1)

    H_dual_vec[0] = assemble(Hn_dual)

    H_ex_vec[0] = assemble(Hn_ex)

    errH_E2H1_vec[0] = np.abs(H_E2H1_vec[0] - H_ex_vec[0])
    errH_H2E1_vec[0] = np.abs(H_H2E1_vec[0] - H_ex_vec[0])
    errH_dual_vec[0] = np.abs(H_dual_vec[0] - H_ex_vec[0])

    Hdot_vec[0] = assemble(Hdot_n)
    bdflow_vec[0] = assemble(bdflow_n)
    bdflow_ex_vec[0] = assemble(bdflow_ex_n)

    errL2_E_2_vec[0] = errornorm(E_ex, E0_2, norm_type="L2")
    errL2_E_1_vec[0] = errornorm(H_ex, E0_1, norm_type="L2")

    errL2_H_2_vec[0] = errornorm(H_ex, H0_2, norm_type="L2")
    errL2_H_1_vec[0] = errornorm(H_ex, H0_1, norm_type="L2")

    errHcurl_E_1_vec[0] = errornorm(E_ex, E0_1, norm_type="Hcurl")
    errHdiv_E_2_vec[0] = errornorm(E_ex, E0_2, norm_type="Hdiv")

    errHcurl_H_1_vec[0] = errornorm(E_ex, H0_1, norm_type="Hcurl")
    errHdiv_H_2_vec[0] = errornorm(E_ex, H0_2, norm_type="Hdiv")

    err_E21_vec[0] = np.sqrt(assemble(inner(E0_2 - E0_1, E0_2 - E0_1) * dx))
    err_H21_vec[0] = np.sqrt(assemble(inner(H0_2 - H0_1, H0_2 - H0_1) * dx))

    ## Settings of intermediate variables and matrices for the 2 linear systems

    a_formE2H1 = m_formE2H1(v_E2, E_2, v_H1, H_1) - 0.5*dt*j_formE2H1(v_E2, E_2, v_H1, H_1)
    a_formH2E1 = m_formH2E1(v_H2, H_2, v_E1, E_1) - 0.5*dt*j_formH2E1(v_H2, H_2, v_E1, E_1)

    print("Computation of the solution with n elem " + str(n_el) + " n time " + str(n_t) + " deg " + str(deg))
    print("==============")

    for ii in tqdm(range(n_t)):

        input_H1 = interpolate(H_ex_mid, VH_1)
        input_E1 = interpolate(E_ex_mid, VE_1)

        ## Integration of E2H1 system (Electric natural)

        A_E2H1 = assemble(a_formE2H1, bcs=bc_H, mat_type='aij')

        b_formE2H1 = m_formE2H1(v_E2, En_2, v_H1, Hn_1) + dt*(0.5*j_formE2H1(v_E2, En_2, v_H1, Hn_1) + bdflowE2H1(v_H1, input_E1))
        b_vecE2H1 = assemble(b_formE2H1)

        solve(A_E2H1, en1_E2_H1, b_vecE2H1, solver_parameters=params)

        ## Integration of 10 system (Magnetic natural)

        A_H2E1 = assemble(a_formH2E1, bcs=bc_E, mat_type='aij')

        b_formH2E1 = m_formH2E1(v_H2, Hn_2, v_E1, En_1) + dt * (0.5 * j_formH2E1(v_H2, Hn_2, v_E1, En_1) + bdflowH2E1(v_E1, input_H1))

        b_vecH2E1 = assemble(b_formH2E1)

        solve(A_H2E1, en1_H2_E1, b_vecH2E1, solver_parameters=params)

        # Computation of energy rate and fluxes

        enmid_E2_H1.assign(0.5 * (en_E2_H1 + en1_E2_H1))
        enmid_H2_E1.assign(0.5 * (en_H2_E1 + en1_H2_E1))

        Hdot_vec[ii] = assemble(Hdot_n)

        bdflow_mid_vec[ii] = assemble(bdflow_midn)

        yhat_E2H1 = assemble(y_nmid_essE2H1).vector().get_local()[dofsE2_H1_H]
        u_midn_E2H1 = enmid_E2_H1.vector().get_local()[dofsE2_H1_H]

        uhat_E2H1 = assemble(u_nmid_natE2H1).vector().get_local()[dofsE2_H1_E]
        y_midn_E2H1 = enmid_E2_H1.vector().get_local()[dofsE2_H1_E]

        bdflowE2H1_nat = np.dot(uhat_E2H1, y_midn_E2H1)
        bdflowE2H1_ess = np.dot(yhat_E2H1, u_midn_E2H1)
        bdflowE2H1_mid_vec[ii] = bdflowE2H1_nat + bdflowE2H1_ess

        yhat_H2E1 = assemble(y_nmid_essH2E1).vector().get_local()[dofsH2_E1_E]
        u_midn_H2E1 = enmid_H2_E1.vector().get_local()[dofsH2_E1_E]

        uhat_H2E1 = assemble(u_nmid_natH2E1).vector().get_local()[dofsH2_E1_H]
        y_midn_H2E1 = enmid_H2_E1.vector().get_local()[dofsH2_E1_H]

        bdflowH2E1_nat = np.dot(uhat_H2E1, y_midn_H2E1)
        bdflowH2E1_ess = np.dot(yhat_H2E1, u_midn_H2E1)
        bdflowH2E1_mid_vec[ii] = bdflowH2E1_nat + bdflowH2E1_ess


    #     # New assign
    #
    #     en_32.assign(en1_32)
    #     en_10.assign(en1_10)
    #
    #     un_1, pn_0 = en_10.split()
    #     pn_3, un_2 = en_32.split()
    #
    #     bdflow_vec[ii+1] = assemble(bdflow_n)
    #
    #     H_32_vec[ii+1] = assemble(Hn_32)
    #     H_10_vec[ii+1] = assemble(Hn_10)
    #
    #     H_31_vec[ii+1] = assemble(Hn_31)
    #     H_02_vec[ii+1] = assemble(Hn_02)
    #
    #     H_3210_vec[ii+1] = assemble(Hn_3210)
    #
    #     p_3P[ii+1] = pn_3.at(Ppoint)
    #     p_0P[ii+1] = pn_0.at(Ppoint)
    #
    #     t.assign(float(t) + float(dt))
    #     t_1.assign(float(t_1) + float(dt))
    #
    #     H_ex_vec[ii + 1] = assemble(Hn_ex)
    #
    #     bdflow_ex_vec[ii + 1] = assemble(bdflow_ex_n)
    #
    #     # print(bdflow_ex_vec[ii+1])
    #     errH_32_vec[ii + 1] = np.abs(H_32_vec[ii + 1] - H_ex_vec[ii + 1])
    #     errH_10_vec[ii + 1] = np.abs(H_10_vec[ii + 1] - H_ex_vec[ii + 1])
    #     errH_3210_vec[ii + 1] = np.abs(H_3210_vec[ii + 1] - H_ex_vec[ii + 1])
    #
    #     errL2_p_3_vec[ii + 1] = errornorm(p_ex, pn_3, norm_type="L2")
    #     errL2_u_1_vec[ii + 1] = errornorm(u_ex, un_1, norm_type="L2")
    #     errL2_p_0_vec[ii + 1] = errornorm(p_ex, pn_0, norm_type="L2")
    #     errL2_u_2_vec[ii + 1] = errornorm(u_ex, un_2, norm_type="L2")
    #
    #     errHcurl_u_1_vec[ii + 1] = errornorm(u_ex, un_1, norm_type="Hcurl")
    #     errH1_p_0_vec[ii + 1] = errornorm(p_ex, pn_0, norm_type="H1")
    #     errHdiv_u_2_vec[ii + 1] = errornorm(u_ex, un_2, norm_type="Hdiv")
    #
    #     err_p30_vec[ii + 1] = np.sqrt(assemble(inner(pn_3 - pn_0, pn_3 - pn_0) * dx))
    #     err_u12_vec[ii + 1] = np.sqrt(assemble(inner(un_2 - un_1, un_2 - un_1) * dx))

    #     # print(r"Initial and final 32 energy:")
    #     # print(r"Inital: ", H_32_vec[0])
    #     # print(r"Final: ", H_32_vec[-1])
    #     # print(r"Delta: ", H_32_vec[-1] - H_32_vec[0])
    #     #
    #     # print(r"Initial and final 10 energy:")
    #     # print(r"Inital: ", H_10_vec[0])
    #     # print(r"Final: ", H_10_vec[-1])
    #     # print(r"Delta: ", H_10_vec[-1] - H_10_vec[0])
    #     #
    #     # plt.figure()
    #     # plt.plot(t_vec, p_3P, 'r-', label=r'$p_3$')
    #     # plt.plot(t_vec, p_0P, 'b-', label=r'$p_0$')
    #     # plt.plot(t_vec, om_t * np.sin(om_x * Ppoint[0] + phi_x) * np.sin(om_y * Ppoint[1] + phi_y) \
    #     #          * np.cos(om_t * t_vec + phi_t), 'g-', label=r'exact $p$')
    #     # plt.xlabel(r'Time [s]')
    #     # plt.title(r'$p$ at ' + str(Ppoint))
    #     # plt.legend()
    #
    #     # err_p_3 = np.sqrt(np.sum(float(dt) * np.power(err_p_3_vec, 2)))
    #     # err_u_1 = np.sqrt(np.sum(float(dt) * np.power(err_u_1_vec, 2)))
    #     # err_p_0 = np.sqrt(np.sum(float(dt) * np.power(err_p_0_vec, 2)))
    #     # err_u_2 = np.sqrt(np.sum(float(dt) * np.power(err_u_2_vec, 2)))
    #     #
    #     # err_p_3 = max(err_p_3_vec)
    #     # err_u_1 = max(err_u_1_vec)
    #     #
    #     # err_p_0 = max(err_p_0_vec)
    #     # err_u_2 = max(err_u_2_vec)
    #     #
    #     # err_p30 = max(err_p30_vec)
    #     # err_u12 = max(err_u12_vec)
    #
    # errL2_p_3 = errL2_p_3_vec[-1]
    # errL2_u_1 = errL2_u_1_vec[-1]
    #
    # errL2_p_0 = errL2_p_0_vec[-1]
    # errL2_u_2 = errL2_u_2_vec[-1]
    #
    # errHcurl_u_1 = errHcurl_u_1_vec[-1]
    #
    # errH1_p_0 = errH1_p_0_vec[-1]
    # errHdiv_u_2 = errHdiv_u_2_vec[-1]
    #
    # err_p30 = err_p30_vec[-1]
    # err_u12 = err_u12_vec[-1]
    #
    # errH_3210 = errH_3210_vec[-1]
    # errH_10 = errH_10_vec[-1]
    # errH_32 = errH_32_vec[-1]
    #
    # int_bd_flow = np.zeros((1 + n_t,))
    #
    # for i in range(n_t):
    #     int_bd_flow[i+1] = int_bd_flow[i] + dt*bdflow_mid_vec[i]
    #
    # H_df_vec = H_3210_vec[0] + int_bd_flow
    #
    # dict_res = {"t_span": t_vec, "energy_ex": H_ex_vec, "energy_df": H_df_vec, "energy_3210": H_3210_vec,\
    #             "energy_32": H_32_vec, "energy_01": H_10_vec, "energy_31": H_31_vec, "energy_02": H_02_vec, \
    #             "power": Hdot_vec, "flow": bdflow_vec, "flow_ex": bdflow_ex_vec, "int_flow": int_bd_flow, \
    #             "flow_mid": bdflow_mid_vec, "flow10_mid": bdflow10_mid_vec, "flow32_mid": bdflow32_mid_vec,\
    #             "err_p3": errL2_p_3, "err_u1": [errL2_u_1, errHcurl_u_1], \
    #             "err_p0": [errL2_p_0, errH1_p_0], "err_u2": [errL2_u_2, errHdiv_u_2], "err_p30": err_p30, \
    #             "err_u12": err_u12, "err_H": [errH_3210, errH_10, errH_32]}
    #
    # return dict_res


bd_cond = input("Enter bc: ")
save_plots = input("Save plots: ")

n_elem = 1
pol_deg = 1

n_time = 1
t_fin = 5

dt = t_fin / n_time

results = compute_err(n_elem, n_time, pol_deg, t_fin, bd_cond=bd_cond)


#
# t_vec = results["t_span"]
# Hdot_vec = results["power"]
#
# bdflow_vec = results["flow"]
# bdflow_mid = results["flow_mid"]
#
# bdflow10_mid = results["flow10_mid"]
# bdflow32_mid = results["flow32_mid"]
# int_bdflow = results["int_flow"]
#
# H_df = results["energy_df"]
# H_3210 = results["energy_3210"]
#
# H_32 = results["energy_32"]
# H_01 = results["energy_01"]
#
# H_31 = results["energy_31"]
# H_02 = results["energy_02"]
#
# H_ex = results["energy_ex"]
# bdflow_ex_vec = results["flow_ex"]
#
# errL2_p3 = results["err_p3"]
# errL2_u1, errHcurl_u1 = results["err_u1"]
# errL2_p0, errH1_p0 = results["err_p0"]
# errL2_u2, errHdiv_u2 = results["err_u2"]
#
# err_Hs, err_H10, err_H32 = results["err_H"]
#
#
#
# plt.figure()
# plt.plot(t_vec[1:]-dt/2, Hdot_vec - bdflow_mid, 'r-.')
# plt.xlabel(r'Time $[\mathrm{s}]$')
# plt.ylabel(r'$P -<e^\partial_{h}, f^\partial_{h}>_{\partial M}$')
# plt.title(r'Power balance conservation')
#
# if save_plots:
#     plt.savefig(path_fig + "pow_bal" + geo_case + bc_case + ".pdf", format="pdf")
#
# plt.figure()
# plt.plot(t_vec[1:]-dt/2, np.diff(H_01)/dt - bdflow10_mid, 'r-.')
# plt.xlabel(r'Time $[\mathrm{s}]$')
# plt.ylabel(r'$\dot{H}^{01} - <e^\partial_{h}, f^\partial_{h}>_{\Gamma_q} - \mathbf{u}^p \widehat{\mathbf{y}}^q$')
# plt.title(r'Conservation law $\dot{H}^{01}$')
#
# if save_plots:
#     plt.savefig(path_fig + "pow_bal10" + geo_case + bc_case + ".pdf", format="pdf")
#
# plt.figure()
# plt.plot(t_vec[1:]-dt/2, np.diff(H_32)/dt - bdflow32_mid, 'r-.')
# plt.xlabel(r'Time $[\mathrm{s}]$')
# plt.ylabel(r'$\dot{H}^{32} - <e^\partial_{h}, f^\partial_{h}>_{\Gamma_p} - \mathbf{u}^q \widehat{\mathbf{y}}^p$')
# plt.title(r'Conservation law $\dot{H}^{32}$')
#
# if save_plots:
#     plt.savefig(path_fig + "pow_bal32" + geo_case + bc_case + ".pdf", format="pdf")
#
# plt.figure()
# plt.plot(t_vec, bdflow_vec - bdflow_ex_vec, 'r-.')
# plt.xlabel(r'Time $[\mathrm{s}]$')
# plt.ylabel(r'$<e^\partial_{h}, f^\partial_{h}>_{\partial M} - <e^\partial_{\mathrm{ex}}, f^\partial_{\mathrm{ex}}>_{\partial M}$')
# plt.title(r'Discrete and exact boundary flow')
#
# if save_plots:
#     plt.savefig(path_fig + "bd_flow" + geo_case + bc_case + ".pdf", format="pdf")
#
#
# plt.figure()
# plt.plot(t_vec[1:]-dt/2, np.abs(np.diff(H_01)/dt - bdflow_mid), '-v', label=r"$\dot{H}^{01}$")
# plt.plot(t_vec[1:]-dt/2, np.abs(np.diff(H_32)/dt - bdflow_mid), '--', label=r"$\dot{H}^{32}$")
# plt.plot(t_vec[1:]-dt/2, np.abs(np.diff(H_02)/dt - bdflow_mid), '-.+', label=r"$\dot{H}^{02}$")
# plt.plot(t_vec[1:]-dt/2, np.abs(np.diff(H_31)/dt - bdflow_mid), '--*', label=r"$\dot{H}^{31}$")
# plt.plot(t_vec[1:]-dt/2, np.abs(np.diff(H_3210)/dt - bdflow_mid), '-.', label=r'$\dot{H}^{3201}$')
# plt.xlabel(r'Time $[\mathrm{s}]$')
# plt.title(r'$|\dot{H}_h - <e^\partial_{h}, f^\partial_{h}>_{\partial M}|$')
# plt.legend()
#
# if save_plots:
#     plt.savefig(path_fig + "dHdt" + geo_case + bc_case + ".pdf", format="pdf")
#
# plt.figure()
# plt.plot(t_vec, np.abs((H_01 - H_01[0]) - (H_ex-H_ex[0])), '-v', label=r'$\Delta H^{01}$')
# plt.plot(t_vec, np.abs((H_32 - H_32[0]) - (H_ex-H_ex[0])), '--', label=r'$\Delta H^{32}$')
# plt.plot(t_vec, np.abs((H_02 - H_02[0]) - (H_ex-H_ex[0])), '--+', label=r'$\Delta H^{02}$')
# plt.plot(t_vec, np.abs((H_31 - H_31[0]) - (H_ex-H_ex[0])), '--*', label=r'$\Delta H^{31}$')
# plt.plot(t_vec, np.abs((H_3210 - H_3210[0]) - (H_ex-H_ex[0])), '-.', label=r'$\Delta H^{3210}$')
# plt.plot(t_vec, np.abs(int_bdflow - (H_ex-H_ex[0])), '-.+', label=r'$\int_0^t P(\tau) d\tau$')
# plt.xlabel(r'Time $[\mathrm{s}]$')
# plt.title(r'$|\Delta H_h - \Delta H_{\mathrm{ex}}|$')
# plt.legend()
#
# if save_plots:
#     plt.savefig(path_fig + "deltaH" + geo_case + bc_case + ".pdf", format="pdf")
#
# plt.figure()
# plt.plot(t_vec, np.abs(H_01 - H_ex), '-v', label=r'$H^{01}$')
# plt.plot(t_vec, np.abs(H_32 - H_ex), '--', label=r'$H^{32}$')
# plt.plot(t_vec, np.abs(H_02 - H_ex), '--+', label=r'$H^{02}$')
# plt.plot(t_vec, np.abs(H_31 - H_ex), '--*', label=r'$H^{31}$')
# plt.plot(t_vec, np.abs(H_3210 - H_ex), '-.', label=r'$H^{3210}$')
# plt.plot(t_vec, np.abs(H_df - H_ex), '-.+', label=r'$H_{\mathrm{df}}$')
# plt.xlabel(r'Time $[\mathrm{s}]$')
# plt.title(r'$|H_h - H_{\mathrm{ex}}|$')
# plt.legend()
#
# if save_plots:
#     plt.savefig(path_fig + "H" + geo_case + bc_case + ".pdf", format="pdf")
#
# plt.show()


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
#
# print("Error Hs: " + str(err_Hs))
# print("Error H_10: " + str(err_H10))
# print("Error H_32: " + str(err_H32))
