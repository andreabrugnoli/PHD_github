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

path_fig = "/home/andrea/Pictures/PythonPlots/DualField_Maxwell3D/"
bc_case = "_EH"
geo_case = "_3D"


def compute_err(n_el, n_t, deg=1, t_fin=1, bd_cond="H"):
    """Compute the numerical solution of the Maxwell equations with the dual field method

        Parameters:
        n_el: number of elements for the discretization
        n_t: number of time instants
        deg: polynomial degree for finite
        Returns:
        some plots

       """

    # c_0 = 299792458
    # mu_0 = 1.25663706212 * 10 ** (-6)
    # eps_0 = 8.85418781762 * 10 ** (-12)
    # print(c_0, (np.sqrt(eps_0*mu_0))**(-1))

    mu_0 = 3/2
    eps_0 = 2
    c_0 = (np.sqrt(eps_0*mu_0))**(-1)

    def m_formE2H1(v_E2, E_2, v_H1, H_1):
        m_form = inner(v_E2, eps_0*E_2) * dx + inner(v_H1, mu_0*H_1) * dx

        return m_form

    def m_formH2E1(v_H2, H_2, v_E1, E_1):
        m_form = inner(v_H2, mu_0*H_2) * dx + inner(v_E1, eps_0*E_1) * dx

        return m_form

    def j_formE2H1(v_E2, E_2, v_H1, H_1):
        j_form = dot(v_E2, curl(H_1)) * dx - dot(curl(v_H1), E_2) * dx

        return j_form

    def j_formH2E1(v_H2, H_2, v_E1, E_1):
        j_form = -dot(v_H2, curl(E_1)) * dx + dot(curl(v_E1), H_2) * dx

        return j_form

    def bdflowE2H1(v_H1, E_1):
        # b_form = -dot(cross(vH_1, n_ver), E_1) * ds
        # b_form = -dot(vH_1, cross(n_ver, E_1)) * ds
        b_form = dot(cross(v_H1, E_1), n_ver) * ds

        return b_form

    def bdflowH2E1(v_E1, H_1):
        # b_form = dot(cross(vE_1, n_ver), H_1) * ds
        # b_form = dot(vE_1, cross(n_ver, H_1)) * ds
        b_form = -dot(cross(v_E1, H_1), n_ver) * ds

        return b_form

    L = 1
    mesh = BoxMesh(n_el, n_el, n_el, 1, 1/2, 1/2)
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

    print("Space dimensions: ")
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

    om_x = 1
    om_y = 1
    om_z = 1

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

    # params = {"ksp_type": "gmres", "ksp_gmres_restart":100,\
    #          "pc_type": "hypre", 'pc_hypre_type': 'boomeramg'}

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

    curl_g = as_vector([om_y*sin(om_x*x+phi_x)*cos(om_y * y + phi_y)*cos(om_z * z + phi_z),
                       -(om_x+om_z)*cos(om_x*x+phi_x)*sin(om_y * y + phi_y)*cos(om_z * z + phi_z),
                        om_y*cos(om_x*x+phi_x)*cos(om_y * y + phi_y)*sin(om_z * z + phi_z)]) # curl(g_fun)

    # curl_g = curl(g_fun)

    E_ex = mu_0*g_fun * dft
    E_ex_1 = mu_0*g_fun * dft_1

    H_ex = -curl_g * ft
    H_ex_1 = -curl_g * ft_1

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

    # print("dofs on Gamma_H for E21")
    # print(dofsE2_H1_H)
    # print("dofs on Gamma_E for E21")
    # print(dofsE2_H1_E)
    #
    # print("dofs on Gamma_E for H21")
    # print(dofsH2_E1_E)
    # print("dofs on Gamma_H for H21")
    # print(dofsH2_E1_H)


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

    Hn_E2H1 = 0.5 * (inner(En_2, eps_0 * En_2) * dx + inner(Hn_1, mu_0 * Hn_1) * dx)
    Hn_H2E1 = 0.5 * (inner(Hn_2, mu_0 * Hn_2) * dx + inner(En_1, eps_0 * En_1) * dx)

    Hn_E2H2 = 0.5 * (inner(En_2, eps_0 * En_2) * dx + inner(Hn_2, mu_0 * Hn_2) * dx)
    Hn_E1H1 = 0.5 * (inner(En_1, eps_0 * En_1) * dx + inner(Hn_1, mu_0 * Hn_1) * dx)

    Hn_dual = 0.5 * (dot(En_1, eps_0 * En_2) * dx + dot(Hn_1, mu_0 * Hn_2) * dx)

    Hn_ex = 0.5 * (inner(E_ex, eps_0 * E_ex) * dx(domain=mesh) + inner(H_ex, mu_0 * H_ex) * dx(domain=mesh))

    Hdot_n = 1/dt*(dot(Enmid_1, eps_0 * (En1_2 - En_2)) * dx(domain=mesh) + dot(Hnmid_1, mu_0 * (Hn1_2 - Hn_2)) * dx(domain=mesh))

    # bdflow_midn = dot(cross(Enmid_1, n_ver), Hnmid_1) * ds(domain=mesh)
    bdflow_midn = -dot(cross(Enmid_1, Hnmid_1), n_ver) * ds(domain=mesh)

    y_nmid_essE2H1 = 1 / dt * m_formE2H1(v_E2, En1_2 - En_2, v_H1, Hn1_1 - Hn_1)\
                   - j_formE2H1(v_E2, Enmid_2, v_H1, Hnmid_1)
    u_nmid_natE2H1 = bdflowE2H1(v_H1, Enmid_1)


    y_nmid_essH2E1 = 1 / dt * m_formH2E1(v_H2, Hn1_2-Hn_2, v_E1, En1_1 - En_1) \
              - j_formH2E1(v_H2, Hnmid_2, v_E1, Enmid_1)
    u_nmid_natH2E1 = bdflowH2E1(v_E1, Hnmid_1)

    # bdflow_n = dot(cross(En_1, n_ver), Hn_1) * ds(domain=mesh)
    # bdflow_ex_n = dot(cross(E_ex, n_ver), H_ex) * ds(domain=mesh)

    bdflow_n = -dot(cross(En_1, Hn_1), n_ver) * ds(domain=mesh)
    bdflow_ex_n = -dot(cross(E_ex, H_ex), n_ver) * ds(domain=mesh)

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

    errL2_E2_vec = np.zeros((1 + n_t,))
    errL2_H1_vec = np.zeros((1 + n_t,))

    errL2_H2_vec = np.zeros((1 + n_t,))
    errL2_E1_vec = np.zeros((1 + n_t,))

    errHcurl_E1_vec = np.zeros((1 + n_t,))
    errHdiv_E2_vec = np.zeros((1 + n_t,))

    errHcurl_H1_vec = np.zeros((1 + n_t,))
    errHdiv_H2_vec = np.zeros((1 + n_t,))

    err_E21_vec = np.zeros((1 + n_t,))
    err_H21_vec = np.zeros((1 + n_t,))

    errH_E2H1_vec = np.zeros((1 + n_t,))
    errH_H2E1_vec = np.zeros((1 + n_t,))
    errH_dual_vec = np.zeros((1 + n_t,))

    divE2_vec = np.zeros((1 + n_t,))
    divH2_vec = np.zeros((1 + n_t,))

    divE2_vec[0] = assemble((div(En_2))**2*dx)
    divH2_vec[0] = assemble((div(Hn_2))**2*dx)

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

    errL2_E2_vec[0] = errornorm(E_ex, E0_2, norm_type="L2")
    errL2_E1_vec[0] = errornorm(H_ex, E0_1, norm_type="L2")

    errHdiv_E2_vec[0] = errornorm(E_ex, E0_2, norm_type="Hdiv")
    errHcurl_E1_vec[0] = errornorm(E_ex, E0_1, norm_type="Hcurl")

    errL2_H2_vec[0] = errornorm(H_ex, H0_2, norm_type="L2")
    errL2_H1_vec[0] = errornorm(H_ex, H0_1, norm_type="L2")

    errHdiv_H2_vec[0] = errornorm(H_ex, H0_2, norm_type="Hdiv")
    errHcurl_H1_vec[0] = errornorm(H_ex, H0_1, norm_type="Hcurl")

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

        # New assign

        en_E2_H1.assign(en1_E2_H1)
        en_H2_E1.assign(en1_H2_E1)

        En_2, Hn_1 = en_E2_H1.split()
        Hn_2, En_1 = en_H2_E1.split()

        bdflow_vec[ii+1] = assemble(bdflow_n)

        H_E2H1_vec[ii+1] = assemble(Hn_E2H1)
        H_H2E1_vec[ii+1] = assemble(Hn_H2E1)

        H_E2H2_vec[ii+1] = assemble(Hn_E2H2)
        H_E1H1_vec[ii+1] = assemble(Hn_E1H1)

        H_dual_vec[ii+1] = assemble(Hn_dual)

        t.assign(float(t) + float(dt))
        t_1.assign(float(t_1) + float(dt))

        H_ex_vec[ii + 1] = assemble(Hn_ex)

        bdflow_ex_vec[ii + 1] = assemble(bdflow_ex_n)

        # print(bdflow_ex_vec[ii+1])

        divE2_vec[ii+1] = assemble((div(En_2)) ** 2 * dx)
        divH2_vec[ii+1] = assemble((div(Hn_2)) ** 2 * dx)

        errH_E2H1_vec[ii + 1] = np.abs(H_E2H1_vec[ii + 1] - H_ex_vec[ii + 1])
        errH_H2E1_vec[ii + 1] = np.abs(H_H2E1_vec[ii + 1] - H_ex_vec[ii + 1])
        errH_dual_vec[ii + 1] = np.abs(H_dual_vec[ii + 1] - H_ex_vec[ii + 1])

        errL2_E2_vec[ii + 1] = errornorm(E_ex, En_2, norm_type="L2")
        errL2_E1_vec[ii + 1] = errornorm(E_ex, En_1, norm_type="L2")

        errHdiv_E2_vec[ii + 1] = errornorm(E_ex, En_2, norm_type="Hdiv")
        errHcurl_E1_vec[ii + 1] = errornorm(E_ex, En_1, norm_type="Hcurl")

        errL2_H2_vec[ii + 1] = errornorm(H_ex, Hn_2, norm_type="L2")
        errL2_H1_vec[ii + 1] = errornorm(H_ex, Hn_1, norm_type="L2")

        errHdiv_H2_vec[ii + 1] = errornorm(H_ex, Hn_2, norm_type="Hdiv")
        errHcurl_H1_vec[ii + 1] = errornorm(H_ex, Hn_1, norm_type="Hcurl")

        err_E21_vec[ii + 1] = np.sqrt(assemble(inner(En_2 - En_1, En_2 - En_1) * dx))

        err_H21_vec[ii + 1] = np.sqrt(assemble(inner(Hn_2 - Hn_1, Hn_2 - Hn_1) * dx))

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

    errL2_E2 = errL2_E2_vec[-1]
    errL2_E1 = errL2_E1_vec[-1]

    errL2_H2 = errL2_H2_vec[-1]
    errL2_H1 = errL2_H1_vec[-1]

    errHdiv_H2 = errHdiv_H2_vec[-1]

    errHdiv_E2 = errHdiv_E2_vec[-1]

    errHcurl_E1 = errHcurl_E1_vec[-1]
    errHcurl_H1 = errHcurl_H1_vec[-1]

    err_E21 = err_E21_vec[-1]

    err_H21 = err_H21_vec[-1]

    errH_dual = errH_dual_vec[-1]
    errH_E2H1 = errH_E2H1_vec[-1]
    errH_H2E1 = errH_H2E1_vec[-1]

    int_bd_flow = np.zeros((1 + n_t,))

    for i in range(n_t):
        int_bd_flow[i+1] = int_bd_flow[i] + dt*bdflow_mid_vec[i]

    H_df_vec = H_dual_vec[0] + int_bd_flow

    dict_res = {"t_span": t_vec, "energy_ex": H_ex_vec, "energy_df": H_df_vec, "energy_dual": H_dual_vec,\
                "energy_E2H1": H_E2H1_vec, "energy_H2E1": H_H2E1_vec, "energy_E2H2": H_E2H2_vec, "energy_E1H1": H_E1H1_vec, \
                "power": Hdot_vec, "flow": bdflow_vec, "flow_ex": bdflow_ex_vec, "int_flow": int_bd_flow, \
                "flow_mid": bdflow_mid_vec, "flowE2H1_mid": bdflowE2H1_mid_vec, "flowH2E1_mid": bdflowH2E1_mid_vec,\
                "err_E2": [errL2_E2, errHdiv_E2], "err_E1": [errL2_E1, errHcurl_E1], \
                "err_H2": [errL2_H2, errHdiv_H2], "err_H1": [errL2_H1, errHcurl_H1], \
                "err_E21": err_E21, "err_H21": err_H21, "err_H": [errH_dual, errH_E2H1, errH_H2E1], \
                "divE2": divE2_vec, "divH2": divH2_vec}

    return dict_res


bd_cond = input("Enter bc: ")
save_plots = input("Save plots: ")

n_elem = 2
pol_deg = 3

n_time = 200
t_fin = 5

dt = t_fin / n_time

results = compute_err(n_elem, n_time, pol_deg, t_fin, bd_cond=bd_cond)

t_vec = results["t_span"]
Hdot_vec = results["power"]

bdflow_vec = results["flow"]
bdflow_mid = results["flow_mid"]

bdflowE2H1_mid = results["flowE2H1_mid"]
bdflowH2E1_mid = results["flowH2E1_mid"]
int_bdflow = results["int_flow"]

H_df = results["energy_df"]
H_dual = results["energy_dual"]

H_E2H1 = results["energy_E2H1"]
H_H2E1 = results["energy_H2E1"]

H_E2H2 = results["energy_E2H2"]
H_E1H1 = results["energy_E1H1"]

H_ex = results["energy_ex"]
bdflow_ex_vec = results["flow_ex"]

errL2_E2, errHdiv_E2 = results["err_E2"]
errL2_E1, errHcurl_E1 = results["err_E1"]

errL2_H2, errHdiv_H2 = results["err_H2"]
errL2_H1, errHcurl_H1 = results["err_H1"]

err_Hs, err_H_E2H1, err_H_H2E1 = results["err_H"]

divE2 = results["divE2"]
divH2 = results["divH2"]

plt.figure()
plt.plot(t_vec, divE2, 'r-.', label=r"\mathrm{d}^2(E^2_h)")
plt.xlabel(r'Time $[\mathrm{s}]$')
plt.ylabel(r'$||d^2 E^2_h||_{L^2}$')

if save_plots:
    plt.savefig(path_fig + "div_E2" + geo_case + bc_case + ".pdf", format="pdf")

plt.figure()
plt.plot(t_vec, divH2, 'b-.', label=r"\mathrm{d}^2(H^2_h)")
plt.xlabel(r'Time $[\mathrm{s}]$')
plt.ylabel(r'$||d^2 H^2_h||_{L^2}$')

if save_plots:
    plt.savefig(path_fig + "div_H2" + geo_case + bc_case + ".pdf", format="pdf")

plt.figure()
plt.plot(t_vec[1:]-dt/2, Hdot_vec-bdflow_mid, 'r-.')
plt.xlabel(r'Time $[\mathrm{s}]$')
plt.ylabel(r'$P_h -<e^\partial_{h}, f^\partial_{h}>_{\partial M}$')
plt.title(r'Power balance conservation')

if save_plots:
    plt.savefig(path_fig + "pow_bal" + geo_case + bc_case + ".pdf", format="pdf")

plt.figure()
plt.plot(t_vec[1:]-dt/2, np.diff(H_E2H1)/dt - bdflowE2H1_mid, 'r-.')
plt.xlabel(r'Time $[\mathrm{s}]$')
plt.ylabel(r'$\dot{H}^{E^2 H^1}_h - <e^\partial_{h}, f^\partial_{h}>_{\Gamma_q} - \mathbf{u}^p \widehat{\mathbf{y}}^q$')
plt.title(r'Conservation law $\dot{H}^{E^2 H^1}_h$')

if save_plots:
    plt.savefig(path_fig + "pow_balE2H1" + geo_case + bc_case + ".pdf", format="pdf")

plt.figure()
plt.plot(t_vec[1:]-dt/2, np.diff(H_H2E1)/dt - bdflowH2E1_mid, 'r-.')
plt.xlabel(r'Time $[\mathrm{s}]$')
plt.ylabel(r'$\dot{H}^{H^2 E^1}_h - <e^\partial_{h}, f^\partial_{h}>_{\Gamma_p} - \mathbf{u}^q \widehat{\mathbf{y}}^p$')
plt.title(r'Conservation law $\dot{H}^{H^2 E^1}$')

if save_plots:
    plt.savefig(path_fig + "pow_balH2E1" + geo_case + bc_case + ".pdf", format="pdf")

plt.figure()
plt.plot(t_vec, bdflow_vec-bdflow_ex_vec, 'r-.')
plt.xlabel(r'Time $[\mathrm{s}]$')
plt.ylabel(r'$<e^\partial_{h}, f^\partial_{h}>_{\partial M} - <e^\partial_{\mathrm{ex}}, f^\partial_{\mathrm{ex}}>_{\partial M}$')
plt.title(r'Discrete and exact boundary flow')

if save_plots:
    plt.savefig(path_fig + "bd_flow" + geo_case + bc_case + ".pdf", format="pdf")


plt.figure()
plt.plot(t_vec[1:]-dt/2, np.abs(np.diff(H_E2H1)/dt - bdflow_mid), '-v', label=r"$\dot{H}^{E^2 H^1}_h$")
plt.plot(t_vec[1:]-dt/2, np.abs(np.diff(H_H2E1)/dt - bdflow_mid), '--', label=r"$\dot{H}^{H^2 E^1}_h$")
# plt.plot(t_vec[1:]-dt/2, np.abs(np.diff(H_E1H1)/dt - bdflow_mid), '-.+', label=r"$\dot{H}^{E^1 H^1}_h$")
# plt.plot(t_vec[1:]-dt/2, np.abs(np.diff(H_E2H2)/dt - bdflow_mid), '--*', label=r"$\dot{H}^{E^2 H^2}_h$")
plt.plot(t_vec[1:]-dt/2, np.abs(np.diff(H_dual)/dt - bdflow_mid), '-.', label=r'$\dot{H}^{dual}_h$')
plt.xlabel(r'Time $[\mathrm{s}]$')
plt.title(r'$|\dot{H}_h - <e^\partial_{h}, f^\partial_{h}>_{\partial M}|$')
plt.legend()

if save_plots:
    plt.savefig(path_fig + "dHdt" + geo_case + bc_case + ".pdf", format="pdf")

plt.figure()
plt.plot(t_vec, np.abs((H_E2H1 - H_E2H1[0]) - (H_ex-H_ex[0])), '-v', label=r'$\Delta H^{E^2 H^1}_h$')
plt.plot(t_vec, np.abs((H_H2E1 - H_H2E1[0]) - (H_ex-H_ex[0])), '--', label=r'$\Delta H^{H^2 E^1}_h$')
# plt.plot(t_vec, np.abs((H_E1H1 - H_E1H1[0]) - (H_ex-H_ex[0])), '--+', label=r'$\Delta H^{E^1 H^1}_h$')
# plt.plot(t_vec, np.abs((H_E2H2 - H_E2H2[0]) - (H_ex-H_ex[0])), '--*', label=r'$\Delta H^{E^2 H^2}_h$')
plt.plot(t_vec, np.abs((H_dual - H_dual[0]) - (H_ex-H_ex[0])), '-.', label=r'$\Delta H^{dual}_h$')
plt.plot(t_vec, np.abs(int_bdflow - (H_ex-H_ex[0])), '-.+', label=r'$\int_0^t P_h(\tau) d\tau$')
plt.xlabel(r'Time $[\mathrm{s}]$')
plt.title(r'$|\Delta H_h - \Delta H_{\mathrm{ex}}|$')
plt.legend()

if save_plots:
    plt.savefig(path_fig + "deltaH" + geo_case + bc_case + ".pdf", format="pdf")

plt.figure()
plt.plot(t_vec, np.abs(H_E2H1 - H_ex), '-v', label=r'$H^{E^2 H^1}_h$')
plt.plot(t_vec, np.abs(H_H2E1 - H_ex), '--', label=r'$H^{H^2 E^1}_h$')
# plt.plot(t_vec, np.abs(H_E1H1 - H_ex), '--+', label=r'$H^{E^1 H^1}_h$')
# plt.plot(t_vec, np.abs(H_E2H2 - H_ex), '--*', label=r'$H^{E^2 H^2}_h$')
plt.plot(t_vec, np.abs(H_dual - H_ex), '-.', label=r'$H^{dual}_h$')
plt.plot(t_vec, np.abs(H_df - H_ex), '-.+', label=r'$H_{\mathrm{df}}$')
plt.xlabel(r'Time $[\mathrm{s}]$')
plt.title(r'$|H_h - H_{\mathrm{ex}}|$')
plt.legend()

if save_plots:
    plt.savefig(path_fig + "H" + geo_case + bc_case + ".pdf", format="pdf")

plt.show()

print("Error L2 E2: " + str(errL2_E2))
print("Error Hdiv E2: " + str(errHdiv_E2))

print("Error L2 H2: " + str(errL2_H2))
print("Error Hdiv H2: " + str(errHdiv_H2))

print("Error L2 E1: " + str(errL2_E1))
print("Error Hcurl E1: " + str(errHcurl_E1))

print("Error L2 H1: " + str(errL2_H1))
print("Error Hcurl H1: " + str(errHcurl_H1))

print("Error Hs: " + str(err_Hs))
print("Error H_E2H1: " + str(err_H_E2H1))
print("Error H_H2E1: " + str(err_H_H2E1))
