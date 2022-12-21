import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "1"

from firedrake import *
from tqdm import tqdm
# from time import sleep
import matplotlib.pyplot as plt
from tools_plotting import setup
import pickle


from FEEC.DiscretizationInterconnection.maxwell_eq.exact_eigensolution import exact_sol_maxwell3D
from FEEC.DiscretizationInterconnection.slate_syntax.solve_hybrid_system import solve_hybrid

from spaces_forms_hybridmaxwell import spacesE1H2, spacesE2H1, \
    m_formE1H2, m_formE2H1, j_formE1H2, j_formE2H1, \
    constr_locE1H2, constr_locE2H1,\
    constr_globalE1H2, constr_globalE2H1, \
    assign_exactE1H2, assign_exactE2H1, \
    bdflowE1H2, bdflowE2H1,  \
    project_ex_W1nor

from FEEC.DiscretizationInterconnection.dofs_bd_hybrid import dofs_ess_nat


def compute_err(n_el, n_t, deg=1, t_fin=1, bd_cond="EH"):
    """Compute the numerical solution of the wave equation with a DG method based on interconnection

        Parameters:
        n_el: number of elements for the discretization
        n_t: number of time instants
        deg: polynomial degree for finite
        Returns:
        some plots
       """

    mesh = BoxMesh(n_el, n_el, n_el, 1, 1, 1)
    n_ver = FacetNormal(mesh)
    h_cell = CellDiameter(mesh)

    WE1H2_loc, VE1_tan, V12 = spacesE1H2(mesh, deg)
    V_E1H2 = WE1H2_loc * VE1_tan

    v_E1H2 = TestFunction(V_E1H2)
    vE1, vH2, vH1_nor, vE1_tan = split(v_E1H2)

    e_E1H2 = TrialFunction(V_E1H2)
    E1, H2, H1_nor, E1_tan = split(e_E1H2)

    print("Conforming Galerkin 12 dim: " + str(V12.dim()))
    print("Conforming Galerkin 12 (2 broken) dim: " + str(V12.sub(0).dim()+WE1H2_loc.sub(1).dim()))
    print("Hybrid 12 dim: " + str(VE1_tan.dim()))

    WE2H1_loc, VH1_tan, V21 = spacesE2H1(mesh, deg)
    V_E2H1 = WE2H1_loc * VH1_tan

    print("Conforming Galerkin 21 dim: " + str(V21.dim()))
    print("Conforming Galerkin 21 (2 broken) dim: " + str(WE2H1_loc.sub(0).dim()+V21.sub(1).dim()))
    print("Hybrid 21 dim: " + str(VH1_tan.dim()))

    v_E2H1 = TestFunction(V_E2H1)
    vE2, vH1, vE1_nor, vH1_tan = split(v_E2H1)

    e_E2H1 = TrialFunction(V_E2H1)
    E2, H1, E1_nor, H1_tan = split(e_E2H1)

    dx = Measure('dx')
    ds = Measure('ds')
    dS = Measure('dS')

    dt = Constant(t_fin / n_t)

    t_vec = np.linspace(0, n_t * float(dt), 1 + n_t)

    t = Constant(0.0)
    t_1 = Constant(dt)

    E_ex, H_ex, E_ex_1, H_ex_1 = exact_sol_maxwell3D(mesh, t, t_1)
    E_ex_mid = 0.5 * (E_ex + E_ex_1)
    H_ex_mid = 0.5 * (H_ex + H_ex_1)

    if bd_cond == "H":
        bc_H = [DirichletBC(VH1_tan, H_ex_1, "on_boundary")]
        bc_E = []

    elif bd_cond == "E":
        bc_E = [DirichletBC(VE1_tan, E_ex_1, "on_boundary")]
        bc_H = []
    else:
        bc_H = [DirichletBC(VH1_tan, H_ex_1, 1), \
                DirichletBC(VH1_tan, H_ex_1, 3),
                DirichletBC(VH1_tan, H_ex_1, 5)]

        bc_E = [DirichletBC(VE1_tan, E_ex_1, 2), \
                DirichletBC(VE1_tan, E_ex_1, 4),
                DirichletBC(VE1_tan, E_ex_1, 6)]


    dofsVE1_tan_E, dofsVE1_tan_H = dofs_ess_nat(bc_E, WE1H2_loc, VE1_tan)
    dofsVH1_tan_H, dofsVH1_tan_E = dofs_ess_nat(bc_H, WE2H1_loc, VH1_tan)

    # Initial condition 01 and 32
    en_E1H2 = Function(V_E1H2, name="e_E1H2 n")
    assign_exactE1H2(E_ex, H_ex, en_E1H2, WE1H2_loc, VE1_tan, V12)
    En_1, Hn_2, Hn_1_nor, En_1_tan = en_E1H2.split()

    enmid_E1H2 = Function(V_E1H2, name="e_E1H2 n+1/2")
    Enmid_1, Hnmid_2, Hnmid_1_nor, Enmid_1_tan = enmid_E1H2.split()

    en_E2H1 = Function(V_E2H1, name="e_E2H1 n")
    assign_exactE2H1(E_ex, H_ex, en_E2H1, WE2H1_loc, VH1_tan, V21)
    En_2, Hn_1, En_1_nor, Hn_1_tan = en_E2H1.split()

    enmid_E2H1 = Function(V_E2H1, name="e_E2H1 n+1/2")
    Enmid_2, Hnmid_1, Enmid_1_nor, Hnmid_1_tan = enmid_E2H1.split()

    # Exact quantities
    Hn_ex = 0.5 * (inner(E_ex, E_ex) * dx(domain=mesh) + inner(H_ex, H_ex) * dx(domain=mesh))
    bdflow_ex_nmid = -dot(cross(E_ex_mid, H_ex_mid), n_ver) * ds(domain=mesh)

    bdflow_ex_mid_vec = np.zeros((n_t,))
    H_ex_vec = np.zeros((1 + n_t,))

    H_ex_vec[0] = assemble(Hn_ex)

    # Power balance combining primal and dual
    bdflow_num_nmid = -dot(cross(Enmid_1, Hnmid_1), n_ver) * ds(domain=mesh)

    bdflow_num_mid_vec = np.zeros((n_t,))
    Hdot_nmid_vec = np.zeros((n_t,))

    # Results E1H2
    Hn_E1H2 = 0.5 * (inner(En_1, En_1) * dx(domain=mesh) + inner(Hn_2, Hn_2) * dx(domain=mesh))
    H_E1H2_vec = np.zeros((1 + n_t,))
    errH_E1H2_vec = np.zeros((1 + n_t,))

    errL2_E_1_vec = np.zeros((1 + n_t,))
    errL2_H_2_vec = np.zeros((1 + n_t,))
    errHcurl_E_1_vec = np.zeros((1 + n_t,))
    errHdiv_H_2_vec = np.zeros((1 + n_t,))
    errL2_H_1nor_vec = np.zeros((n_t,))
    errL2_E_1tan_vec = np.zeros((1 + n_t,))

    bdflow_E1H2_mid_vec = np.zeros((n_t,))

    H_E1H2_vec[0] = assemble(Hn_E1H2)
    errH_E1H2_vec[0] = np.abs(H_E1H2_vec[0] - H_ex_vec[0])

    errL2_E_1_vec[0] = norm(E_ex-En_1)
    errHcurl_E_1_vec[0] = errornorm(E_ex, En_1, norm_type="Hcurl")
    errL2_H_2_vec[0] = norm(H_ex-Hn_2)
    errHdiv_H_2_vec[0] = errornorm(H_ex, Hn_2, norm_type="Hdiv")
    err_E_1tan = h_cell * (E_ex - En_1_tan) ** 2
    errL2_E_1tan_vec[0] = sqrt(assemble((err_E_1tan('+') + err_E_1tan('-')) * dS + err_E_1tan * ds))

    # Results E2H1
    Hn_E2H1 = 0.5 * (inner(En_2, En_2) * dx(domain=mesh) + inner(Hn_1, Hn_1) * dx(domain=mesh))
    H_E2H1_vec = np.zeros((1 + n_t,))
    errH_E2H1_vec = np.zeros((1 + n_t,))

    errL2_E_2_vec = np.zeros((1 + n_t,))
    errL2_H_1_vec = np.zeros((1 + n_t,))
    errHdiv_E_2_vec = np.zeros((1 + n_t,))
    errHcurl_H_1_vec = np.zeros((1 + n_t,))
    errL2_E_1nor_vec = np.zeros((n_t,))
    errL2_H_1tan_vec = np.zeros((1 + n_t,))

    bdflow_E2H1_mid_vec = np.zeros((n_t,))
    errL2_E_1nor_vec = np.zeros((n_t,))

    H_E2H1_vec[0] = assemble(Hn_E2H1)
    errH_E2H1_vec[0] = np.abs(H_E2H1_vec[0] - H_ex_vec[0])

    errL2_E_2_vec[0] = norm(E_ex - En_2)
    errHdiv_E_2_vec[0] = errornorm(E_ex, En_2, norm_type="Hdiv")
    errL2_H_1_vec[0] = norm(H_ex - Hn_1)
    errHcurl_H_1_vec[0] = errornorm(H_ex, Hn_1, norm_type="Hcurl")
    err_H_1tan = h_cell * (H_ex - Hn_1_tan) ** 2
    errL2_H_1tan_vec[0] = sqrt(assemble((err_H_1tan('+') + err_H_1tan('-')) * dS + err_H_1tan * ds))

    # Dual Field representation
    err_E12_vec = np.zeros((1 + n_t,))
    err_H12_vec = np.zeros((1 + n_t,))

    err_E12_vec[0] = norm(En_1 - En_2)
    err_H12_vec[0] = norm(Hn_1 - Hn_2)

    ## Settings of intermediate variables and matrices for the 2 linear systems
    # Bilinear form E1H2
    a_formE1H2 = m_formE1H2(vE1, E1, vH2, H2) - 0.5 * dt * j_formE1H2(vE1, E1, vH2, H2) \
               - 0.5 * dt * constr_locE1H2(vE1, E1, vH1_nor, H1_nor, n_ver) \
               - 0.5 * dt * constr_globalE1H2(vH1_nor, H1_nor, vE1_tan, E1_tan, n_ver)

    # Bilinear form E2H1
    a_formE2H1 = m_formE2H1(vE2, E2, vH1, H1) - 0.5 * dt * j_formE2H1(vE2, E2, vH1, H1) \
               - 0.5 * dt * constr_locE2H1(vH1, H1, vE1_nor, E1_nor, n_ver) \
               - 0.5 * dt * constr_globalE2H1(vE1_nor, E1_nor, vH1_tan, H1_tan, n_ver)

    print("Computation of the solution with n elem " + str(n_el) + " n time " + str(n_t) + " deg " + str(deg))
    print("==============")

    for ii in tqdm(range(n_t)):

        input_E = interpolate(E_ex_mid, V21.sub(1))
        input_H = interpolate(H_ex_mid, V12.sub(0))

        ## Integration of E1H2 system (H natural)

        b_formE1H2 = m_formE1H2(vE1, En_1, vH2, Hn_2) + 0.5 * dt * j_formE1H2(vE1, En_1, vH2, Hn_2) \
                   + 0.5 * dt * constr_locE1H2(vE1, En_1, vH1_nor, Hn_1_nor, n_ver) \
                   + 0.5 * dt * constr_globalE1H2(vH1_nor, Hn_1_nor, vE1_tan, En_1_tan, n_ver)\
                   + dt * bdflowE1H2(vE1_tan, input_H, n_ver)

        en1_E1H2 = solve_hybrid(a_formE1H2, b_formE1H2, bc_E, VE1_tan, WE1H2_loc)
        En1_1, Hn1_2, Hn1_1_nor, En1_1_tan = en1_E1H2.split()

        enmid_E1H2.assign(0.5 * (en_E1H2 + en1_E1H2))
        Enmid_1, Hnmid_2, Hnmid_1_nor, Enmid_1_tan = enmid_E1H2.split()

        ## Integration of E2H1 system (E natural)

        b_formE2H1 = m_formE2H1(vE2, En_2, vH1, Hn_1) + 0.5 * dt * j_formE2H1(vE2, En_2, vH1, Hn_1) \
                   + 0.5 * dt * constr_locE2H1(vH1, Hn_1, vE1_nor, En_1_nor, n_ver) \
                   + 0.5 * dt * constr_globalE2H1(vE1_nor, En_1_nor, vH1_tan, Hn_1_tan, n_ver) \
                   + dt * bdflowE2H1(vH1_tan, input_E, n_ver)

        en1_E2H1 = solve_hybrid(a_formE2H1, b_formE2H1, bc_H, VH1_tan, WE2H1_loc)
        En1_2, Hn1_1, En1_1_nor, Hn1_1_tan = en1_E2H1.split()

        enmid_E2H1.assign(0.5 * (en_E2H1 + en1_E2H1))
        Enmid_2, Hnmid_1, Enmid_1_nor, Hnmid_1_tan = enmid_E2H1.split()

        if len(bc_H) == 0:
            bdflow_natE1H2 = 0
            bdflow_essE2H1 = 0
        else:
            y_natE1H2 = enmid_E1H2.vector().get_local()[dofsVE1_tan_H]
            u_natE1H2 = assemble(bdflowE1H2(vE1_tan, input_H, n_ver)).vector().get_local()[dofsVE1_tan_H]
            bdflow_natE1H2 = np.dot(y_natE1H2, u_natE1H2)

            form_tn_E2H1 = inner(cross(vH1_tan, n_ver), cross(Enmid_1_nor, n_ver))
            y_essE2H1_form = (form_tn_E2H1('+') + form_tn_E2H1('-')) * dS + form_tn_E2H1 * ds(domain=mesh)
            y_essE2H1 = assemble(y_essE2H1_form).vector().get_local()[dofsVH1_tan_H]
            u_essE2H1 = assemble(enmid_E2H1).vector().get_local()[dofsVH1_tan_H]
            bdflow_essE2H1 = np.dot(y_essE2H1, u_essE2H1)

        if len(bc_E) == 0:
            bdflow_essE1H2 = 0
            bdflow_natE2H1 = 0

        else:
            y_natE2H1 = enmid_E2H1.vector().get_local()[dofsVH1_tan_E]
            u_natE2H1 = assemble(bdflowE2H1(vH1_tan, input_E, n_ver)).vector().get_local()[dofsVH1_tan_E]
            bdflow_natE2H1 = np.dot(y_natE2H1, u_natE2H1)

            form_tn_E1H2 = -inner(cross(vE1_tan, n_ver), cross(Hnmid_1_nor, n_ver))
            # Include dS because of vertices
            y_essE1H2_form = (form_tn_E1H2('+') + form_tn_E1H2('-')) * dS + form_tn_E1H2 * ds(domain=mesh)
            y_essE1H2 = assemble(y_essE1H2_form).vector().get_local()[dofsVE1_tan_E]
            u_essE1H2 = assemble(enmid_E1H2).vector().get_local()[dofsVE1_tan_E]
            bdflow_essE1H2 = np.dot(y_essE1H2, u_essE1H2)

        # Power
        Hdot_nmid = 1 / dt * (dot(Enmid_1, En1_2 - En_2) * dx(domain=mesh) + dot(Hnmid_1, Hn1_2 - Hn_2) * dx(domain=mesh))
        Hdot_nmid_vec[ii] = assemble(Hdot_nmid)

        # Different power flow
        bdflow_E1H2_mid_vec[ii] = bdflow_natE1H2 + bdflow_essE1H2
        bdflow_E2H1_mid_vec[ii] = bdflow_natE2H1 + bdflow_essE2H1

        bdflow_ex_mid_vec[ii] = assemble(bdflow_ex_nmid)
        bdflow_num_mid_vec[ii] = assemble(bdflow_num_nmid)

        # New assign
        en_E1H2.assign(en1_E1H2)
        En_1, Hn_2, Hn_1_nor, En_1_tan = en_E1H2.split()

        en_E2H1.assign(en1_E2H1)
        En_2, Hn_1, En_1_nor, Hn_1_tan = en_E2H1.split()

        H_E1H2_vec[ii + 1] = assemble(Hn_E1H2)
        H_E2H1_vec[ii + 1] = assemble(Hn_E2H1)

        t.assign(float(t) + float(dt))
        t_1.assign(float(t_1) + float(dt))
        H_ex_vec[ii + 1] = assemble(Hn_ex)
        # Error H
        errH_E1H2_vec[ii + 1] = np.abs(H_E1H2_vec[ii + 1] - H_ex_vec[ii + 1])
        errH_E2H1_vec[ii + 1] = np.abs(H_E2H1_vec[ii + 1] - H_ex_vec[ii + 1])
        # Error E1H2
        errL2_E_1_vec[ii + 1] = norm(E_ex - En_1)
        errHcurl_E_1_vec[ii + 1] = errornorm(E_ex, En_1, norm_type="Hcurl")
        errL2_H_2_vec[ii + 1] = norm(H_ex - Hn_2)
        errHdiv_H_2_vec[ii + 1] = errornorm(H_ex, Hn_2, norm_type="Hdiv")
        err_E_1tan = h_cell * cross(E_ex - En_1_tan, n_ver) ** 2
        errL2_E_1tan_vec[ii+1] = sqrt(assemble((err_E_1tan('+') + err_E_1tan('-')) * dS + err_E_1tan * ds))
        # Computation of the error for lambda nor
        # First project normal trace of H_ex onto Vnor
        Hex_nor_p = project_ex_W1nor(H_ex, WE1H2_loc.sub(2))
        err_H_1nor = h_cell * cross(Hex_nor_p - Hn_1_nor, n_ver) ** 2
        errL2_H_1nor_vec[ii] = sqrt(assemble((err_H_1nor('+') + err_H_1nor('-')) * dS + err_H_1nor * ds))

        # Error E2H1
        errL2_E_2_vec[ii + 1] = norm(E_ex - En_2)
        errHdiv_H_2_vec[ii + 1] = errornorm(E_ex, En_2, norm_type="Hdiv")
        errL2_H_1_vec[ii + 1] = norm(H_ex - Hn_1)
        errHcurl_H_1_vec[ii + 1] = errornorm(H_ex, Hn_1, norm_type="Hcurl")
        err_H_1tan = h_cell * cross(H_ex - Hn_1_tan, n_ver) ** 2
        errL2_H_1tan_vec[ii + 1] = sqrt(assemble((err_H_1tan('+') + err_H_1tan('-')) * dS + err_H_1tan * ds))
        # Computation of the error for lambda nor
        # First project normal trace of u_e onto Vnor
        Eex_nor_p = project_ex_W1nor(E_ex, WE2H1_loc.sub(2))
        err_E_1nor = h_cell * cross(Eex_nor_p - En_1_nor, n_ver) ** 2
        errL2_H_1nor_vec[ii] = sqrt(assemble((err_E_1nor('+') + err_E_1nor('-')) * dS + err_E_1nor * ds))

        # Dual Field
        err_E12_vec[ii + 1] = norm(En_2 - En_1)
        err_H12_vec[ii + 1] = norm(Hn_2 - Hn_1)

    # Error E1H2
    errH_E1H2 = errH_E1H2_vec[-1]

    errL2_E_1 = errL2_E_1_vec[-1]
    errL2_H_2 = errL2_H_2_vec[-1]

    errHcurl_E_1 = errHcurl_E_1_vec[-1]
    errHdiv_H_2 = errHdiv_H_2_vec[-1]

    errL2_H_1nor = errL2_H_1nor_vec[-1]
    errL2_E_1tan = errL2_E_1tan_vec[-1]

    # Error 32
    errH_E2H1 = errH_E2H1_vec[-1]

    errL2_E_2 = errL2_E_2_vec[-1]
    errL2_H_1 = errL2_H_1_vec[-1]

    errHdiv_E_2 = errHdiv_E_2_vec[-1]
    errHcurl_H_1 = errHcurl_H_1_vec[-1]

    errL2_E_1nor = errL2_E_1nor_vec[-1]
    errL2_H_1tan = errL2_H_1tan_vec[-1]

    # Dual field representation error
    err_E12 = err_E12_vec[-1]
    err_H12 = err_H12_vec[-1]

    dict_res = {"t_span": t_vec, "energy_ex": H_ex_vec, "flow_ex_mid": bdflow_ex_mid_vec,\
                "flow_num_mid": bdflow_num_mid_vec, "Hdot_num_mid": Hdot_nmid_vec, \
                "energy_12": H_E1H2_vec, "energy_21": H_E2H1_vec, \
                "flow12_mid": bdflow_E1H2_mid_vec, "err_E1": [errL2_E_1, errHcurl_E_1],\
                "err_H2": [errL2_H_2, errHdiv_H_2],     "err_E1tan": errL2_E_1tan, "err_H1nor": errL2_H_1nor, \
                "flow21_mid": bdflow_E2H1_mid_vec, "err_E2": [errL2_E_2, errHdiv_E_2], \
                "err_H1": [errL2_H_1, errHcurl_H_1], "err_H1tan": errL2_H_1tan, "err_E1nor": errL2_E_1nor, \
                "err_E12": err_E12, "err_H12": err_H12, "err_H": [errH_E1H2, errH_E2H1],
                }

    return dict_res


bd_cond = 'EH' #input("Enter bc: ")

n_elem = 1
pol_deg = 1

n_time = 10
t_fin = 1

dt = t_fin / n_time

results = compute_err(n_elem, n_time, pol_deg, t_fin, bd_cond=bd_cond)

# # dictres_file = open("results_hybridwave.pkl", "wb")
# # pickle.dump(results, dictres_file)
# # dictres_file.close()

t_vec = results["t_span"]

bdflow12_mid = results["flow12_mid"]
bdflow21_mid = results["flow21_mid"]

H_12 = results["energy_12"]
H_21 = results["energy_21"]

H_ex = results["energy_ex"]

bdflow_ex_nmid = results["flow_ex_mid"]
bdflow_num_nmid = results["flow_num_mid"]
Hdot_num_nmid = results["Hdot_num_mid"]

err_H12, err_H21 = results["err_H"]

plt.figure()
plt.plot(t_vec[1:]-dt/2, np.diff(H_12)/dt-bdflow12_mid, 'r-.', label="Power bal 12")
plt.xlabel(r'Time $[\mathrm{s}]$')
plt.legend()

plt.figure()
plt.plot(t_vec[1:]-dt/2, np.diff(H_12)/dt, 'r-.', label="DHdt 12")
plt.plot(t_vec[1:]-dt/2, bdflow12_mid, 'b-.', label="flow 12")
plt.xlabel(r'Time $[\mathrm{s}]$')
plt.legend()

plt.figure()
plt.plot(t_vec[1:]-dt/2, np.diff(H_21)/dt-bdflow21_mid, 'r-.', label="Power bal 21")
plt.xlabel(r'Time $[\mathrm{s}]$')
plt.legend()

plt.figure()
plt.plot(t_vec[1:]-dt/2, np.diff(H_21)/dt, 'r-.', label="DHdt 21")
plt.plot(t_vec[1:]-dt/2, bdflow21_mid, 'b-.', label="flow 21")
plt.xlabel(r'Time $[\mathrm{s}]$')
plt.legend()

plt.figure()
plt.plot(t_vec[1:]-dt/2, Hdot_num_nmid-bdflow_num_nmid, 'r-.', label="Discrete Power flow")
plt.xlabel(r'Time $[\mathrm{s}]$')
plt.legend()

plt.figure()
plt.plot(t_vec[1:]-dt/2, bdflow_ex_nmid-bdflow_num_nmid, 'r-.', label="Err power bal")
plt.xlabel(r'Time $[\mathrm{s}]$')
plt.legend()


plt.show()

# dictres_file = open("results_wave.pkl", "wb")
# pickle.dump(results, dictres_file)
# dictres_file.close()
#


