import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "1"

from firedrake import *
from tqdm import tqdm
# from time import sleep
import matplotlib.pyplot as plt
from tools_plotting import setup


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

def compute_err(n_el, n_t, deg=1, t_fin=1, bd_cond="D", dim="2D"):


    mesh = BoxMesh(n_el, n_el, n_el, 1, 1 / 2, 1 / 2)
    n_ver = FacetNormal(mesh)

    WE1H2_loc, VE1_tan, V12, V1W2 = spacesE1H2(mesh, deg)
    V_E1H2_hyb = WE1H2_loc * VE1_tan

    v_E1H2_hyb = TestFunction(V_E1H2_hyb)
    vE1_hyb, vH2_hyb, vH1_nor, vE1_tan = split(v_E1H2_hyb)

    e_E1H2_hyb = TrialFunction(V_E1H2_hyb)
    E1_hyb, H2_hyb, H1_nor, E1_tan = split(e_E1H2_hyb)

    v_E1H2 = TestFunction(V1W2)
    vE1, vH2 = split(v_E1H2)

    e_E1H2 = TrialFunction(V1W2)
    E1, H2 = split(e_E1H2)

    print("Conforming Galerkin 12 dim: " + str(V12.dim()))
    print("Conforming Galerkin 12 (2 broken) dim: " + str(V12.sub(0).dim() + WE1H2_loc.sub(1).dim()))
    print("Hybrid 12 dim: " + str(VE1_tan.dim()))

    WE2H1_loc, VH1_tan, V21, W2V1 = spacesE2H1(mesh, deg)
    V_E2H1_hyb = WE2H1_loc * VH1_tan

    print("Conforming Galerkin 21 dim: " + str(V21.dim()))
    print("Conforming Galerkin 21 (2 broken) dim: " + str(WE2H1_loc.sub(0).dim() + V21.sub(1).dim()))
    print("Hybrid 21 dim: " + str(VH1_tan.dim()))

    v_E2H1_hyb = TestFunction(V_E2H1_hyb)
    vE2_hyb, vH1_hyb, vE1_nor, vH1_tan = split(v_E2H1_hyb)

    e_E2H1_hyb = TrialFunction(V_E2H1_hyb)
    E2_hyb, H1_hyb, E1_nor, H1_tan = split(e_E2H1_hyb)

    v_E2H1 = TestFunction(W2V1)
    vE2, vH1 = split(v_E2H1)

    e_E2H1 = TrialFunction(W2V1)
    E2, H1 = split(e_E2H1)

    dt = Constant(t_fin / n_t)

    t_vec = np.linspace(0, n_t * float(dt), 1 + n_t)

    t = Constant(0.0)
    t_1 = Constant(dt)

    E_ex, H_ex, E_ex_1, H_ex_1 = exact_sol_maxwell3D(mesh, t, t_1)
    E_ex_mid = 0.5 * (E_ex + E_ex_1)
    H_ex_mid = 0.5 * (H_ex + H_ex_1)

    if bd_cond == "H":
        bc_H_hyb = [DirichletBC(VH1_tan, H_ex_1, "on_boundary")]
        bc_E_hyb = []

        bc_H = [DirichletBC(W2V1.sub(1), H_ex_1, "on_boundary")]
        bc_E = []

    elif bd_cond == "E":
        bc_E_hyb = [DirichletBC(VE1_tan, E_ex_1, "on_boundary")]
        bc_H_hyb = []

        bc_E = [DirichletBC(V1W2.sub(0), E_ex_1, "on_boundary")]
        bc_H = []
    else:
        bc_H_hyb = [DirichletBC(VH1_tan, H_ex_1, 1), \
                    DirichletBC(VH1_tan, H_ex_1, 3),
                    DirichletBC(VH1_tan, H_ex_1, 5)]

        bc_E_hyb = [DirichletBC(VE1_tan, E_ex_1, 2), \
                    DirichletBC(VE1_tan, E_ex_1, 4),
                    DirichletBC(VE1_tan, E_ex_1, 6)]

        bc_H = [DirichletBC(W2V1.sub(1), H_ex_1, 1), \
                DirichletBC(W2V1.sub(1), H_ex_1, 3),
                DirichletBC(W2V1.sub(1), H_ex_1, 5)]

        bc_E = [DirichletBC(V1W2.sub(0), E_ex_1, 2), \
                DirichletBC(V1W2.sub(0), E_ex_1, 4),
                DirichletBC(V1W2.sub(0), E_ex_1, 6)]

    # Initial condition 01 and 32 hybrid

    en_E1H2_hyb = Function(V_E1H2_hyb, name="e_E1H2 n")
    assign_exactE1H2(E_ex, H_ex, en_E1H2_hyb, WE1H2_loc, VE1_tan, V12)
    En_1_hyb, Hn_2_hyb, Hn_1_nor, En_1_tan = en_E1H2_hyb.split()

    en_E2H1_hyb = Function(V_E2H1_hyb, name="e_E2H1 n")
    assign_exactE2H1(E_ex, H_ex, en_E2H1_hyb, WE2H1_loc, VH1_tan, V21)
    En_2_hyb, Hn_1_hyb, En_1_nor, Hn_1_tan = en_E2H1_hyb.split()

    # Initial condition for continuos

    try:
        E0_1 = interpolate(E_ex, V1W2.sub(0))
    except NotImplementedError:
        E0_1 = project(E_ex, V1W2.sub(0))

    try:
        H0_2 = interpolate(H_ex, V1W2.sub(1))
    except NotImplementedError:
        H0_2 = project(H_ex, V1W2.sub(1))


    en_E1H2 = Function(V1W2, name="e_E1H2 n")
    en_E1H2.sub(0).assign(E0_1)
    en_E1H2.sub(1).assign(H0_2)

    En_1, Hn_2 = en_E1H2.split()
    en1_E1H2 = Function(V1W2, name="e_E1H2 n+1")

    try:
        E0_2 = interpolate(E_ex, W2V1.sub(0))
    except NotImplementedError:
        E0_2 = project(E_ex, W2V1.sub(0))

    try:
        H0_1 = interpolate(H_ex, W2V1.sub(1))
    except NotImplementedError:
        H0_1 = project(H_ex, W2V1.sub(1))

    en_E2H1 = Function(W2V1, name="e_E2H1 n")
    en_E2H1.sub(0).assign(E0_2)
    en_E2H1.sub(1).assign(H0_1)

    En_2, Hn_1 = en_E2H1.split()
    en1_E2H1 = Function(W2V1, name="e_E2H1 n+1")

    # Error variables
    err_E1 = np.zeros((n_t,))
    err_E2 = np.zeros((n_t,))
    err_H1 = np.zeros((n_t,))
    err_H2 = np.zeros((n_t,))

    err_divE2 = np.zeros((n_t+1,))
    err_divH2 = np.zeros((n_t+1,))

    err_divE2[0] = norm(div(En_2))
    err_divH2[0] = norm(div(Hn_2))

    # Bilinear form E1H2
    a_formE1H2_hyb = m_formE1H2(vE1_hyb, E1_hyb, vH2_hyb, H2_hyb) - 0.5 * dt * j_formE1H2(vE1_hyb, E1_hyb, vH2_hyb, H2_hyb) \
                 - 0.5 * dt * constr_locE1H2(vE1_hyb, E1_hyb, vH1_nor, H1_nor, n_ver) \
                 - 0.5 * dt * constr_globalE1H2(vH1_nor, H1_nor, vE1_tan, E1_tan, n_ver)

    # Bilinear form E2H1
    a_formE2H1_hyb = m_formE2H1(vE2_hyb, E2_hyb, vH1_hyb, H1_hyb) - 0.5 * dt * j_formE2H1(vE2_hyb, E2_hyb, vH1_hyb, H1_hyb) \
                 - 0.5 * dt * constr_locE2H1(vH1_hyb, H1_hyb, vE1_nor, E1_nor, n_ver) \
                 - 0.5 * dt * constr_globalE2H1(vE1_nor, E1_nor, vH1_tan, H1_tan, n_ver)

    # Bilinear form continous

    a_formE1H2 = m_formE1H2(vE1, E1, vH2, H2) - 0.5 * dt * j_formE1H2(vE1, E1, vH2, H2)

    a_formE2H1 = m_formE2H1(vE2, E2, vH1, H1) - 0.5 * dt * j_formE2H1(vE2, E2, vH1, H1)

    print("Computation of the solution with n elem " + str(n_el) + " n time " + str(n_t) + " deg " + str(deg))
    print("==============")

    for ii in tqdm(range(n_t)):
        input_E = interpolate(E_ex_mid, V21.sub(1))
        input_H = interpolate(H_ex_mid, V12.sub(0))

        ## Integration of E1H2 system

        b_formE1H2_hyb = m_formE1H2(vE1_hyb, En_1_hyb, vH2_hyb, Hn_2_hyb)\
                         + 0.5 * dt * j_formE1H2(vE1_hyb, En_1_hyb, vH2_hyb, Hn_2_hyb) \
                     + 0.5 * dt * constr_locE1H2(vE1_hyb, En_1_hyb, vH1_nor, Hn_1_nor, n_ver) \
                     + 0.5 * dt * constr_globalE1H2(vH1_nor, Hn_1_nor, vE1_tan, En_1_tan, n_ver) \
                     + dt * bdflowE1H2(vE1_tan, input_H, n_ver)

        en1_E1H2_hyb = solve_hybrid(a_formE1H2_hyb, b_formE1H2_hyb, bc_E_hyb, VE1_tan, WE1H2_loc)

        A_E1H2 = assemble(a_formE1H2, bcs=bc_E, mat_type='aij')

        b_formE1H2 = m_formE1H2(vE1, En_1, vH2, Hn_2)\
                    + 0.5 * dt * j_formE1H2(vE1, En_1, vH2, Hn_2) \
                    + dt * bdflowE1H2(vE1, input_H, n_ver)

        b_vecE1H2 = assemble(b_formE1H2)

        solve(A_E1H2, en1_E1H2, b_vecE1H2)

        ## Integration of E2H1 system (E natural)

        b_formE2H1_hyb = m_formE2H1(vE2_hyb, En_2_hyb, vH1_hyb, Hn_1_hyb)\
                    + 0.5 * dt * j_formE2H1(vE2_hyb, En_2_hyb, vH1_hyb, Hn_1_hyb) \
                    + 0.5 * dt * constr_locE2H1(vH1_hyb, Hn_1_hyb, vE1_nor, En_1_nor, n_ver) \
                    + 0.5 * dt * constr_globalE2H1(vE1_nor, En_1_nor, vH1_tan, Hn_1_tan, n_ver) \
                    + dt * bdflowE2H1(vH1_tan, input_E, n_ver)

        en1_E2H1_hyb = solve_hybrid(a_formE2H1_hyb, b_formE2H1_hyb, bc_H_hyb, VH1_tan, WE2H1_loc)

        A_E2H1 = assemble(a_formE2H1, bcs=bc_H, mat_type='aij')

        b_formE2H1 = m_formE2H1(vE2, En_2, vH1, Hn_1) + 0.5 * dt * j_formE2H1(vE2, En_2, vH1, Hn_1) \
                     + dt * bdflowE2H1(vH1, input_E, n_ver)

        b_vecE2H1 = assemble(b_formE2H1)

        solve(A_E2H1, en1_E2H1, b_vecE2H1)

        # New assign hybrid
        en_E1H2_hyb.assign(en1_E1H2_hyb)
        En_1_hyb, Hn_2_hyb, Hn_1_nor, En_1_tan = en_E1H2_hyb.split()

        en_E2H1_hyb.assign(en1_E2H1_hyb)
        En_2_hyb, Hn_1_hyb, En_1_nor, Hn_1_tan = en_E2H1_hyb.split()

        # New assign continous
        en_E1H2.assign(en1_E1H2)
        en_E2H1.assign(en1_E2H1)

        En_1, Hn_2 = en_E1H2.split()
        En_2, Hn_1 = en_E2H1.split()

        t.assign(float(t) + float(dt))
        t_1.assign(float(t_1) + float(dt))

        err_E1[ii] = norm(En_1_hyb - En_1)
        err_E2[ii] = norm(En_2_hyb - En_2)

        err_H1[ii] = norm(Hn_1_hyb - Hn_1)
        err_H2[ii] = norm(Hn_2_hyb - Hn_2)

        err_divE2[ii+1] = norm(div(En_2))
        err_divH2[ii+1] = norm(div(Hn_2))


    return t_vec, err_E1, err_E2, err_H1, err_H2, err_divE2, err_divH2


bc_case = 'EH' #input("Enter bc: ")
geo_case = '3D'
n_elem = 4
pol_deg = 3

n_time = 3
t_fin = 1

dt = t_fin / n_time

t_vec, err_E1, err_E2, err_H1, err_H2, divE2, divH2 = compute_err(n_elem, n_time, pol_deg, t_fin, bd_cond=bc_case)

home_dir =os.environ['HOME']

path_fig = "/home/andrea/Pictures/PythonPlots/Hybridization_maxwell/"

if not os.path.exists(path_fig):
    # If it doesn't exist, create it
    os.makedirs(path_fig)

plt.figure()
plt.plot(t_vec[1:], err_E1, 'r-.')
plt.xlabel(r'Time')
plt.title("$||E^1_{\mathrm{cont}} - E^1_{\mathrm{hyb}}||_{L^2}$")
plt.savefig(path_fig + "diff_E1_conthyb" + geo_case + bc_case + ".pdf", format="pdf")

plt.figure()
plt.plot(t_vec[1:], err_H2, 'r-.')
plt.xlabel(r'Time')
plt.title("$||H^2_{\mathrm{cont}} - H^2_{\mathrm{hyb}}||_{L^2}$")
plt.savefig(path_fig + "diff_H2_conthyb" + geo_case + bc_case + ".pdf", format="pdf")

plt.figure()
plt.plot(t_vec[1:], err_E2, 'r-.')
plt.xlabel(r'Time')
plt.title("$||\widehat{E}^2_{\mathrm{cont}} - \widehat{E}^2_{\mathrm{hyb}}||_{L^2}$")
plt.savefig(path_fig + "diff_E2_conthyb" + geo_case + bc_case + ".pdf", format="pdf")

plt.figure()
plt.plot(t_vec[1:], err_H1, 'r-.')
plt.xlabel(r'Time')
plt.title("$||\widehat{H}^1_{\mathrm{cont}} - \widehat{H}^1_{\mathrm{hyb}}||_{L^2}$")
plt.savefig(path_fig + "diff_H1_conthyb" + geo_case + bc_case + ".pdf", format="pdf")

plt.figure()
plt.plot(t_vec, divE2, 'r-.') # , label=r"\mathrm{d}^2(E^2_h)")
plt.xlabel(r'Time')
plt.ylabel(r'$||\mathrm{d}^2 E^2_h||_{L^2}$')

plt.savefig(path_fig + "div_E2_" + geo_case + "_" + bc_case + ".pdf", format="pdf")

plt.figure()
plt.plot(t_vec, divH2, 'b-.') #, label=r"\mathrm{d}^2(H^2_h)")
plt.xlabel(r'Time')
plt.ylabel(r'$||\mathrm{d}^2 H^2_h||_{L^2}$')

plt.savefig(path_fig + "div_H2_" + geo_case + "_" + bc_case + ".pdf", format="pdf")

plt.show()
