import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "1"

from firedrake import *
from tqdm import tqdm
# from time import sleep
import matplotlib.pyplot as plt
from tools_plotting import setup


from FEEC.DiscretizationInterconnection.wave_eq.exact_eigensolution import exact_sol_wave3D, exact_sol_wave2D
from FEEC.DiscretizationInterconnection.slate_syntax.solve_hybrid_system import solve_hybrid

from spaces_forms_hybridwave import spaces01, spaces32, \
    m_form01, m_form32, j_form01, j_form32, constr_loc01, constr_loc32, constr_global01, constr_global32, \
    assign_exact01, assign_exact32, \
    neumann_flow0, dirichlet_flow2

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
        mesh = RectangleMesh(n_el, n_el, 1, 1 / 2)
    else:
        mesh = BoxMesh(n_el, n_el, n_el, 1, 1 / 2, 1 / 2)
    n_ver = FacetNormal(mesh)


    W01_loc, V0_tan, V01 = spaces01(mesh, deg)
    V01_hyb = W01_loc * V0_tan

    v_01_hyb = TestFunction(V01_hyb)
    v0_hyb, v1_hyb, v0_nor, v0_tan = split(v_01_hyb)

    e_01_hyb = TrialFunction(V01_hyb)
    p0_hyb, u1_hyb, u0_nor, p0_tan = split(e_01_hyb)

    v_01 = TestFunction(V01)
    v0, v1 = split(v_01)

    e_01 = TrialFunction(V01)
    p0, u1 = split(e_01)

    print("Conforming Galerkin 01 dim: " + str(V01.dim()))
    print("Conforming Galerkin 01 (1 broken) dim: " + str(V01.sub(0).dim()+W01_loc.sub(1).dim()))
    print("Hybrid 01 dim: " + str(V0_tan.dim()))

    W32_loc, V2_tan, V32 = spaces32(mesh, deg)
    V32_hyb = W32_loc * V2_tan

    print("Conforming Galerkin 32 dim: " + str(V32.dim()))
    print("Hybrid 32 dim: " + str(V2_tan.dim()))

    v_32_hyb = TestFunction(V32_hyb)
    v3_hyb, v2_hyb, v2_nor, v2_tan = split(v_32_hyb)

    e_32_hyb = TrialFunction(V32_hyb)
    p3_hyb, u2_hyb, p2_nor, u2_tan = split(e_32_hyb)

    v_32 = TestFunction(V32)
    v3, v2 = split(v_32)

    e_32 = TrialFunction(V32)
    p3, u2 = split(e_32)

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
        bc_D_hyb = [DirichletBC(V0_tan, p_ex_1, "on_boundary")]
        bc_N_hyb = []

        bc_D = [DirichletBC(V01.sub(0), p_ex_1, "on_boundary")]
        bc_N = []
    elif bd_cond == "N":
        bc_D_hyb = []
        bc_N_hyb = [DirichletBC(V2_tan, u_ex_1, "on_boundary")]

        bc_D = []
        bc_N = [DirichletBC(V32.sub(1), u_ex_1, "on_boundary")]

    else:
        if dim=="2D":
            bc_D_hyb = [DirichletBC(V0_tan, p_ex_1, 1),
                        DirichletBC(V0_tan, p_ex_1, 3)]

            bc_N_hyb = [DirichletBC(V2_tan, u_ex_1, 2), \
                        DirichletBC(V2_tan, u_ex_1, 4)]

            bc_D = [DirichletBC(V01.sub(0), p_ex_1, 1),
                    DirichletBC(V01.sub(0), p_ex_1, 3)]

            bc_N = [DirichletBC(V32.sub(1), u_ex_1, 2), \
                    DirichletBC(V32.sub(1), u_ex_1, 4)]
        else:
            bc_D_hyb = [DirichletBC(V0_tan, p_ex_1, 1),
                        DirichletBC(V0_tan, p_ex_1, 3),
                        DirichletBC(V0_tan, p_ex_1, 5)]

            bc_N_hyb = [DirichletBC(V2_tan, u_ex_1, 2), \
                        DirichletBC(V2_tan, u_ex_1, 4),
                        DirichletBC(V2_tan, u_ex_1, 6)]

            bc_D = [DirichletBC(V01.sub(0), p_ex_1, 1),
                    DirichletBC(V01.sub(0), p_ex_1, 3),
                    DirichletBC(V01.sub(0), p_ex_1, 5)]

            bc_N = [DirichletBC(V32.sub(1), u_ex_1, 2), \
                    DirichletBC(V32.sub(1), u_ex_1, 4),
                    DirichletBC(V32.sub(1), u_ex_1, 6)]


    # Initial condition 01 and 32 hybrid
    en_01_hyb = Function(V01_hyb, name="e n")
    assign_exact01(p_ex, u_ex, en_01_hyb, W01_loc, V0_tan, V01)
    pn_0_hyb, un_1_hyb, un_0_nor, pn_0_tan = en_01_hyb.split()

    en_32_hyb = Function(V32_hyb, name="e n")
    assign_exact32(p_ex, u_ex, en_32_hyb, W32_loc, V2_tan, V32)
    pn_3_hyb, un_2_hyb, pn_2_nor, un_2_tan = en_32_hyb.split()

    # Initial condition for continuos

    try:
        u0_1 = interpolate(u_ex, V01.sub(1))
    except NotImplementedError:
        u0_1 = project(u_ex, V01.sub(1))

    p0_0 = interpolate(p_ex, V01.sub(0))

    en_01 = Function(V01, name="e_01 n")
    en_01.sub(0).assign(p0_0)
    en_01.sub(1).assign(u0_1)

    pn_0, un_1 = en_01.split()

    p0_3 = interpolate(p_ex, V32.sub(0))
    u0_2 = interpolate(u_ex, V32.sub(1))

    en_32 = Function(V32, name="e_32 n")
    en_32.sub(0).assign(p0_3)
    en_32.sub(1).assign(u0_2)

    pn_3, un_2 = en_32.split()

    en1_01 = Function(V01, name="e_10 n+1")
    en1_32 = Function(V32, name="e_32 n+1")

    # Error variables
    err_p0 = np.zeros((n_t,))
    err_p3 = np.zeros((n_t,))
    err_u1 = np.zeros((n_t,))
    err_u2 = np.zeros((n_t,))

    # Bilinear form 01 hybrid
    a_form01_hyb = m_form01(v0_hyb, p0_hyb, v1_hyb, u1_hyb) - 0.5 * dt * j_form01(v0_hyb, p0_hyb, v1_hyb, u1_hyb) \
               - 0.5 * dt * constr_loc01(v0_hyb, p0_hyb, v0_nor, u0_nor) \
               - 0.5 * dt * constr_global01(v0_nor, u0_nor, v0_tan, p0_tan)

    # Bilinear form 32 hybrid
    a_form32_hyb = m_form32(v3_hyb, p3_hyb, v2_hyb, u2_hyb) - 0.5 * dt * j_form32(v3_hyb, p3_hyb, v2_hyb, u2_hyb) \
               - 0.5 * dt * constr_loc32(v2_hyb, u2_hyb, v2_nor, p2_nor, n_ver) \
               - 0.5 * dt * constr_global32(v2_nor, p2_nor, v2_tan, u2_tan, n_ver)

    a_form01 = m_form01(v0, p0, v1, u1) - 0.5 * dt * j_form01(v0, p0, v1, u1)
    a_form32 = m_form32(v3, p3, v2, u2) - 0.5 * dt * j_form32(v3, p3, v2, u2)

    print("Computation of the solution with n elem " + str(n_el) + " n time " + str(n_t) + " deg " + str(deg))
    print("==============")

    for ii in tqdm(range(n_t)):

        input_u = interpolate(u_ex_mid, V32.sub(1))
        input_p = interpolate(p_ex_mid, V01.sub(0))

        ## Integration of 10 system (Neumann natural)

        b_form01_hyb = m_form01(v0_hyb, pn_0_hyb, v1_hyb, un_1_hyb) + 0.5 * dt * j_form01(v0_hyb, pn_0_hyb, v1_hyb, un_1_hyb) \
                   + 0.5 * dt * constr_loc01(v0_hyb, pn_0_hyb, v0_nor, un_0_nor) \
                   + 0.5 * dt * constr_global01(v0_nor, un_0_nor, v0_tan, pn_0_tan) \
                   + dt * neumann_flow0(v0_tan, input_u, n_ver)

        en1_01_hyb = solve_hybrid(a_form01_hyb, b_form01_hyb, bc_D_hyb, V0_tan, W01_loc)

        A_01 = assemble(a_form01, bcs=bc_D, mat_type='aij')

        b_form01 = m_form01(v0, pn_0, v1, un_1) + dt * (0.5 * j_form01(v0, pn_0, v1, un_1) \
                                                          + neumann_flow0(v0, input_u, n_ver))

        b_vec01 = assemble(b_form01)

        solve(A_01, en1_01, b_vec01)

        ## Integration of 32 system (Dirichlet natural)

        b_form32_hyb = m_form32(v3_hyb, pn_3_hyb, v2_hyb, un_2_hyb) + 0.5 * dt * j_form32(v3_hyb, pn_3_hyb, v2_hyb, un_2_hyb) \
                   + 0.5 * dt * constr_loc32(v2_hyb, un_2_hyb, v2_nor, pn_2_nor, n_ver) \
                   + 0.5 * dt * constr_global32(v2_nor, pn_2_nor, v2_tan, un_2_tan, n_ver) \
                   + dt * dirichlet_flow2(v2_tan, input_p, n_ver)

        en1_32_hyb = solve_hybrid(a_form32_hyb, b_form32_hyb, bc_N_hyb, V2_tan, W32_loc)

        A_32 = assemble(a_form32, bcs=bc_N, mat_type='aij')

        b_form32 = m_form32(v3, pn_3, v2, un_2) + dt * (0.5 * j_form32(v3, pn_3, v2, un_2) \
                                                        + dirichlet_flow2(v2, input_p, n_ver))
        b_vec32 = assemble(b_form32)

        solve(A_32, en1_32, b_vec32)

        # New assign hybrid
        en_01_hyb.assign(en1_01_hyb)
        pn_0_hyb, un_1_hyb, un_0_nor, pn_0_tan = en_01_hyb.split()

        en_32_hyb.assign(en1_32_hyb)
        pn_3_hyb, un_2_hyb, pn_2_nor, un_2_tan = en_32_hyb.split()

        # New assign continous
        en_32.assign(en1_32)
        en_01.assign(en1_01)

        pn_0, un_1 = en_01.split()
        pn_3, un_2 = en_32.split()

        t.assign(float(t) + float(dt))
        t_1.assign(float(t_1) + float(dt))

        err_p0[ii] = norm(pn_0_hyb - pn_0)
        err_p3[ii] = norm(pn_3_hyb - pn_3)
        err_u1[ii] = norm(un_1_hyb - un_1)
        err_u2[ii] = norm(un_2_hyb - un_2)


    return t_vec, err_p0, err_p3, err_u1, err_u2


bc_case = 'DN' #input("Enter bc: ")
geo_case = '3D'
n_elem = 4
pol_deg = 3

n_time = 100
t_fin = 1

dt = t_fin / n_time

t_vec, err_p0, err_p3, err_u1, err_u2 = compute_err(n_elem, n_time, pol_deg, t_fin, bd_cond=bc_case, dim=geo_case)

path_fig = "/home/andrea/Pictures/PythonPlots/Hybridization_wave/"

plt.figure()
plt.plot(t_vec[1:], err_p0, 'r-.')
plt.xlabel(r'Time')
plt.title("$||p^0_{\mathrm{cont}} - p^0_{\mathrm{hyb}}||_{L^2}$")
plt.savefig(path_fig + "diff_p0_conthyb" + geo_case + bc_case + ".pdf", format="pdf")

plt.figure()
plt.plot(t_vec[1:], err_u1, 'r-.')
plt.xlabel(r'Time')
plt.title("$||u^1_{\mathrm{cont}} - u^1_{\mathrm{hyb}}||_{L^2}$")
plt.savefig(path_fig + "diff_u1_conthyb" + geo_case + bc_case + ".pdf", format="pdf")

plt.figure()
plt.plot(t_vec[1:], err_p3, 'r-.')
plt.xlabel(r'Time')
plt.title("$||\widehat{p}^3_{\mathrm{cont}} - \widehat{p}^3_{\mathrm{hyb}}||_{L^2}$")
plt.savefig(path_fig + "diff_p3_conthyb" + geo_case + bc_case + ".pdf", format="pdf")


plt.figure()
plt.plot(t_vec[1:], err_u2, 'r-.')
plt.xlabel(r'Time')
plt.title("$||\widehat{u}^2_{\mathrm{cont}} - \widehat{u}^2_{\mathrm{hyb}}||_{L^2}$")
plt.savefig(path_fig + "diff_u2_conthyb" + geo_case + bc_case + ".pdf", format="pdf")


plt.show()
