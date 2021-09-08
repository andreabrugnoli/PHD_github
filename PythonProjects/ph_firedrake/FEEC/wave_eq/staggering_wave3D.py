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
    mesh = CubeMesh(n_el, n_el, n_el, L)
    n_ver = FacetNormal(mesh)

    P_0 = FiniteElement("CG", tetrahedron, deg)
    P_1 = FiniteElement("N1curl", tetrahedron, deg, variant='integral')
    # P_2 = FiniteElement("RT", tetrahedron, deg)
    # Integral evaluation on Raviart-Thomas for deg=3 completely freezes interpolation
    P_2 = FiniteElement("RT", tetrahedron, deg, variant='integral')
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

    # # e0_32 = project(as_vector([v_ex, sig_ex[0], sig_ex[1], sig_ex[2]]), V_32)
    # # e0_10 = project(as_vector([sig_ex[0], sig_ex[1], sig_ex[2], v_ex]), V_10)

    if bd_cond=="D":
        bc_D = DirichletBC(V_10.sub(1), p_ex, "on_boundary")
        bc_N = None
    elif bd_cond=="N":
        bc_N = DirichletBC(V_32.sub(1), u_ex, "on_boundary")
        bc_D = None
    else:
        bc_D = None
        bc_N = None

    Ppoint = (L/5, L/5, L/5)

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

    params = {"mat_type": "aij",
              "snes_type": "ksponly",
              "ksp_type": "preonly",
              "pc_type": "lu"}

    t_vec = np.linspace(0, n_t * float(dt), 1 + n_t)

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

    Hn_32 = 0.5 * (inner(pn_3, pn_3) * dx + inner(un_2, un_2) * dx)
    Hn_10 = 0.5 * (inner(pn_0, pn_0) * dx + inner(un_1, un_1) * dx)

    Hdot_n = div(un_2) * pn_0 * dx + inner(grad(pn_0), un_2) * dx
    bdflow_n = pn_0 * dot(un_2, n_ver) * ds

    H_32_vec = np.zeros((1 + n_t,))
    H_10_vec = np.zeros((1 + n_t,))

    Hdot_vec = np.zeros((1 + n_t,))
    bdflow_vec = np.zeros((1 + n_t,))

    err_p_3_vec = np.zeros((1 + n_t,))
    err_u_1_vec = np.zeros((1 + n_t,))

    err_p_0_vec = np.zeros((1 + n_t,))
    err_u_2_vec = np.zeros((1 + n_t,))

    err_p30_vec = np.zeros((1 + n_t,))
    err_u12_vec = np.zeros((1 + n_t,))

    err_p_3_vec[0] = errornorm(p_ex, p0_3, norm_type="L2")
    err_u_1_vec[0] = errornorm(u_ex, u0_1, norm_type="L2")
    err_p_0_vec[0] = errornorm(p_ex, p0_0, norm_type="H1")
    err_u_2_vec[0] = errornorm(u_ex, u0_2, norm_type="Hdiv")

    diff0_p30 = project(p0_3 - p0_0, V_3)
    diff0_u12 = project(u0_2 - u0_1, V_1)

    err_p30_vec[0] = errornorm(Constant(0), diff0_p30, norm_type="L2")
    err_u12_vec[0] = errornorm(Constant((0.0, 0.0)), diff0_u12, norm_type="L2")

    H_32_vec[0] = assemble(Hn_32)
    H_10_vec[0] = assemble(Hn_10)

    Hdot_vec[0] = assemble(Hdot_n)
    bdflow_vec[0] = assemble(bdflow_n)

    print("First explicit step")
    print("==============")

    b0_form32 = m_form32(v_3, p0_3, v_2, u0_2) + dt / 2 * (j_form32(v_3, p0_3, v_2, u0_2) + bdflow32(v_2, p0_0))

    A0_32 = assemble(m_form32(v_3, p_3, v_2, u_2), bcs=bc_N, mat_type='aij')
    b0_32 = assemble(b0_form32)

    solve(A0_32, enmid_32, b0_32, solver_parameters=params)

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
        # pnmid_3, unmid_2 = enmid_32.split()
        # pnmid1_3, unmid1_2 = enmid1_32.split()
        #
        # en1_32.sub(0).assign(0.5*(pnmid_3 + pnmid1_3))
        # en1_32.sub(1).assign(0.5*(unmid_2 + unmid1_2))
        #
        # en_32.assign(en1_32)

        en_32.assign(0.5*(enmid_32 + enmid1_32))

        en_10.assign(en1_10)
        enmid_32.assign(enmid1_32)

        un_1, pn_0 = en_10.split()
        pn_3, un_2 = en_32.split()

        Hdot_vec[ii+1] = assemble(Hdot_n)
        bdflow_vec[ii+1] = assemble(bdflow_n)

        H_32_vec[ii+1] = assemble(Hn_32)
        H_10_vec[ii+1] = assemble(Hn_10)

        p_3P[ii+1] = pn_3.at(Ppoint)
        p_0P[ii+1] = pn_0.at(Ppoint)

        t.assign(float(t) + float(dt))

        err_p_3_vec[ii+1] = errornorm(p_ex, pn_3, norm_type="L2")
        err_u_1_vec[ii+1] = errornorm(u_ex, un_1, norm_type="L2")
        err_p_0_vec[ii+1] = errornorm(p_ex, pn_0, norm_type="H1")
        err_u_2_vec[ii+1] = errornorm(u_ex, un_2, norm_type="Hdiv")

        diffn_p30 = project(pn_3 - pn_0, V_3)
        diffn_u12 = project(un_2 - un_1, V_1)

        err_p30_vec[ii + 1] = errornorm(Constant(0), diffn_p30, norm_type="L2")
        err_u12_vec[ii + 1] = errornorm(Constant((0.0, 0.0)), diffn_u12, norm_type="L2")

    print(r"Initial and final 32 energy:")
    print(r"Inital: ", H_32_vec[0])
    print(r"Final: ", H_32_vec[-1])
    print(r"Delta: ", H_32_vec[-1] - H_32_vec[0])

    print(r"Initial and final 10 energy:")
    print(r"Inital: ", H_10_vec[0])
    print(r"Final: ", H_10_vec[-1])
    print(r"Delta: ", H_10_vec[-1] - H_10_vec[0])

    plt.figure()
    plt.plot(t_vec, p_3P, 'r-', label=r'$p_3$')
    plt.plot(t_vec, p_0P, 'b-', label=r'$p_0$')
    plt.plot(t_vec, om_t * np.sin(om_x * Ppoint[0] + phi_x) * np.sin(om_y * Ppoint[1] + phi_y) * \
             np.sin(om_z * Ppoint[2] + phi_z) * np.cos(om_t * t_vec + phi_t), 'g-', label=r'exact $p$')
    plt.xlabel(r'Time [s]')
    plt.title(r'$p$ at ' + str(Ppoint))
    plt.legend()

    plt.show()

    err_p_3 = err_p_3_vec[-1]
    err_u_1 = err_u_1_vec[-1]

    err_p_0 = err_p_0_vec[-1]
    err_u_2 = err_u_2_vec[-1]

    dict_res = {"t_span": t_vec, "energy_32": H_32_vec, "energy_10": H_10_vec, "power": Hdot_vec, \
                "flow": bdflow_vec, "err_p3": err_p_3, "err_u1": err_u_1, "err_p0": err_p_0, "err_u2": err_u_2}

    return dict_res

n_elem = 8
pol_deg = 2

n_time = 100
t_fin = 1

results = compute_err(n_elem, n_time, pol_deg, t_fin)

t_vec = results["t_span"]
Hdot_vec = results["power"]
bdflow_vec = results["flow"]

H_32 = results["energy_32"]
H_10 = results["energy_10"]

plt.figure()
plt.plot(t_vec, H_32, 'r', label=r'$H_{32}$')
plt.plot(t_vec, H_10, 'b', label=r'$H_{10}$')
plt.xlabel(r'Time [s]')
plt.title(r' Mixed energy')
plt.legend()


plt.figure()
plt.plot(t_vec, Hdot_vec - bdflow_vec, 'r--', label=r'Energy residual')
plt.xlabel(r'Time [s]')
plt.title(r'Energy residual')
plt.legend()

diffH_L2Hdiv = np.diff(H_32)
diffH_H1Hcurl = np.diff(H_10)
Delta_t = np.diff(t_vec)
int_bdflow = np.zeros((n_time, ))

for i in range(n_time):
    int_bdflow[i] = 0.5*Delta_t[i]*(bdflow_vec[i+1] + bdflow_vec[i])

plt.figure()
plt.plot(t_vec[1:], diffH_L2Hdiv, 'ro', label=r'$\Delta H_{32}$')
plt.plot(t_vec[1:], diffH_H1Hcurl, 'b--', label=r'$\Delta H_{10}$')
plt.plot(t_vec[1:], int_bdflow, '*-', label=r'Bd flow int')
plt.xlabel(r'Time [s]')
plt.title(r'Energy balance')
plt.legend()


plt.show()
