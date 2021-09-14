## This is a first test to solve the wave equation in 2D domains using the dual filed method

# from warnings import simplefilter
# simplefilter(action='ignore', category=DeprecationWarning)

import os
os.environ["OMP_NUM_THREADS"] = "1"

# import numpy as np
from firedrake import *
from irksome import GaussLegendre, Dt, AdaptiveTimeStepper, TimeStepper, LobattoIIIA
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
    L=1
    mesh = BoxMesh(n_el, n_el, n_el, L, 1/2*L, 1/2*L)
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

    # Vp = FunctionSpace(mesh, 'DG', deg-1)
    # Vq = FunctionSpace(mesh, 'N1curl', deg, variant='integral')
    #
    # Vp_d = FunctionSpace(mesh, 'CG', deg)
    # Vq_d = FunctionSpace(mesh, 'RT', deg, variant='integral')

    V_3102 = V_3 * V_1 * V_0 * V_2

    v_3102 = TestFunction(V_3102)
    v_3, v_1, v_0, v_2 = split(v_3102)

    e_3102 = TrialFunction(V_3102)
    p_3, u_1, p_0, u_2 = split(e_3102)

    dx = Measure('dx')
    ds = Measure('ds')

    x, y, z = SpatialCoordinate(mesh)

    om_x = 1
    om_y = 1
    om_z = 1
    om_t = np.sqrt(om_x ** 2 + om_y ** 2 + om_z ** 2)
    phi_x = 0
    phi_y = 0
    phi_z = 0
    phi_t = 0

    t = Constant(0.0)

    ft = 2 * sin(om_t * t + phi_t) + 3 * cos(om_t * t + phi_t)
    dft_t = om_t * (2 * cos(om_t * t + phi_t) - 3 * sin(om_t * t + phi_t))  # diff(dft_t, t)

    gxyz = cos(om_x * x + phi_x) * sin(om_y * y + phi_y) * sin(om_z * z + phi_z)

    dgxyz_x = - om_x * sin(om_x * x + phi_x) * sin(om_y * y + phi_y) * sin(om_z * z + phi_z)
    dgxyz_y = om_y * cos(om_x * x + phi_x) * cos(om_y * y + phi_y) * sin(om_z * z + phi_z)
    dgxyz_z = om_z * cos(om_x * x + phi_x) * cos(om_y * y + phi_y) * cos(om_z * z + phi_z)

    # w_ex = gxy * ft

    p_ex = gxyz * dft_t
    u_ex = as_vector([dgxyz_x * ft,
                      dgxyz_y * ft,
                      dgxyz_z * ft])  # grad(gxy)

    p0_3 = interpolate(p_ex, V_3)
    u0_1 = interpolate(u_ex, V_1)
    p0_0 = interpolate(p_ex, V_0)
    u0_2 = interpolate(u_ex, V_2)

    e0_3102 = Function(V_3102)
    e0_3102.sub(0).assign(p0_3)
    e0_3102.sub(1).assign(u0_1)
    e0_3102.sub(2).assign(p0_0)
    e0_3102.sub(3).assign(u0_2)

    # in_cond = project(as_vector([v_ex, sig_ex[0], sig_ex[1], v_ex, sig_ex[0], sig_ex[1]]), V)

    pn_3, un_1, pn_0, un_2 = split(e0_3102)

    Hn_32 = 0.5 * (inner(pn_3, pn_3) * dx + inner(un_2, un_2) * dx)
    Hn_10 = 0.5 * (inner(pn_0, pn_0) * dx + inner(un_1, un_1) * dx)

    Hn_ex = 0.5 * (inner(p_ex, p_ex) * dx(domain=mesh) + inner(u_ex, u_ex) * dx(domain=mesh))

    Hdot_n = div(un_2) * pn_0 * dx + inner(grad(pn_0), un_2) * dx
    bdflow_n = pn_0 * dot(un_2, n_ver) * ds

    bdflow_ex_n = p_ex * dot(u_ex, n_ver) * ds(domain=mesh)

    H_32_vec = np.zeros((1 + n_t,))
    H_10_vec = np.zeros((1 + n_t,))
    H_ex_vec = np.zeros((1 + n_t,))

    bdflow_vec = np.zeros((1 + n_t,))
    bdflow_ex_vec = np.zeros((1 + n_t,))

    Hdot_vec = np.zeros((1 + n_t,))

    err_p_3_vec = np.zeros((1 + n_t,))
    err_u_1_vec = np.zeros((1 + n_t,))

    err_p_0_vec = np.zeros((1 + n_t,))
    err_u_2_vec = np.zeros((1 + n_t,))

    err_p30_vec = np.zeros((1 + n_t,))
    err_u12_vec = np.zeros((1 + n_t,))

    H_32_vec[0] = assemble(Hn_32)
    H_10_vec[0] = assemble(Hn_10)
    H_ex_vec[0] = assemble(Hn_ex)

    Hdot_vec[0] = assemble(Hdot_n)
    bdflow_vec[0] = assemble(bdflow_n)
    bdflow_ex_vec[0] = assemble(bdflow_ex_n)

    err_p_3_vec[0] = errornorm(p_ex, p0_3, norm_type="L2")
    err_u_1_vec[0] = errornorm(u_ex, u0_1, norm_type="L2")
    err_p_0_vec[0] = errornorm(p_ex, p0_0, norm_type="L2")
    err_u_2_vec[0] = errornorm(u_ex, u0_2, norm_type="L2")

    diff0_p30 = project(p0_3 - p0_0, V_3)
    diff0_u12 = project(u0_2 - u0_1, V_1)

    err_p30_vec[0] = errornorm(Constant(0), diff0_p30, norm_type="L2")
    err_u12_vec[0] = errornorm(Constant((0.0, 0.0, 0.0)), diff0_u12, norm_type="L2")

    m_form = inner(v_3, Dt(pn_3)) * dx + inner(v_1, Dt(un_1)) * dx \
             + inner(v_0, Dt(pn_0)) * dx + inner(v_2, Dt(un_2)) * dx

    # Check for sign in adjoint system
    j_form = dot(v_3, div(un_2)) * dx + dot(v_1, grad(pn_0)) * dx - dot(grad(v_0), un_1) * dx - dot(div(v_2), pn_3) * dx \
               + dot(v_2, n_ver) * pn_0 * ds # + v_0 * dot(un_2, n_ver) * ds
    # Form defininig the problem
    f_form = m_form - j_form

    # Method of manifactured solution: check demo on Firedrake irksome
    # rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact))

    dt = Constant(t_fin / n_t)
    butcher_tableau = GaussLegendre(1)

    params = {"mat_type": "aij",
              "snes_type": "ksponly",
              "ksp_type": "preonly",
              "pc_type": "lu"}

    t_vec = np.linspace(0, n_t * float(dt), 1 + n_t)

    if bd_cond=="D":
        bc = DirichletBC(V_3102.sub(2), p_ex, "on_boundary")
    elif bd_cond=="N":
        bc = DirichletBC(V_3102.sub(3), u_ex, "on_boundary")
    else:
        bc = [DirichletBC(V_3102.sub(2), p_ex, 1), \
              DirichletBC(V_3102.sub(2), p_ex, 2), \
              DirichletBC(V_3102.sub(2), p_ex, 3), \
              DirichletBC(V_3102.sub(3), u_ex, 4), \
              DirichletBC(V_3102.sub(3), u_ex, 5), \
              DirichletBC(V_3102.sub(3), u_ex, 6)]

    stepper = TimeStepper(f_form, butcher_tableau, t, dt, e0_3102,
                          bcs=bc, solver_parameters=params)
    Ppoint = (L/5, L/5, L/5)

    p_0P = np.zeros((1 + n_t,))
    p_0P[0] = interpolate(p_ex, V_0).at(Ppoint)

    p_3P = np.zeros((1 + n_t,))
    p_3P[0] = interpolate(p_ex, V_3).at(Ppoint)

    print("Computation of the solution")
    print("==============")

    for ii in tqdm(range(n_t)):
        stepper.advance()

        Hdot_vec[ii + 1] = assemble(Hdot_n)
        bdflow_vec[ii + 1] = assemble(bdflow_n)

        H_32_vec[ii + 1] = assemble(Hn_32)
        H_10_vec[ii + 1] = assemble(Hn_10)

        t.assign(float(t) + float(dt))

        H_ex_vec[ii + 1] = assemble(Hn_ex)

        bdflow_ex_vec[ii + 1] = assemble(bdflow_ex_n)

        err_p_3_vec[ii + 1] = errornorm(p_ex, interpolate(pn_3, V_3), norm_type="L2")
        err_u_1_vec[ii + 1] = errornorm(u_ex, interpolate(un_1, V_1), norm_type="L2")
        err_p_0_vec[ii + 1] = errornorm(p_ex, interpolate(pn_0, V_0), norm_type="L2")
        err_u_2_vec[ii + 1] = errornorm(u_ex, interpolate(un_2, V_2), norm_type="L2")

        diffn_p30 = project(pn_3 - pn_0, V_3)
        diffn_u12 = project(un_2 - un_1, V_2)

        err_p30_vec[ii + 1] = errornorm(Constant(0), diffn_p30, norm_type="L2")
        err_u12_vec[ii + 1] = errornorm(Constant((0.0, 0.0, 0.0)), diffn_u12, norm_type="L2")

        p_3P[ii + 1] = interpolate(pn_3, V_3).at(Ppoint)
        p_0P[ii + 1] = interpolate(pn_0, V_0).at(Ppoint)

    err_p3.assign(pn_3 - interpolate(p_ex, V_3))
    err_p0.assign(pn_0 - interpolate(p_ex, V_0))

    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    contours = trisurf(err_p3, axes=axes, cmap="inferno")
    axes.set_aspect("auto")
    axes.set_title("Error $p_3$")
    fig.colorbar(contours)

    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    contours = trisurf(err_p0, axes=axes, cmap="inferno")
    axes.set_aspect("auto")
    axes.set_title("Error $p_0$")
    fig.colorbar(contours)

    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    contours = trisurf(interpolate(p_ex, V_3), axes=axes, cmap="inferno")
    axes.set_aspect("auto")
    axes.set_title("$p_3$ Exact")
    fig.colorbar(contours)

    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    contours = trisurf(interpolate(p_ex, V_0), axes=axes, cmap="inferno")
    axes.set_aspect("auto")
    axes.set_title("$p_0$ Exact")
    fig.colorbar(contours)

    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    contours = trisurf(pn_3, axes=axes, cmap="inferno")
    axes.set_aspect("auto")
    axes.set_title("$P_3$")
    fig.colorbar(contours)

    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    contours = trisurf(pn_0, axes=axes, cmap="inferno")
    axes.set_aspect("auto")
    axes.set_title("$P_0$")
    fig.colorbar(contours)

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
    plt.plot(t_vec, om_t * np.cos(om_x * Ppoint[0] + phi_x) * np.sin(om_y * Ppoint[1] + phi_y) \
             * np.sin(om_y * Ppoint[1] + phi_y)* (2 * cos(om_t * t_vec + phi_t) - 3 * sin(om_t * t_vec + phi_t)),\
             'g-', label=r'exact $p$')
    plt.xlabel(r'Time [s]')
    plt.title(r'$p$ at ' + str(Ppoint))
    plt.legend()

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

    err_p_3 = err_p_3_vec[-1]
    err_u_1 = err_u_1_vec[-1]

    err_p_0 = err_p_0_vec[-1]
    err_u_2 = err_u_2_vec[-1]

    err_p30 = err_p30_vec[-1]
    err_u12 = err_u12_vec[-1]

    dict_res = {"t_span": t_vec, "energy_32": H_32_vec, "energy_10": H_10_vec, "energy_ex": H_ex_vec, \
                "power": Hdot_vec, "flow": bdflow_vec, "flow_ex": bdflow_ex_vec,\
                "err_p3": err_p_3, "err_u1": err_u_1, "err_p0": err_p_0, "err_u2": err_u_2, \
                "err_p30": err_p30, "err_u12": err_u12}

    return dict_res

n_elem = 5
pol_deg = 1

n_time = 100
t_fin = 1

results = compute_err(n_elem, n_time, pol_deg, t_fin)

t_vec = results["t_span"]
Hdot_vec = results["power"]

bdflow_vec = results["flow"]
bdflow_ex_vec = results["flow_ex"]

H_32 = results["energy_32"]
H_10 = results["energy_10"]
H_ex = results["energy_ex"]

err_p3 = results["err_p3"]
err_u1 = results["err_u1"]
err_p0 = results["err_p0"]
err_u2 = results["err_u2"]

print("Error p3: " + str(err_p3))
print("Error u1: " + str(err_u1))
print("Error p0: " + str(err_p0))
print("Error u2: " + str(err_u2))

plt.figure()
plt.plot(t_vec, H_32, 'r-.', label=r'$H_{32}$')
plt.plot(t_vec, H_10, 'b--', label=r'$H_{10}$')
plt.plot(t_vec, H_ex, '*-', label=r'H Exact')
plt.xlabel(r'Time [s]')
plt.title(r'Energies')
plt.legend()

plt.figure()
plt.plot(t_vec, Hdot_vec, '*-', label=r'Hdot')
plt.plot(t_vec, bdflow_vec, 'r-.', label=r'bd flow')
plt.plot(t_vec, bdflow_ex_vec, 'b--', label=r'bd flow ex')
plt.xlabel(r'Time [s]')
plt.title(r'Boundary flow')
plt.legend()


plt.figure()
plt.plot(t_vec, Hdot_vec - bdflow_vec, 'r--', label=r'Energy residual')
plt.xlabel(r'Time [s]')
plt.title(r'Energy residual')
plt.legend()

plt.show()

# diffH_L2Hdiv = np.diff(H_32)
# diffH_H1Hcurl = np.diff(H_10)
# Delta_t = np.diff(t_vec)
# int_bdflow = np.zeros((n_time, ))
#
# for i in range(n_time):
#     int_bdflow[i] = 0.5*Delta_t[i]*(bdflow_vec[i+1] + bdflow_vec[i])
#
# plt.figure()
# plt.plot(t_vec[1:], diffH_L2Hdiv, 'ro.', label=r'$\Delta H_{32}$')
# plt.plot(t_vec[1:], diffH_H1Hcurl, 'b--', label=r'$\Delta H_{10}$')
# plt.plot(t_vec[1:], int_bdflow, '*-.', label=r'Bd flow int')
# plt.xlabel(r'Time [s]')
# plt.title(r'Energy balance')
# plt.legend()
#
# plt.show()
