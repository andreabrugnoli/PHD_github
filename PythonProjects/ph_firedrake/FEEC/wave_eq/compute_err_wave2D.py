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

    def rot2D(u_vec):
        return u_vec[1].dx(0) - u_vec[0].dx(1)

    L = 1
    mesh = RectangleMesh(n_el, n_el, L, L, quadrilateral=False)
    n_ver = FacetNormal(mesh)

    P_0 = FiniteElement("CG", triangle, deg)
    P_1e = FiniteElement("N1curl", triangle, deg, variant='integral')
    P_1f = FiniteElement("RT", triangle, deg, variant='integral')
    P_2 = FiniteElement("DG", triangle, deg - 1)

    Vp = FunctionSpace(mesh, P_2)
    Vq = FunctionSpace(mesh, P_1e)

    Vp_d = FunctionSpace(mesh, P_0)
    Vq_d = FunctionSpace(mesh, P_1f)

    # Vp = FunctionSpace(mesh, 'DG', deg-1)
    # Vq = FunctionSpace(mesh, 'N1curl', deg, variant='integral')
    #
    # Vp_d = FunctionSpace(mesh, 'CG', deg)
    # Vq_d = FunctionSpace(mesh, 'RT', deg, variant='integral')

    V = Vp * Vq * Vp_d * Vq_d

    v = TestFunction(V)
    vp, vq, vp_d, vq_d = split(v)

    e = TrialFunction(V)
    p, q, p_d, q_d = split(e)

    dx = Measure('dx')
    ds = Measure('ds')

    x, y = SpatialCoordinate(mesh)

    om_x = 3*pi
    om_y = 3*pi
    om_t = np.sqrt(om_x ** 2 + om_y ** 2)
    phi_x = 1
    phi_y = 2
    phi_t = 3

    t = Constant(0.0)
    w_ex = sin(om_x * x + phi_x) * sin(om_y * y + phi_y) * sin(om_t * t + phi_t)

    v_ex = om_t * sin(om_x * x + phi_x) * sin(om_y * y + phi_y) * cos(om_t * t + phi_t)
    sig_ex = as_vector([om_x * cos(om_x * x + phi_x) * sin(om_y * y + phi_y) * sin(om_t * t + phi_t),
                        om_y * sin(om_x * x + phi_x) * cos(om_y * y + phi_y) * sin(om_t * t + phi_t)])


    # v0 = om_t*sin(om_x*x)*sin(om_y*y)
    in_cond = project(as_vector([v_ex, sig_ex[0], sig_ex[1], v_ex, sig_ex[0], sig_ex[1]]), V)

    p0, q0, p0_d, q0_d = split(in_cond)

    E_L2Hdiv = 0.5 * (inner(p0, p0) * dx + inner(q0_d, q0_d) * dx)
    E_H1Hrot = 0.5 * (inner(p0_d, p0_d) * dx + inner(q0, q0) * dx)

    Hdot = div(q0_d) * p0_d * dx + inner(grad(p0_d), q0_d) * dx
    bdflow = p0_d * dot(q0_d, n_ver) * ds

    m_form = inner(vp, Dt(p0)) * dx + inner(vq, Dt(q0)) * dx + inner(vp_d, Dt(p0_d)) * dx + inner(vq_d, Dt(q0_d)) * dx

    # Check for sign in adjoint system
    j_form = dot(vp, div(q0_d)) * dx + dot(vq, grad(p0_d)) * dx - dot(grad(vp_d), q0) * dx - dot(div(vq_d), p0) * dx \
               + vp_d * dot(q0_d, n_ver) * ds + dot(vq_d, n_ver) * p0_d * ds
    # Form defininig the problem
    f_form = m_form - j_form

    # Method of manifactured solution: check demo on Firedrake irksome
    # rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact))

    dt = Constant(t_fin / n_t)
    butcher_tableau = GaussLegendre(deg)
    # butcher_tableau = LobattoIIIA(2)

    params = {"mat_type": "aij",
              "snes_type": "ksponly",
              "ksp_type": "preonly",
              "pc_type": "lu"}

    if bd_cond=="D":
        bc = DirichletBC(V.sub(2), v_ex, "on_boundary")
    elif bd_cond=="N":
        bc = DirichletBC(V.sub(3), sig_ex, "on_boundary")
    else: bc = []

    stepper = TimeStepper(f_form, butcher_tableau, t, dt, in_cond,
                          bcs=bc, solver_parameters=params)

    E_L2Hdiv_vec = np.zeros((1 + n_t,))
    E_H1Hrot_vec = np.zeros((1 + n_t,))

    Hdot_vec = np.zeros((1 + n_t,))
    bdflow_vec = np.zeros((1 + n_t,))

    p_err_L2_vec = np.zeros((1 + n_t,))
    q_err_Hrot_vec = np.zeros((1 + n_t,))

    pd_err_H1_vec = np.zeros((1 + n_t,))
    qd_err_Hdiv_vec = np.zeros((1 + n_t,))

    p_err_L2_vec[0] = np.sqrt(assemble((p0 - v_ex)**2 * dx))
    q_err_Hrot_vec[0] = np.sqrt(assemble(dot(q0 - sig_ex, q0 - sig_ex) * dx))
    # q_err_Hrot_vec[0] = np.sqrt(assemble(dot(q0 - sig_ex, q0 - sig_ex) * dx + rot2D(q0 - sig_ex)**2*dx))

    pd_err_H1_vec[0] = np.sqrt(assemble((p0_d - v_ex)**2 * dx + dot(grad(p0_d - v_ex), grad(p0_d - v_ex))*dx))
    qd_err_Hdiv_vec[0] = np.sqrt(assemble(dot(q0_d - sig_ex, q0_d - sig_ex) * dx + div(q0_d - sig_ex)**2 * dx))

    Ppoint = (L/6, L/4)

    p_P = np.zeros((1+n_t,))
    p_P[0] = interpolate(v_ex, Vp).at(Ppoint)

    pd_P = np.zeros((1+n_t, ))
    pd_P[0] = interpolate(v_ex, Vp_d).at(Ppoint)

    t_vec = np.linspace(0, n_t * float(dt), 1 + n_t)


    E_L2Hdiv_vec[0] = assemble(E_L2Hdiv)
    E_H1Hrot_vec[0] = assemble(E_H1Hrot)

    Hdot_vec[0] = assemble(Hdot)
    bdflow_vec[0] = assemble(bdflow)

    pn = Function(Vp, name="p primal at t_n")
    pn_d = Function(Vp_d, name="p dual at t_n")

    qn = Function(Vq, name="q primal at t_n")
    qn_d = Function(Vq_d, name="q dual at t_n")

    err_p = Function(Vp, name="p error primal at t_n")
    err_pd = Function(Vp_d, name="p error dual at t_n")

    print("Computation of the solution")
    print("==============")

    for ii in tqdm(range(n_t)):
        stepper.advance()

        Hdot_vec[ii+1] = assemble(Hdot)
        bdflow_vec[ii+1] = assemble(bdflow)

        pn.assign(interpolate(p0, Vp))
        pn_d.assign(interpolate(p0_d, Vp_d))

        qn.assign(interpolate(q0, Vq))
        qn_d.assign(interpolate(q0_d, Vq_d))

        p_P[ii+1] = pn.at(Ppoint)
        pd_P[ii+1] = pn_d.at(Ppoint)

        t.assign(float(t) + float(dt))
        # print("Primal energy")
        # print("{0:1.1e} {1:5e}".format(float(t), Ep_vec[ii]))
        # print("Dual energy")
        # print("{0:1.1e} {1:5e}".format(float(t), Ed_vec[ii]))
        # print("Scattering energy")
        # print("{0:1.1e} {1:5e}".format(float(t), Es_vec[ii]))

        # p_err_L2_vec[ii + 1] = np.sqrt(assemble((pn - v_ex) ** 2 * dx))
        # q_err_Hrot_vec[ii + 1] = np.sqrt(assemble(dot(qn - sig_ex, qn - sig_ex) * dx + rot2D(qn - sig_ex) ** 2 * dx))
        #
        # pd_err_H1_vec[ii + 1] = np.sqrt(assemble((pn_d - v_ex) ** 2 * dx + dot(grad(pn_d - v_ex), grad(pn_d - v_ex)) * dx))
        # qd_err_Hdiv_vec[ii + 1] = np.sqrt(assemble(dot(qn_d - sig_ex, qn_d - sig_ex) * dx + div(qn_d - sig_ex) ** 2 * dx))

        p_err_L2_vec[ii + 1] = np.sqrt(assemble((p0 - v_ex) ** 2 * dx))
        q_err_Hrot_vec[ii + 1] = np.sqrt(assemble(dot(q0 - sig_ex, q0 - sig_ex) * dx))
        # q_err_Hrot_vec[ii + 1] = np.sqrt(assemble(dot(q0 - sig_ex, q0 - sig_ex) * dx + rot2D(q0 - sig_ex) ** 2 * dx))

        pd_err_H1_vec[ii + 1] = np.sqrt(assemble((p0_d - v_ex) ** 2 * dx + dot(grad(p0_d - v_ex), grad(p0_d - v_ex)) * dx))
        qd_err_Hdiv_vec[ii + 1] = np.sqrt(assemble(dot(q0_d - sig_ex, q0_d - sig_ex) * dx + div(q0_d - sig_ex) ** 2 * dx))

    err_p.assign(pn - interpolate(v_ex, Vp))
    err_pd.assign(pn_d - interpolate(v_ex, Vp_d))

    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    contours = trisurf(err_p, axes=axes, cmap="inferno")
    axes.set_aspect("auto")
    axes.set_title("Error primal velocity")
    fig.colorbar(contours)

    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    contours = trisurf(err_pd, axes=axes, cmap="inferno")
    axes.set_aspect("auto")
    axes.set_title("Error dual velocity")
    fig.colorbar(contours)

    plt.figure()
    plt.plot(t_vec, p_P, 'r-', label=r'primal $p$')
    plt.plot(t_vec, pd_P, 'b-', label=r'dual $p$')
    plt.plot(t_vec, om_t * np.sin(om_x * Ppoint[0] + phi_x) * np.sin(om_y * Ppoint[1] + phi_y) * \
             np.cos(om_t * t_vec + phi_t), 'g-', label=r'exact $p$')
    plt.xlabel(r'Time [s]')
    plt.title(r'$p$ at ' + str(Ppoint))
    plt.legend()

    plt.show()

    # print(r"Initial and final L2Hdiv energy:")
    # print(r"Inital: ", E_L2Hdiv_vec[0])
    # print(r"Final: ", E_L2Hdiv_vec[-1])
    # print(r"Delta: ", E_L2Hdiv_vec[-1] - E_L2Hdiv_vec[0])
    #
    # print(r"Initial and final H1Hrot energy:")
    # print(r"Inital: ", E_H1Hrot_vec[0])
    # print(r"Final: ", E_H1Hrot_vec[-1])
    # print(r"Delta: ", E_H1Hrot_vec[-1] - E_H1Hrot_vec[0])

    # fig = plt.figure()
    # axes = fig.add_subplot(111, projection='3d')
    # contours = trisurf(pn, axes=axes, cmap="inferno")
    # axes.set_aspect("auto")
    # axes.set_title("Primal velocity")
    # fig.colorbar(contours)
    #
    # fig = plt.figure()
    # axes = fig.add_subplot(111, projection='3d')
    # contours = trisurf(pn_d, axes=axes, cmap="inferno")
    # axes.set_aspect("auto")
    # axes.set_title("Dual velocity")
    # fig.colorbar(contours)


    # p_err_L2 = max(p_err_L2_vec)
    # q_err_Hrot = max(q_err_Hrot_vec)
    #
    # pd_err_H1 = max(pd_err_H1_vec)
    # qd_err_Hdiv = max(qd_err_Hdiv_vec)

    p_err_L2 = p_err_L2_vec[-1]
    q_err_Hrot = q_err_Hrot_vec[-1]

    pd_err_H1 = pd_err_H1_vec[-1]
    qd_err_Hdiv = qd_err_Hdiv_vec[-1]

    dict_res = {"t_span": t_vec, "power": Hdot_vec, "flow": bdflow_vec, "p_err": p_err_L2, \
                "q_err": q_err_Hrot, "pd_err": pd_err_H1, "qd_err": qd_err_Hdiv}

    return dict_res

# results = compute_err(10, 100, 2, 2)



# plt.figure()
# plt.plot(t_vec, 0.5 * (Ep_vec + Ed_vec), 'g', label=r'Both energies')
# plt.plot(t_vec, Ep_vec, 'r', label=r'Primal energies')
# plt.plot(t_vec, Ed_vec, 'b', label=r'Dual energies')
# plt.xlabel(r'Time [s]')
# plt.title(r'Energy')
# plt.legend()
#
# plt.figure()
# plt.plot(t_vec, 0.5 * (Ep_vec + Ed_vec), 'r', label=r'Both energies')
# plt.plot(t_vec, Es_vec, 'b', label=r'Scattering energy')
# plt.xlabel(r'Time [s]')
# plt.title(r'Energy')
# plt.legend()
#
#
# plt.figure()
# plt.plot(t_vec, Hdot_vec - bdflow_vec, 'b--', label=r'Energy residual')
# plt.xlabel(r'Time [s]')
# plt.title(r'Energy residual')
# plt.legend()
# plt.show()