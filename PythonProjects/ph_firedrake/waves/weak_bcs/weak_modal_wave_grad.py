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
# from time import sleaep


def compute_eig(n_el, n_eig_exact, deg=1):
    """Compute the numerical solution of the wave equation with the dual field method

        Parameters:
        n_el: number of elements for the discretization
        n_t: number of time instants
        deg: polynomial degree for finite
        Returns:
        some plots

       """

    Lx = pi
    Ly = pi
    mesh = RectangleMesh(n_el, n_el, Lx, Ly, quadrilateral=False)
    n_ver = FacetNormal(mesh)

    P_1 = FiniteElement("CG", triangle, deg)
    P_2 = FiniteElement("RT", triangle, deg, variant='integral')

    V_1 = FunctionSpace(mesh, P_1)
    V_2 = FunctionSpace(mesh, P_2)

    V = V_1 * V_2

    v = TestFunction(V)
    v_1, v_2 = split(v)

    e = TrialFunction(V)
    e_1, e_2 = split(e)

    dx = Measure('dx')
    ds = Measure('ds')

    x, y = SpatialCoordinate(mesh)

    ## Exact eigenvalues

    lambda_vec = np.empty((n_eig_exact, ))
    mu_vec = np.empty((n_eig_exact, ))

    for i in range(n_eig_exact):
        lambda_vec[i] = ((i+1)*pi/Lx)**2
        mu_vec[i] = (i*pi/Ly)**2


    # in_cond = project(as_vector([v_ex, sig_ex[0], sig_ex[1], v_ex, sig_ex[0], sig_ex[1]]), V)

    p0, q0, p0_d, q0_d = split(in_cond)

    E_L2Hdiv = 0.5*(inner(p0, p0) * dx + inner(q0_d, q0_d) * dx)
    E_H1Hrot = 0.5*(inner(p0_d, p0_d) * dx + inner(q0, q0) * dx)

    Hdot = div(q0_d) * p0_d * dx + inner(grad(p0_d), q0_d) * dx
    bdflow = p0_d * dot(q0_d, n_ver) * ds

    bdflow_ex = v_ex * dot(sig_ex, n_ver) * ds

    m_form = inner(vp, Dt(p0)) * dx + inner(vq, Dt(q0)) * dx + inner(vp_d, Dt(p0_d)) * dx + inner(vq_d, Dt(q0_d)) * dx

    # Check for sign in adjoint system
    j_form = dot(vp, div(q0_d)) * dx + dot(vq, grad(p0_d)) * dx - dot(grad(vp_d), q0) * dx - dot(div(vq_d), p0) * dx \
               + vp_d * dot(q0_d, n_ver) * ds + dot(vq_d, n_ver) * p0_d * ds
    # Form defininig the problem
    f_form = m_form - j_form


    # Method of manifactured solution: check demo on Firedrake irksome
    # rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact))

    dt = Constant(t_fin / n_t)
    butcher_tableau = GaussLegendre(3)
    # butcher_tableau = LobattoIIIA(2)

    params = {"mat_type": "aij",
              "snes_type": "ksponly",
              "ksp_type": "preonly",
              "pc_type": "lu"}

    bc = DirichletBC(V.sub(2), v_ex, "on_boundary")
    # bc = DirichletBC(V.sub(3), sig_ex, "on_boundary")

    stepper = TimeStepper(f_form, butcher_tableau, t, dt, in_cond,
                          bcs=bc, solver_parameters=params)

    E_L2Hdiv_vec = np.zeros((1+n_t, ))
    E_H1Hrot_vec = np.zeros((1+n_t, ))

    Hdot_vec = np.zeros((1 + n_t,))
    bdflow_vec = np.zeros((1 + n_t,))
    bdflow_ex_vec = np.zeros((1 + n_t,))

    Ppoint = (L/5, L/5)

    p_P = np.zeros((1+n_t,))
    p_P[0] = interpolate(v_ex, Vp).at(Ppoint)

    pd_P = np.zeros((1+n_t, ))
    pd_P[0] = interpolate(v_ex, Vp_d).at(Ppoint)

    t_vec = np.linspace(0, n_t * float(dt), 1 + n_t)

    Hdot_vec[0] = assemble(Hdot)
    bdflow_vec[0] = assemble(bdflow)
    bdflow_ex_vec[0] = assemble(bdflow_ex)

    E_L2Hdiv_vec[0] = assemble(E_L2Hdiv)
    E_H1Hrot_vec[0] = assemble(E_H1Hrot)

    print("Time    Energy")
    print("==============")

    pn = Function(Vp, name="p primal at t_n")
    pn_d = Function(Vp_d, name="p dual at t_n")

    err_p = Function(Vp, name="p error primal at t_n")
    err_pd = Function(Vp_d, name="p error dual at t_n")

    for ii in tqdm(range(n_t)):
        stepper.advance()

        Hdot_vec[ii+1] = assemble(Hdot)
        bdflow_vec[ii+1] = assemble(bdflow)

        E_L2Hdiv_vec[ii+1] = assemble(E_L2Hdiv)
        E_H1Hrot_vec[ii+1] = assemble(E_H1Hrot)

        pn.assign(interpolate(p0, Vp))
        pn_d.assign(interpolate(p0_d, Vp_d))

        p_P[ii+1] = pn.at(Ppoint)
        pd_P[ii+1] = pn_d.at(Ppoint)

        t.assign(float(t) + float(dt))
        bdflow_ex_vec[ii + 1] = assemble(bdflow_ex)

    err_p.assign(pn - interpolate(v_ex, Vp))
    err_pd.assign(pn_d - interpolate(v_ex, Vp_d))

    # err_p.assign(interpolate(v_ex, Vp))
    # err_pd.assign(interpolate(v_ex, Vp_d))

    print(r"Initial and final L2Hdiv energy:")
    print(r"Inital: ", E_L2Hdiv_vec[0])
    print(r"Final: ", E_L2Hdiv_vec[-1])
    print(r"Delta: ", E_L2Hdiv_vec[-1] - E_L2Hdiv_vec[0])

    print(r"Initial and final H1Hrot energy:")
    print(r"Inital: ", E_H1Hrot_vec[0])
    print(r"Final: ", E_H1Hrot_vec[-1])
    print(r"Delta: ", E_H1Hrot_vec[-1] - E_H1Hrot_vec[0])

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

    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    contours = trisurf(pn, axes=axes, cmap="inferno")
    axes.set_aspect("auto")
    axes.set_title("Primal velocity")
    fig.colorbar(contours)

    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    contours = trisurf(pn_d, axes=axes, cmap="inferno")
    axes.set_aspect("auto")
    axes.set_title("Dual velocity")
    fig.colorbar(contours)

    plt.figure()
    plt.plot(t_vec, p_P, 'r-', label=r'primal $p$')
    plt.plot(t_vec, pd_P, 'b-', label=r'dual $p$')
    plt.plot(t_vec, om_t * np.sin(om_x * Ppoint[0] + phi_x) * np.sin(om_y * Ppoint[1] + phi_y)\
             * np.cos(om_t * t_vec + phi_t), 'g-',
             label=r'exact $p$')
    plt.xlabel(r'Time [s]')
    plt.title(r'$p$ at ' + str(Ppoint))
    plt.legend()

    dict_res = {"t_span": t_vec, "power": Hdot_vec, \
                "flow": bdflow_vec, "flow_ex": bdflow_ex_vec, "energy_L2Hdiv": E_L2Hdiv_vec, "energy_H1Hrot": E_H1Hrot_vec}

    return dict_res

n_elem = 5
pol_deg = 2

n_time = 50
t_fin = 1
results = compute_sol(n_elem, n_time, pol_deg, t_fin)

t_vec = results["t_span"]
Hdot_vec = results["power"]
bdflow_vec = results["flow"]
bdflow_ex_vec = results["flow_ex"]

EL2Hdiv = results["energy_L2Hdiv"]
EH1Hcurl = results["energy_H1Hrot"]

plt.figure()
plt.plot(t_vec, EL2Hdiv, 'r', label=r'L2Hdiv')
plt.plot(t_vec, EH1Hcurl, 'b', label=r'H1Hrot')
plt.xlabel(r'Time [s]')
plt.title(r' Mixed energy')
plt.legend()


plt.figure()
plt.plot(t_vec, Hdot_vec - bdflow_vec, 'ro', label=r'Energy residual')
plt.xlabel(r'Time [s]')
plt.title(r'Energy residual')
plt.legend()

plt.figure()
plt.plot(t_vec, bdflow_vec, 'r', label=r'bd flow')
plt.plot(t_vec, bdflow_ex_vec, 'b', label=r'bd flow ex')
plt.xlabel(r'Time [s]')
plt.title(r'Boundary flow')
plt.legend()


diffH_L2Hdiv = np.diff(EL2Hdiv)
diffH_H1Hcurl = np.diff(EH1Hcurl)
Delta_t = np.diff(t_vec)
int_bdflow = np.zeros((n_time, ))

for i in range(n_time):
    int_bdflow[i] = 0.5*Delta_t[i]*(bdflow_vec[i+1] + bdflow_vec[i])

plt.figure()
plt.plot(t_vec[1:], diffH_L2Hdiv, 'ro', label=r'Delta H L2Hdiv')
plt.plot(t_vec[1:], diffH_H1Hcurl, 'b--', label=r'Delta H H1Hrot')
plt.plot(t_vec[1:], int_bdflow, '*-', label=r'Bd flow int')
plt.xlabel(r'Time [s]')
plt.title(r'Energy balance')
plt.legend()

plt.show()