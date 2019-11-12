# Convergence test for HHJ

from firedrake import *
import numpy as np
np.set_printoptions(threshold=np.inf)
from math import pi, floor

import matplotlib
import matplotlib.pyplot as plt
import petsc4py

matplotlib.rcParams['text.usetex'] = True

def compute_err(n, r):

    h_mesh = 1/n
    print(h_mesh)

    E = Constant(136 * 10**9) # Pa
    rho = Constant(5600)  # kg/m^3
    nu = Constant(0.3)
    h = Constant(0.001)
    bc_input = 'SSSS'

    Lx = 1
    Ly = 1

    D = Constant(E * h ** 3 / (1 - nu ** 2) / 12)
    fl_rot = Constant(12 / (E * h ** 3))
    # Useful Matrices

    # Operators and functions
    def gradSym(u):
        return 0.5 * (nabla_grad(u) + nabla_grad(u).T)
        # return sym(nabla_grad(u))

    def bending_curv(momenta):
        kappa = fl_rot * ((1+nu)*momenta - nu * Identity(2) * tr(momenta))
        return kappa

    def j_operator(v_p, v_q, e_p, e_q):

        j_form = - inner(grad(grad(v_p)), e_q) * dx \
        + jump(grad(v_p), n_ver) * dot(dot(e_q('+'), n_ver('+')), n_ver('+')) * dS \
        + dot(grad(v_p), n_ver) * dot(dot(e_q, n_ver), n_ver) * ds \
        + inner(v_q, grad(grad(e_p))) * dx \
        - dot(dot(v_q('+'), n_ver('+')), n_ver('+')) * jump(grad(e_p), n_ver) * dS \
        - dot(dot(v_q, n_ver), n_ver) * dot(grad(e_p), n_ver) * ds

        return j_form

    # The unit square mesh is divided in :math:`N\times N` quadrilaterals::

    mesh = RectangleMesh(n, n, Lx, Lx, quadrilateral=False)

    # Domain, Subdomains, Boundary, Suboundaries

    # Finite element defition

    Vp = FunctionSpace(mesh, 'CG', r)
    Vq = FunctionSpace(mesh, 'HHJ', r-1)
    V = Vp * Vq

    # Vgradp = VectorFunctionSpace(mesh, 'CG', r)

    n_Vp = V.sub(0).dim()
    n_Vq = V.sub(1).dim()
    n_V = V.dim()
    print(n_V)

    v = TestFunction(V)
    v_p, v_q = split(v)

    e = TrialFunction(V)
    e_p, e_q = split(e)

    al_p = rho * h * e_p
    al_q = bending_curv(e_q)

    dx = Measure('dx')
    ds = Measure('ds')
    dS = Measure("dS")

    m_form = inner(v_p, al_p) * dx + inner(v_q, al_q) * dx

    n_ver = FacetNormal(mesh)
    s_ver = as_vector([-n_ver[1], n_ver[0]])

    # e_mnn = inner(e_q, outer(n_ver, n_ver))
    # v_mnn = inner(v_q, outer(n_ver, n_ver))
    #
    # e_mns = inner(e_q, outer(n_ver, s_ver))
    # v_mns = inner(v_q, outer(n_ver, s_ver))

    j_1 = - inner(grad(grad(v_p)), e_q) * dx \
          + jump(grad(v_p), n_ver) * dot(dot(e_q('+'), n_ver('+')), n_ver('+')) * dS \
          + dot(grad(v_p), n_ver) * dot(dot(e_q, n_ver), n_ver) * ds

    j_2 = + inner(v_q, grad(grad(e_p))) * dx \
          - dot(dot(v_q('+'), n_ver('+')), n_ver('+')) * jump(grad(e_p), n_ver) * dS \
          - dot(dot(v_q, n_ver), n_ver) * dot(grad(e_p), n_ver) * ds


    j_form = j_1 + j_2

    bcs = []

    bc_p = DirichletBC(V.sub(0), Constant(0.0), "on_boundary")
    bc_q = DirichletBC(V.sub(1), Constant(((0.0, 0.0), (0.0, 0.0))), "on_boundary")
    bcs.append(bc_p)
    bcs.append(bc_q)

    t = 0.
    t_ = Constant(t)
    t_1 = Constant(t)
    t_fin = 1        # total simulation time
    x = mesh.coordinates

    beta = 4*pi/t_fin
    w_exact = sin(pi*x[0]/Lx)*sin(pi*x[1]/Ly)*sin(beta*t_)
    grad_wex = as_vector([pi/Lx*cos(pi*x[0]/Lx)*sin(pi*x[1]/Ly)*sin(beta*t_),
                           pi/Ly*sin(pi*x[0]/Lx)*cos(pi*x[1]/Ly)*sin(beta*t_)])

    v_exact = beta * sin(pi*x[0]/Lx)*sin(pi*x[1]/Ly)*cos(beta*t_)
    grad_vex = as_vector([beta* pi / Lx * cos(pi * x[0] / Lx) * sin(pi * x[1] / Ly) * cos(beta * t_),
                          beta * pi / Ly * sin(pi * x[0] / Lx) * cos(pi * x[1] / Ly) * cos(beta * t_)])

    force = sin(pi*x[0]/Lx)*sin(pi*x[1]/Ly)*sin(beta*t_)*(D *((pi/Lx)**2 + (pi/Ly)**2)**2 - rho*h*beta**2)
    force1 = sin(pi*x[0]/Lx)*sin(pi*x[1]/Ly)*sin(beta*t_1)*(D *((pi/Lx)**2 + (pi/Ly)**2)**2 - rho*h*beta**2)

    f_form = v_p*force*dx
    f_form1 = v_p*force1*dx

    # J = assemble(j_form)
    # M = assemble(m_form)

    # Apply boundary conditions to M, J
    # [bc.apply(J) for bc in bcs]
    # [bc.apply(M) for bc in bcs]

    dt = 0.01*h_mesh
    theta = 0.5

    lhs = m_form - dt*theta*j_form

    e_n1 = Function(V, name="e next")
    e_n = Function(V,  name="e old")
    w_n1 = Function(Vp, name="w old")
    w_n = Function(Vp, name="w next")

    # fw_ex = Function(Vp, name="w exact")
    # fv_ex = Function(Vp, name="v exact")
    #
    # fgradw_ex = Function(Vgradp, name="grad w exact")
    # fgradv_ex = Function(Vgradp, name="grad v exact")

    e_n.sub(0).assign(interpolate(v_exact, Vp))

    ep_n, eq_n = e_n.split()

    w_n.assign(Constant(0.0))

    n_t = int(floor(t_fin/dt) + 1)

    err_L2 = np.zeros((n_t,))
    w_atP = np.zeros((n_t,))
    v_atP = np.zeros((n_t,))
    Ppoint = (Lx/14, Ly/3)
    v_atP[0] = ep_n.at(Ppoint)

    # w_interpolator = Interpolator(w_exact, fw_ex)
    # v_interpolator = Interpolator(v_exact, fv_ex)
    #
    # gradw_interpolator = Interpolator(grad_wex, fgradw_ex)
    # gradv_interpolator = Interpolator(grad_vex, fgradv_ex)

    # fw_ex.interpolate(w_exact)
    # fv_ex.interpolate(v_exact)
    # fgradw_ex.interpolate(grad_wex)
    # fgradv_ex.interpolate(grad_vex)

    # err_L2[0] = np.sqrt(assemble(dot(w_n-w_exact, w_n-w_exact) *dx
    #                      + dot(grad(w_n) - grad_wex, grad(w_n) - grad_wex) * dx))

    err_L2[0] = np.sqrt(assemble(dot(ep_n - v_exact, ep_n - v_exact) * dx
                         + dot(grad(ep_n) - grad_vex, grad(ep_n) - grad_vex) * dx))

    t_vec = np.linspace(0, t_fin, num=n_t)

    A = assemble(lhs, bcs=bcs)

    # param = {'ksp_converged_reason': None,
    #                      'ksp_monitor_true_residual': None,
    #                      'ksp_view': None}

    param = {'ksp_gmres_restart': 100, 'ksp_max_it': 10000}

    # print(e_n.vector().get_local())
    for i in range(1, n_t):

        t_.assign(t)
        t_1.assign(t+dt)

        ep_n, eq_n = e_n.split()
        alp_n = rho * h * ep_n
        alq_n = bending_curv(eq_n)

        # rhs = inner(v_p, alp_n) * dx + inner(v_q, alq_n) * dx \
        #       + dt * (1 - theta) * (- inner(grad(grad(v_p)), eq_n) * dx \
        #       + jump(grad(v_p), n_ver) * dot(dot(eq_n('+'), n_ver('+')), n_ver('+')) * dS \
        #       + dot(grad(v_p), n_ver) * dot(dot(eq_n, n_ver), n_ver) * ds \
        #       + inner(v_q, grad(grad(ep_n))) * dx \
        #       - dot(dot(v_q('+'), n_ver('+')), n_ver('+')) * jump(grad(ep_n), n_ver) * dS \
        #       - dot(dot(v_q, n_ver), n_ver) * dot(grad(ep_n), n_ver) * ds)\
        #       + dt*((1-theta)*f_form + theta*f_form1)

        rhs = inner(v_p, alp_n) * dx + inner(v_q, alq_n) * dx \
              + dt * (1 - theta) * j_operator(v_p, v_q, ep_n, eq_n) \
              + dt * ((1 - theta) * f_form + theta * f_form1)

        b = assemble(rhs, bcs=bcs)

        t += dt
        solve(A, e_n1, b, solver_parameters=param)
        ep_n1, eq_n1 = e_n1.split()

        w_n1.assign(w_n + dt/2*(ep_n + ep_n1))

        e_n.assign(e_n1)
        w_n.assign(w_n1)

        w_atP[i] = w_n1.at(Ppoint)
        v_atP[i] = ep_n1.at(Ppoint)
        t_.assign(t)

        ep_n, eq_n = e_n.split()

        # err_L2[i] = np.sqrt(assemble(dot(w_n1-w_exact, w_n1-w_exact) * dx
        #                      + dot(grad(w_n1)-grad_wex, grad(w_n1)-grad_wex) * dx))
        err_L2[i] = np.sqrt(assemble(dot(ep_n - v_exact, ep_n - v_exact) * dx
                         + dot(grad(ep_n) - grad_vex, grad(ep_n) - grad_vex) * dx))

    plt.figure()
    # plt.plot(t_vec, w_atP, 'r-', label=r'approx $w$')
    # plt.plot(t_vec, np.sin(pi*Ppoint[0]/Lx)*np.sin(pi*Ppoint[1]/Ly)*np.sin(beta*t_vec), 'b-', label=r'exact $w$')
    plt.plot(t_vec, v_atP, 'r-', label=r'approx $v$')
    plt.plot(t_vec, beta * np.sin(pi*Ppoint[0]/Lx)*np.sin(pi*Ppoint[1]/Ly) * np.cos(beta * t_vec), 'b-', label=r'exact $v$')
    plt.xlabel(r'Time [s]')
    plt.title(r'Displacement at $(L_x/2, L_y/2)$')
    plt.legend()
    # plt.show()

    err_quad = np.sqrt(np.sum(dt*np.power(err_L2, 2)))
    err_max = max(err_L2)
    err_last = err_L2[-1]
    return err_last, err_max, err_quad



n_h = 4
n1_vec = np.array([2**(i) for i in range(n_h)])
n2_vec = np.array([2**(i) for i in range(n_h)])
h1_vec = 1./n1_vec
h2_vec = 1./n2_vec

err_vec_r1 = np.zeros((n_h,))
errInf_vec_r1 = np.zeros((n_h,))
errQuad_vec_r1 = np.zeros((n_h,))

err_vec_r2 = np.zeros((n_h,))
errInf_vec_r2 = np.zeros((n_h,))
errQuad_vec_r2 = np.zeros((n_h,))

err_vec_r3 = np.zeros((n_h,))
errInf_vec_r3 = np.zeros((n_h,))
errQuad_vec_r3 = np.zeros((n_h,))

r1_ext_atF = np.zeros((n_h-1,))
r2_ext_atF = np.zeros((n_h-1,))
r3_ext_atF = np.zeros((n_h-1,))

r1_ext_max = np.zeros((n_h-1,))
r2_ext_max = np.zeros((n_h-1,))
r3_ext_max = np.zeros((n_h-1,))

r1_ext_L2 = np.zeros((n_h-1,))
r2_ext_L2 = np.zeros((n_h-1,))
r3_ext_L2 = np.zeros((n_h-1,))

for i in range(n_h):
    err_vec_r1[i], errInf_vec_r1[i], errQuad_vec_r1[i] = compute_err(n1_vec[i], 1)
    err_vec_r2[i], errInf_vec_r2[i], errQuad_vec_r2[i] = compute_err(n2_vec[i], 2)
    err_vec_r3[i], errInf_vec_r3[i], errQuad_vec_r3[i] = compute_err(n2_vec[i], 3)

    if i>0:
        r1_ext_atF[i-1] = np.log(err_vec_r1[i]/err_vec_r1[i-1])/np.log(h1_vec[i]/h1_vec[i-1])
        r1_ext_max[i-1] = np.log(errInf_vec_r1[i]/errInf_vec_r1[i-1])/np.log(h1_vec[i]/h1_vec[i-1])
        r1_ext_L2[i-1] = np.log(errQuad_vec_r1[i]/errQuad_vec_r1[i-1])/np.log(h1_vec[i]/h1_vec[i-1])

        r2_ext_atF[i-1] = np.log(err_vec_r2[i]/err_vec_r2[i-1])/np.log(h2_vec[i]/h2_vec[i-1])
        r2_ext_max[i-1] = np.log(errInf_vec_r2[i]/errInf_vec_r2[i-1])/np.log(h2_vec[i]/h2_vec[i-1])
        r2_ext_L2[i-1] = np.log(errQuad_vec_r2[i]/errQuad_vec_r2[i-1])/np.log(h2_vec[i]/h2_vec[i-1])

        r3_ext_atF[i-1] = np.log(err_vec_r3[i]/err_vec_r3[i-1])/np.log(h2_vec[i]/h2_vec[i-1])
        r3_ext_max[i-1] = np.log(errInf_vec_r3[i]/errInf_vec_r3[i-1])/np.log(h2_vec[i]/h2_vec[i-1])
        r3_ext_L2[i-1] = np.log(errQuad_vec_r3[i]/errQuad_vec_r3[i-1])/np.log(h2_vec[i]/h2_vec[i-1])

order_r1_atF = np.polyfit(np.log(h1_vec), np.log(err_vec_r1), 1)[0]
order_r1_max = np.polyfit(np.log(h1_vec), np.log(errInf_vec_r1), 1)[0]
order_r1_L2 = np.polyfit(np.log(h1_vec), np.log(errQuad_vec_r1), 1)[0]

print("Estimated order of convergence r=1 at T fin: " + str(r1_ext_atF))
print("Interpolated order of convergence r=1 at T fin: " + str(order_r1_atF))
print("Estimated order of convergence r=1 Linf: " + str(r1_ext_max))
print("Interpolated order of convergence r=1 Linf: " + str(order_r1_max))
print("Estimated order of convergence r=1 L2: " + str(r1_ext_L2))
print("Interpolated order of convergence r=1 L2: " + str(order_r1_L2))
print("")


order_r2_atF = np.polyfit(np.log(h2_vec), np.log(err_vec_r2), 1)[0]
order_r2_max = np.polyfit(np.log(h2_vec), np.log(errInf_vec_r2), 1)[0]
order_r2_L2 = np.polyfit(np.log(h2_vec), np.log(errQuad_vec_r2), 1)[0]

print("Estimated order of convergence r=2 at T fin: " + str(r2_ext_atF))
print("Interpolated order of convergence r=2 at T fin: " + str(order_r2_atF))
print("Estimated order of convergence r=2 Linf: " + str(r2_ext_max))
print("Interpolated order of convergence r=2 Linf: " + str(order_r2_max))
print("Estimated order of convergence r=2 L2: " + str(r2_ext_L2))
print("Interpolated order of convergence r=2 L2: " + str(order_r2_L2))
print("")

order_r3_atF = np.polyfit(np.log(h2_vec), np.log(err_vec_r3), 1)[0]
order_r3_max = np.polyfit(np.log(h2_vec), np.log(errInf_vec_r3), 1)[0]
order_r3_L2 = np.polyfit(np.log(h2_vec), np.log(errQuad_vec_r3), 1)[0]

print("Estimated order of convergence r=3 at T fin: " + str(r3_ext_atF))
print("Interpolated order of convergence r=3 at T fin: " + str(order_r3_atF))
print("Estimated order of convergence r=3 Linf: " + str(r3_ext_max))
print("Interpolated order of convergence r=3 Linf: " + str(order_r3_max))
print("Estimated order of convergence r=3 L2: " + str(r3_ext_L2))
print("Interpolated order of convergence r=3 L2: " + str(order_r3_L2))
print("")

plt.figure()
plt.plot(np.log(h1_vec), np.log(err_vec_r1), '-og', label='HHJ 1')
plt.plot(np.log(h1_vec), np.log(errInf_vec_r1), '-+g', label='HHJ 1 $L^\infty$')
plt.plot(np.log(h1_vec), np.log(errQuad_vec_r1), '-*g', label='HHJ 1 $L^2$')
plt.plot(np.log(h1_vec), np.log(h1_vec), '-vg', label=r'$h$')

plt.plot(np.log(h2_vec), np.log(err_vec_r2), '-or', label='HHJ 2')
plt.plot(np.log(h2_vec), np.log(errInf_vec_r2), '-+r', label='HHJ 2 $L^\infty$')
plt.plot(np.log(h2_vec), np.log(errQuad_vec_r2), '-*r', label='HHJ 2 $L^2$')
plt.plot(np.log(h2_vec), np.log(h2_vec**2), '-vr', label=r'$h^2$')

plt.plot(np.log(h2_vec), np.log(err_vec_r3), '-ob', label='HHJ 3')
plt.plot(np.log(h2_vec), np.log(errInf_vec_r3), '-+b', label='HHJ 3 $L^\infty$')
plt.plot(np.log(h2_vec), np.log(errQuad_vec_r3), '-*b', label='HHJ 3 $L^2$')
plt.plot(np.log(h2_vec), np.log(h2_vec**3), '-vb', label=r'$h^3$')

plt.xlabel(r'Mesh size')
plt.title(r'Error at $T_f$')
plt.legend()
plt.show()



