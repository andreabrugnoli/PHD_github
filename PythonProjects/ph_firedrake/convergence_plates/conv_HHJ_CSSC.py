# Convergence test for HHJ

from firedrake import *
import numpy as np
np.set_printoptions(threshold=np.inf)
from math import pi, floor

import matplotlib
import matplotlib.pyplot as plt
import petsc4py

matplotlib.rcParams['text.usetex'] = True
save_res = False
bc_input = 'CSSC'


def compute_constants():
    A = np.array([[np.cosh(pi), - np.cosh(pi), -np.sinh(pi), np.sinh(pi)],
                  [-pi * np.sinh(pi), pi * np.sinh(pi) + np.cosh(pi), pi * np.cosh(pi),
                   -(np.sinh(pi) + pi * np.cosh(pi))],
                  [np.cosh(pi), np.cosh(pi), np.sinh(pi), np.sinh(pi)],
                  [pi * np.sinh(pi), pi * np.sinh(pi) + np.cosh(pi), pi * np.cosh(pi),
                   np.sinh(pi) + pi * np.cosh(pi)]])

    b = np.array([np.sin(pi), -pi * np.cos(pi), -np.sin(pi), -pi * np.cos(pi)])

    a, b, c, d = np.linalg.solve(A, b)

    return a, b, c, d


def compute_err(n, r):

    h_mesh = 1/n

    rho = Constant(5600)  # kg/m^3
    h = Constant(0.001)

    Lx = 2
    Ly = 2

    def bending_curv(momenta):
        kappa = momenta
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

    # The boundary edges in this mesh are numbered as follows:

    # 1: plane x == 0
    # 2: plane x == 1
    # 3: plane y == 0
    # 4: plane y == 1

    j_form = j_operator(v_p, v_q, e_p, e_q)

    bcp_C_l = DirichletBC(V.sub(0), Constant(0.0), 1)
    bcp_C_r = DirichletBC(V.sub(0), Constant(0.0), 2)

    bcp_S_d = DirichletBC(V.sub(0), Constant(0.0), 3)
    bcp_S_u = DirichletBC(V.sub(0), Constant(0.0), 4)

    bcq_S_d = DirichletBC(V.sub(1), Constant(((0.0, 0.0), (0.0, 0.0))), 3)
    bcq_S_u = DirichletBC(V.sub(1), Constant(((0.0, 0.0), (0.0, 0.0))), 4)

    bcs = [bcp_C_l, bcp_C_r, bcp_S_d, bcp_S_u, bcq_S_d, bcq_S_u]

    t = 0.
    t_ = Constant(t)
    t_1 = Constant(t)
    t_fin = 1        # total simulation time
    x_til, y_til = SpatialCoordinate(mesh)
    x = x_til - 1
    y = y_til - 1

    beta = 1

    a, b, c, d = compute_constants()
    wst = ((a + b*x)*cosh(pi*x) + (c+d*x)*sinh(pi*x) + sin(pi*x))*sin(pi*y)
    fst = 4*pi**4*sin(pi*x)*sin(pi*y)

    wst_x = (b*cosh(pi*x) + (a+b*x)*pi*sinh(pi*x) + d*sinh(pi*x) + (c+d*x)*pi*cosh(pi*x) + pi*cos(pi*x))*sin(pi*y)
    wst_y = ((a + b*x)*cosh(pi*x) + (c+d*x)*sinh(pi*x) + sin(pi*x))*pi*cos(pi*y)

    wst_xx = (2*b*pi*sinh(pi*x) + (a+b*x)*pi**2*cosh(pi*x) + 2*d*pi*cosh(pi*x) + (c+d*x)*pi**2*sinh(pi*x) - pi**2*sin(pi*x))*sin(pi*y)
    wst_yy = - ((a + b*x)*cosh(pi*x) + (c+d*x)*sinh(pi*x) + sin(pi*x))*pi**2*sin(pi*y)
    wst_xy = (b*cosh(pi*x) + (a+b*x)*pi*sinh(pi*x) + d*sinh(pi*x) + (c+d*x)*pi*cosh(pi*x) + pi*cos(pi*x))*pi*cos(pi*y)

    wst_xxx = (3*b*pi**2*cosh(pi*x) + (a+b*x)*pi**3*sinh(pi*x) + 3*d*pi**2*sinh(pi*x) + (c+d*x)*pi**3*cosh(pi*x)\
               - pi**3*cos(pi*x))*sin(pi*y)

    wst_xyy = -(b * cosh(pi * x) + (a + b * x) * pi * sinh(pi * x) + d * sinh(pi * x) + (c + d * x) * pi * cosh(
        pi * x) + pi * cos(pi * x)) * pi**2 * sin(pi * y)

    # tol = 1e-13
    # print(abs(interpolate(wst, Vp).at(0, Ly/4)))
    # print(abs(interpolate(wst_x, Vp).at(0, Ly/4)))
    # print(abs(interpolate(wst_xx, Vp).at(Lx, Ly/4)))
    # print(abs(interpolate(wst_xxx + 2*wst_xyy, Vp).at(Lx, Ly/4)))
    # assert(abs(interpolate(wst, Vp).at(0, Ly/4))<=tol)
    # assert (abs(interpolate(wst_x, Vp).at(0, Ly/4)) <= tol)
    # assert(abs(interpolate(wst_xx, Vp).at(Lx, Ly/4))<=tol)
    # assert (abs(interpolate(wst_xxx + 2*wst_xyy, Vp).at(Lx, Ly/4)) <=tol)

    wdyn = wst * sin(beta*t_)
    wdyn_xx = wst_xx*sin(beta*t_)
    wdyn_yy = wst_yy*sin(beta*t_)
    wdyn_xy = wst_xy*sin(beta*t_)

    dt_w = beta*wst*cos(beta*t_)
    dt_w_x = beta*wst_x*cos(beta*t_)
    dt_w_y = beta*wst_y*cos(beta*t_)

    v_ex = dt_w
    grad_vex = as_vector([dt_w_x, dt_w_y])

    kappa_ex = as_tensor([[wdyn_xx, wdyn_xy],
                          [wdyn_xy, wdyn_yy]])

    sigma_ex = kappa_ex

    dtt_w = -beta**2*wst*sin(beta*t_)
    dtt_w1 = -beta**2*wst*sin(beta*t_1)

    fdyn = fst * sin(beta * t_) + rho*h*dtt_w
    fdyn1 = fst * sin(beta * t_1) + rho * h * dtt_w1

    f_form = v_p*fdyn*dx
    f_form1 = v_p*fdyn1*dx

    J = assemble(j_form)
    M = assemble(m_form)

    # Apply boundary conditions to M, J
    [bc.apply(J) for bc in bcs]
    [bc.apply(M) for bc in bcs]

    dt = 0.1*h_mesh
    theta = 0.5

    lhs = m_form - dt*theta*j_form

    e_n1 = Function(V, name="e next")
    e_n = Function(V,  name="e old")
    w_n1 = Function(Vp, name="w old")
    w_n = Function(Vp, name="w next")

    e_n.sub(0).assign(interpolate(v_ex, Vp))

    ep_n, eq_n = e_n.split()

    w_n.assign(Constant(0.0))

    n_t = int(floor(t_fin/dt) + 1)

    # w_err_H1 = np.zeros((n_t,))
    v_err_H1 = np.zeros((n_t,))
    sig_err_L2 = np.zeros((n_t,))

    w_atP = np.zeros((n_t,))
    v_atP = np.zeros((n_t,))
    Ppoint = (Lx/6, Ly/3)
    v_atP[0] = ep_n.at(Ppoint)

    # w_err_H1[0] = np.sqrt(assemble(dot(w_n-w_exact, w_n-w_exact) *dx
    #                      + dot(grad(w_n) - grad_wex, grad(w_n) - grad_wex) * dx))
    v_err_H1[0] = np.sqrt(assemble(dot(ep_n - v_ex, ep_n - v_ex) * dx
                                   + dot(grad(ep_n) - grad_vex, grad(ep_n) - grad_vex) * dx))

    sig_err_L2[0] = np.sqrt(assemble(inner(eq_n - sigma_ex, eq_n - sigma_ex) * dx))

    t_vec = np.linspace(0, t_fin, num=n_t)

    A = assemble(lhs, bcs=bcs, mat_type='aij')

    # param = {'ksp_converged_reason': None,
    #                      'ksp_monitor_true_residual': None,
    #                      'ksp_view': None}

    param = {"ksp_type": "preonly", "pc_type": "lu"}

    # print(e_n.vector().get_local())
    for i in range(1, n_t):

        t_.assign(t)
        t_1.assign(t+dt)

        ep_n, eq_n = e_n.split()
        alp_n = rho * h * ep_n
        alq_n = bending_curv(eq_n)

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

        # w_err_H1[i] = np.sqrt(assemble(dot(w_n1-w_exact, w_n1-w_exact) * dx
        #                      + dot(grad(w_n1)-grad_wex, grad(w_n1)-grad_wex) * dx))
        v_err_H1[i] = np.sqrt(assemble(dot(ep_n1 - v_ex, ep_n1 - v_ex) * dx
                                       + dot(grad(ep_n1) - grad_vex, grad(ep_n1) - grad_vex) * dx))

        sig_err_L2[i] = np.sqrt(assemble(inner(eq_n1 - sigma_ex, eq_n1 - sigma_ex) * dx))

    plt.figure()
    wst_atP = interpolate(wst, Vp).at(Ppoint)
    vex_atP = wst_atP * beta * np.cos(beta * t_vec)
    plt.plot(t_vec, v_atP, 'r-', label=r'approx $v$')
    plt.plot(t_vec, vex_atP, 'b-', label=r'exact $v$')
    plt.xlabel(r'Time [s]')
    plt.title(r'Displacement at' + str(Ppoint))
    plt.legend()
    # plt.show()

    # v_err_last = w_err_H1[-1]
    # v_err_max = max(w_err_H1)
    # v_err_quad = np.sqrt(np.sum(dt * np.power(w_err_H1, 2)))

    v_err_last = v_err_H1[-1]
    v_err_max = max(v_err_H1)
    v_err_quad = np.sqrt(np.sum(dt * np.power(v_err_H1, 2)))

    sig_err_last = sig_err_L2[-1]
    sig_err_max = max(sig_err_L2)
    sig_err_quad = np.sqrt(np.sum(dt * np.power(sig_err_L2, 2)))

    return v_err_last, v_err_max, v_err_quad, sig_err_last, sig_err_max, sig_err_quad


n_h = 2
n1_vec = np.array([2**(i+2) for i in range(n_h)])
n2_vec = np.array([2**(i+1) for i in range(n_h)])
h1_vec = 1./n1_vec
h2_vec = 1./n2_vec

v_err_r1 = np.zeros((n_h,))
v_errInf_r1 = np.zeros((n_h,))
v_errQuad_r1 = np.zeros((n_h,))

v_err_r2 = np.zeros((n_h,))
v_errInf_r2 = np.zeros((n_h,))
v_errQuad_r2 = np.zeros((n_h,))

v_err_r3 = np.zeros((n_h,))
v_errInf_r3 = np.zeros((n_h,))
v_errQuad_r3 = np.zeros((n_h,))

v_r1_atF = np.zeros((n_h-1,))
v_r1_max = np.zeros((n_h-1,))
v_r1_L2 = np.zeros((n_h-1,))

v_r2_atF = np.zeros((n_h-1,))
v_r2_max = np.zeros((n_h-1,))
v_r2_L2 = np.zeros((n_h-1,))

v_r3_atF = np.zeros((n_h-1,))
v_r3_max = np.zeros((n_h-1,))
v_r3_L2 = np.zeros((n_h-1,))

sig_err_r1 = np.zeros((n_h,))
sig_errInf_r1 = np.zeros((n_h,))
sig_errQuad_r1 = np.zeros((n_h,))

sig_err_r2 = np.zeros((n_h,))
sig_errInf_r2 = np.zeros((n_h,))
sig_errQuad_r2 = np.zeros((n_h,))

sig_err_r3 = np.zeros((n_h,))
sig_errInf_r3 = np.zeros((n_h,))
sig_errQuad_r3 = np.zeros((n_h,))

sig_r1_atF = np.zeros((n_h-1,))
sig_r1_max = np.zeros((n_h-1,))
sig_r1_L2 = np.zeros((n_h-1,))

sig_r2_atF = np.zeros((n_h-1,))
sig_r2_max = np.zeros((n_h-1,))
sig_r2_L2 = np.zeros((n_h-1,))

sig_r3_atF = np.zeros((n_h-1,))
sig_r3_max = np.zeros((n_h-1,))
sig_r3_L2 = np.zeros((n_h-1,))


for i in range(n_h):
    v_err_r1[i], v_errInf_r1[i], v_errQuad_r1[i],\
    sig_err_r1[i], sig_errInf_r1[i], sig_errQuad_r1[i] = compute_err(n1_vec[i], 1)
    v_err_r2[i], v_errInf_r2[i], v_errQuad_r2[i], sig_err_r2[i],\
    sig_errInf_r2[i], sig_errQuad_r2[i] = compute_err(n1_vec[i], 2)
    v_err_r3[i], v_errInf_r3[i], v_errQuad_r3[i], sig_err_r3[i],\
    sig_errInf_r3[i], sig_errQuad_r3[i] = compute_err(n2_vec[i], 3)

    if i>0:
        v_r1_atF[i-1] = np.log(v_err_r1[i]/v_err_r1[i-1])/np.log(h1_vec[i]/h1_vec[i-1])
        v_r1_max[i-1] = np.log(v_errInf_r1[i]/v_errInf_r1[i-1])/np.log(h1_vec[i]/h1_vec[i-1])
        v_r1_L2[i-1] = np.log(v_errQuad_r1[i]/v_errQuad_r1[i-1])/np.log(h1_vec[i]/h1_vec[i-1])

        v_r2_atF[i-1] = np.log(v_err_r2[i]/v_err_r2[i-1])/np.log(h1_vec[i]/h1_vec[i-1])
        v_r2_max[i-1] = np.log(v_errInf_r2[i]/v_errInf_r2[i-1])/np.log(h1_vec[i]/h1_vec[i-1])
        v_r2_L2[i-1] = np.log(v_errQuad_r2[i]/v_errQuad_r2[i-1])/np.log(h1_vec[i]/h1_vec[i-1])

        v_r3_atF[i-1] = np.log(v_err_r3[i]/v_err_r3[i-1])/np.log(h2_vec[i]/h2_vec[i-1])
        v_r3_max[i-1] = np.log(v_errInf_r3[i]/v_errInf_r3[i-1])/np.log(h2_vec[i]/h2_vec[i-1])
        v_r3_L2[i-1] = np.log(v_errQuad_r3[i]/v_errQuad_r3[i-1])/np.log(h2_vec[i]/h2_vec[i-1])

        sig_r1_atF[i - 1] = np.log(sig_err_r1[i] / sig_err_r1[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])
        sig_r1_max[i - 1] = np.log(sig_errInf_r1[i] / sig_errInf_r1[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])
        sig_r1_L2[i - 1] = np.log(sig_errQuad_r1[i] / sig_errQuad_r1[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])

        sig_r2_atF[i - 1] = np.log(sig_err_r2[i] / sig_err_r2[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])
        sig_r2_max[i - 1] = np.log(sig_errInf_r2[i] / sig_errInf_r2[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])
        sig_r2_L2[i - 1] = np.log(sig_errQuad_r2[i] / sig_errQuad_r2[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])

        sig_r3_atF[i - 1] = np.log(sig_err_r3[i] / sig_err_r3[i - 1]) / np.log(h2_vec[i] / h2_vec[i - 1])
        sig_r3_max[i - 1] = np.log(sig_errInf_r3[i] / sig_errInf_r3[i - 1]) / np.log(h2_vec[i] / h2_vec[i - 1])
        sig_r3_L2[i - 1] = np.log(sig_errQuad_r3[i] / sig_errQuad_r3[i - 1]) / np.log(h2_vec[i] / h2_vec[i - 1])


path_res = "./convergence_results_kirchhoff/"
if save_res:
    np.save(path_res + bc_input + "_h1", h1_vec)
    np.save(path_res + bc_input + "_h2", h1_vec)
    np.save(path_res + bc_input + "_h3", h2_vec)

    np.save(path_res + bc_input + "_v_errF_r1", v_err_r1)
    np.save(path_res + bc_input + "_v_errInf_r1", v_errInf_r1)
    np.save(path_res + bc_input + "_v_errQuad_r1", v_errQuad_r1)

    np.save(path_res + bc_input + "_v_errF_r2", v_err_r2)
    np.save(path_res + bc_input + "_v_errInf_r2", v_errInf_r2)
    np.save(path_res + bc_input + "_v_errQuad_r2", v_errQuad_r2)

    np.save(path_res + bc_input + "_v_errF_r3", v_err_r3)
    np.save(path_res + bc_input + "_v_errInf_r3", v_errInf_r3)
    np.save(path_res + bc_input + "_v_errQuad_r3", v_errQuad_r3)

    np.save(path_res + bc_input + "_sig_errF_r1", sig_err_r1)
    np.save(path_res + bc_input + "_sig_errInf_r1", sig_errInf_r1)
    np.save(path_res + bc_input + "_sig_errQuad_r1", sig_errQuad_r1)

    np.save(path_res + bc_input + "_sig_errF_r2", sig_err_r2)
    np.save(path_res + bc_input + "_sig_errInf_r2", sig_errInf_r2)
    np.save(path_res + bc_input + "_sig_errQuad_r2", sig_errQuad_r2)

    np.save(path_res + bc_input + "_sig_errF_r3", sig_err_r3)
    np.save(path_res + bc_input + "_sig_errInf_r3", sig_errInf_r3)
    np.save(path_res + bc_input + "_sig_errQuad_r3", sig_errQuad_r3)

v_r1int_atF = np.polyfit(np.log(h1_vec), np.log(v_err_r1), 1)[0]
v_r1int_max = np.polyfit(np.log(h1_vec), np.log(v_errInf_r1), 1)[0]
v_r1int_L2 = np.polyfit(np.log(h1_vec), np.log(v_errQuad_r1), 1)[0]

print("Estimated order of convergence r=1 for v at T fin: " + str(v_r1_atF))
print("Interpolated order of convergence r=1 for v at T fin: " + str(v_r1int_atF))
print("Estimated order of convergence r=1 for v Linf: " + str(v_r1_max))
print("Interpolated order of convergence r=1 for v Linf: " + str(v_r1int_max))
print("Estimated order of convergence r=1 for v L2: " + str(v_r1_L2))
print("Interpolated order of convergence r=1 for v L2: " + str(v_r1int_L2))
print("")

v_r2int = np.polyfit(np.log(h1_vec), np.log(v_err_r2), 1)[0]
v_r2int_max = np.polyfit(np.log(h1_vec), np.log(v_errInf_r2), 1)[0]
v_r2int_L2 = np.polyfit(np.log(h1_vec), np.log(v_errQuad_r2), 1)[0]

print("Estimated order of convergence r=2 for v at T fin: " + str(v_r2_atF))
print("Interpolated order of convergence r=2 for v at T fin: " + str(v_r2int))
print("Estimated order of convergence r=2 for v Linf: " + str(v_r2_max))
print("Interpolated order of convergence r=2 for v Linf: " + str(v_r2int_max))
print("Estimated order of convergence r=2 for v L2: " + str(v_r2_L2))
print("Interpolated order of convergence r=2 for v L2: " + str(v_r2int_L2))
print("")

v_r3int = np.polyfit(np.log(h2_vec), np.log(v_err_r3), 1)[0]
v_r3int_max = np.polyfit(np.log(h2_vec), np.log(v_errInf_r3), 1)[0]
v_r3int_L2 = np.polyfit(np.log(h2_vec), np.log(v_errQuad_r3), 1)[0]

print("Estimated order of convergence r=3 for v at T fin: " + str(v_r3_atF))
print("Interpolated order of convergence r=3 for v at T fin: " + str(v_r3int))
print("Estimated order of convergence r=3 for v Linf: " + str(v_r3_max))
print("Interpolated order of convergence r=3 for v Linf: " + str(v_r3int_max))
print("Estimated order of convergence r=3 for v L2: " + str(v_r3_L2))
print("Interpolated order of convergence r=3 for v L2: " + str(v_r3int_L2))
print("")

plt.figure()

# plt.plot(np.log(h1_vec), np.log(v_r1_atF), ':o', label='HHJ 1')
plt.plot(np.log(h1_vec), np.log(v_errInf_r1), '-.+', label='HHJ 1 $L^\infty$')
plt.plot(np.log(h1_vec), np.log(v_errQuad_r1), '--*', label='HHJ 1 $L^2$')
plt.plot(np.log(h1_vec), np.log(h1_vec), '-v', label=r'$h$')

# plt.plot(np.log(h1_vec), np.log(v_r2_atF), ':o', label='HHJ 2')
plt.plot(np.log(h1_vec), np.log(v_errInf_r2), '-.+', label='HHJ 2 $L^\infty$')
plt.plot(np.log(h1_vec), np.log(v_errQuad_r2), '--*', label='HHJ 2 $L^2$')
plt.plot(np.log(h1_vec), np.log(h1_vec**2), '-v', label=r'$h^2$')

# plt.plot(np.log(h2_vec), np.log(v_r3_atF), ':o', label='HHJ 3')
plt.plot(np.log(h2_vec), np.log(v_errInf_r3), '-.+', label='HHJ 3 $L^\infty$')
plt.plot(np.log(h2_vec), np.log(v_errQuad_r3), '--*', label='HHJ 3 $L^2$')
plt.plot(np.log(h2_vec), np.log(h2_vec**3), '-v', label=r'$h^3$')

plt.xlabel(r'log(Mesh size $h$)')
plt.ylabel(r'log(Error Velocity)')
plt.title(r'Velocity Error vs Mesh size')
plt.legend()
path_fig = "/home/a.brugnoli/Plots_Videos/Python/Plots/Kirchhoff_plots/Convergence/firedrake/"
if save_res:
    plt.savefig(path_fig  + bc_input + "_vel.eps", format="eps")

sig_r1int_atF = np.polyfit(np.log(h1_vec), np.log(sig_err_r1), 1)[0]
sig_r1int_max = np.polyfit(np.log(h1_vec), np.log(sig_errInf_r1), 1)[0]
sig_r1int_L2 = np.polyfit(np.log(h1_vec), np.log(sig_errQuad_r1), 1)[0]

print("Estimated order of convergence r=1 for sigma at T fin: " + str(sig_r1_atF))
print("Interpolated order of convergence r=1 for sigma at T fin: " + str(sig_r1int_atF))
print("Estimated order of convergence r=1 for sigma Linf: " + str(sig_r1_max))
print("Interpolated order of convergence r=1 for sigma Linf: " + str(sig_r1int_max))
print("Estimated order of convergence r=1 for sigma L2: " + str(sig_r1_L2))
print("Interpolated order of convergence r=1 for sigma L2: " + str(sig_r1int_L2))
print("")

sig_r2int = np.polyfit(np.log(h1_vec), np.log(sig_err_r2), 1)[0]
sig_r2int_max = np.polyfit(np.log(h1_vec), np.log(sig_errInf_r2), 1)[0]
sig_r2int_L2 = np.polyfit(np.log(h1_vec), np.log(sig_errQuad_r2), 1)[0]

print("Estimated order of convergence r=2 for sigma at T fin: " + str(sig_r2_atF))
print("Interpolated order of convergence r=2 for sigma at T fin: " + str(sig_r2int))
print("Estimated order of convergence r=2 for sigma Linf: " + str(sig_r2_max))
print("Interpolated order of convergence r=2 for sigma Linf: " + str(sig_r2int_max))
print("Estimated order of convergence r=2 for sigma L2: " + str(sig_r2_L2))
print("Interpolated order of convergence r=2 for sigma L2: " + str(sig_r2int_L2))
print("")

sig_r3int = np.polyfit(np.log(h2_vec), np.log(sig_err_r3), 1)[0]
sig_r3int_max = np.polyfit(np.log(h2_vec), np.log(sig_errInf_r3), 1)[0]
sig_r3int_L2 = np.polyfit(np.log(h2_vec), np.log(sig_errQuad_r3), 1)[0]

print("Estimated order of convergence r=3 for sigma at T fin: " + str(sig_r3_atF))
print("Interpolated order of convergence r=3 for sigma at T fin: " + str(sig_r3int))
print("Estimated order of convergence r=3 for sigma Linf: " + str(sig_r3_max))
print("Interpolated order of convergence r=3 for sigma Linf: " + str(sig_r3int_max))
print("Estimated order of convergence r=3 for sigma L2: " + str(sig_r3_L2))
print("Interpolated order of convergence r=3 for sigma L2: " + str(sig_r3int_L2))
print("")

plt.figure()

# plt.plot(np.log(h1_vec), np.log(sig_r1_atF), ':o', label='HHJ 1')
plt.plot(np.log(h1_vec), np.log(sig_errInf_r1), '-.+', label='HHJ 1 $L^\infty$')
plt.plot(np.log(h1_vec), np.log(sig_errQuad_r1), '--*', label='HHJ 1 $L^2$')
plt.plot(np.log(h1_vec), np.log(h1_vec), '-v', label=r'$h$')

# plt.plot(np.log(h1_vec), np.log(sig_r2_atF), ':o', label='HHJ 2')
plt.plot(np.log(h1_vec), np.log(sig_errInf_r2), '-.+', label='HHJ 2 $L^\infty$')
plt.plot(np.log(h1_vec), np.log(sig_errQuad_r2), '--*', label='HHJ 2 $L^2$')
plt.plot(np.log(h1_vec), np.log(h1_vec**2), '-v', label=r'$h^2$')

# plt.plot(np.log(h2_vec), np.log(sig_r3_atF), ':o', label='HHJ 3')
plt.plot(np.log(h2_vec), np.log(sig_errInf_r3), '-.+', label='HHJ 3 $L^\infty$')
plt.plot(np.log(h2_vec), np.log(sig_errQuad_r3), '--*', label='HHJ 3 $L^2$')
plt.plot(np.log(h2_vec), np.log(h2_vec**3), '-v', label=r'$h^3$')

plt.xlabel(r'log(Mesh size $h$)')
plt.ylabel(r'log(Error Stress)')
plt.title(r'Stress Error vs Mesh size')
plt.legend()
if save_res:
    plt.savefig(path_fig + bc_input + "_sigma.eps", format="eps")
plt.show()