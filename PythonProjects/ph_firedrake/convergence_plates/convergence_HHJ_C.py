# Convergence test for HHJ

from firedrake import *
import numpy as np
np.set_printoptions(threshold=np.inf)
from math import pi, floor

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['text.usetex'] = True

bc_input = 'C'
save_res = False
path_mesh = "/home/a.brugnoli/GitProjects/PythonProjects/ph_firedrake/thin_plate/circle_meshes/"

def compute_err(n, r):

    h_mesh = 1/n

    E = 136 * 10**9 # Pa
    rho = 5600  # kg/m^3
    nu = 0.3
    h = 0.001

    D = E * h ** 3 / (1 - nu ** 2) / 12
    fl_rot = 12 / (E * h ** 3)
    R = 1
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

    mesh = Mesh(path_mesh + "circle_n" + str(n) + ".msh")

    # plot(mesh);
    # plt.show()


    # Finite element defition

    Vp = FunctionSpace(mesh, 'CG', r)
    Vq = FunctionSpace(mesh, 'HHJ', r - 1)
    V = Vp * Vq
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
    bcs.append(DirichletBC(V.sub(0), Constant(0.0), "on_boundary"))

    degr = 4
    t = 0
    t_fin = 1        # total simulation time

    t_ = Constant(t)
    t_1 = Constant(t)

    beta = 1 # 4*pi/t_fin
    f_0 = 1
    R = 1
    x = mesh.coordinates

    w_exact = f_0/(64*D)*(R**2-(x[0]**2+x[1]**2))**2*sin(beta*t_)
    grad_wex = as_vector([-f_0/(16*D)*(R**2-(x[0]**2+x[1]**2))*sin(beta*t_)*x[0],
                          -f_0/(16*D)*(R**2-(x[0]**2+x[1]**2))*sin(beta*t_)*x[1]])

    v_exact = f_0*beta/(64*D)*(R**2-(x[0]**2+x[1]**2))**2*cos(beta*t_)
    grad_vex = as_vector([-f_0*beta/(16*D)*(R**2-(x[0]**2+x[1]**2))*cos(beta*t_)*x[0],
                          -f_0*beta/(16*D)*(R**2-(x[0]**2+x[1]**2))*cos(beta*t_)*x[1]])

    dxx_wex = -f_0/(16*D)*(R**2-(3*x[0]**2+x[1]**2) )*sin(beta*t_)
    dyy_wex = -f_0/(16*D)*(R**2-(x[0]**2+3*x[1]**2) )*sin(beta*t_)
    dxy_wex = f_0/(8*D)*x[0]*x[1]*sin(beta*t_)
    sigma_ex = as_tensor([[D*(dxx_wex + nu*dyy_wex), D*(1-nu)*dxy_wex],
                         [D*(1-nu)*dxy_wex, D*(dyy_wex + nu*dxx_wex)]])

    force = f_0*sin(beta*t_)*(1 - rho*h*beta**2/(64*D)*(R**2-(x[0]**2+x[1]**2))**2)

    force1 = f_0*sin(beta*t_1)*(1 - rho*h*beta**2/(64*D)*(R**2-(x[0]**2+x[1]**2))**2)

    f_form = v_p*force*dx
    f_form1 = v_p*force1*dx

    # J, M = PETScMatrix(), PETScMatrix()

    dt = 0.1*h_mesh
    theta = 0.5

    lhs = m_form - dt*theta*j_form
    A = assemble(lhs, bcs=bcs, mat_type='aij')

    e_n1 = Function(V)
    e_n = Function(V)
    w_n1 = Function(Vp)
    w_n = Function(Vp)

    e_n.sub(0).assign(interpolate(v_exact, Vp))

    ep_n, eq_n = e_n.split()

    w_n.assign(Constant(0.0))

    n_t = int(floor(t_fin/dt) + 1)

    v_err_H1 = np.zeros((n_t,))
    sig_err_L2 = np.zeros((n_t,))

    w_atP = np.zeros((n_t,))
    v_atP = np.zeros((n_t,))
    Ppoint = (R/3, R/7)
    v_atP[0] = ep_n.at(Ppoint[0], Ppoint[1])

    # w_err_H1[0] = np.sqrt(assemble(dot(w_n-w_exact, w_n-w_exact) *dx
    #                      + dot(grad(w_n) - grad_wex, grad(w_n) - grad_wex) * dx))
    v_err_H1[0] = np.sqrt(assemble(dot(ep_n - v_exact, ep_n - v_exact) * dx
                         + dot(grad(ep_n) - grad_vex, grad(ep_n) - grad_vex) * dx))

    sig_err_L2[0] = np.sqrt(assemble(inner(eq_n - sigma_ex, eq_n - sigma_ex) * dx))

    t_vec = np.linspace(0, t_fin, num=n_t)

    param = {"ksp_type": "preonly", "pc_type": "lu"}

    for i in range(1, n_t):

        t_.assign(t)
        t_1.assign(t + dt)

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

        solve(A, e_n1, b, solver_parameters=param)

        t += dt

        ep_n, eq_n = e_n.split()
        ep_n1, eq_n1 = e_n1.split()

        w_n1.assign(w_n + dt / 2 * (ep_n + ep_n1))
        w_n.assign(w_n1)

        e_n.assign(e_n1)

        w_atP[i] = w_n1.at(Ppoint[0], Ppoint[1])
        v_atP[i] = ep_n1.at(Ppoint[0], Ppoint[1])

        t_.assign(t)


        # w_err_H1[i] = np.sqrt(assemble(dot(w_n1-w_exact, w_n1-w_exact) * dx
        #                      + dot(grad(w_n1)-grad_wex, grad(w_n1)-grad_wex) * dx))
        v_err_H1[i] = np.sqrt(assemble(dot(ep_n1 - v_exact, ep_n1 - v_exact) * dx
                         + dot(grad(ep_n1) - grad_vex, grad(ep_n1) - grad_vex) * dx))

        sig_err_L2[i] = np.sqrt(assemble(inner(eq_n1 - sigma_ex, eq_n1 - sigma_ex) * dx))

    plt.figure()
    # plt.plot(t_vec, w_atP, 'r-', label=r'approx $w$')
    # plt.plot(t_vec, f_0/(64*D)*(R**2-(Ppoint[0]**2+Ppoint[1]**2))**2*np.sin(beta*t_vec), 'b-', label=r'exact $w$')
    plt.plot(t_vec, v_atP, 'r-', label=r'approx $v$')
    plt.plot(t_vec, f_0*beta/(64*D)*(R**2-(Ppoint[0]**2+Ppoint[1]**2))**2*np.cos(beta*t_vec), 'b-', label=r'exact $v$')
    plt.xlabel(r'Time [s]')
    plt.title(r'Displacement at ' +  str(Ppoint))
    plt.legend()

    v_err_last = v_err_H1[-1]
    v_err_max = max(v_err_H1)
    v_err_quad = np.sqrt(np.sum(dt * np.power(v_err_H1, 2)))

    sig_err_last = sig_err_L2[-1]
    sig_err_max = max(sig_err_L2)
    sig_err_quad = np.sqrt(np.sum(dt * np.power(sig_err_L2, 2)))

    return v_err_last, v_err_max, v_err_quad, sig_err_last, sig_err_max, sig_err_quad


n_h = 3
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

if save_res:
    np.save("./convergence_results/" + bc_input + "_h1", h1_vec)
    np.save("./convergence_results/" + bc_input + "_h2", h1_vec)
    np.save("./convergence_results/" + bc_input + "_h3", h2_vec)

    np.save("./convergence_results/" + bc_input + "_v_errF_r1", v_err_r1)
    np.save("./convergence_results/" + bc_input + "_v_errInf_r1", v_errInf_r1)
    np.save("./convergence_results/" + bc_input + "_v_errQuad_r1", v_errQuad_r1)

    np.save("./convergence_results/" + bc_input + "_v_errF_r2", v_err_r2)
    np.save("./convergence_results/" + bc_input + "_v_errInf_r2", v_errInf_r2)
    np.save("./convergence_results/" + bc_input + "_v_errQuad_r2", v_errQuad_r2)

    np.save("./convergence_results/" + bc_input + "_v_errF_r3", v_err_r3)
    np.save("./convergence_results/" + bc_input + "_v_errInf_r3", v_errInf_r3)
    np.save("./convergence_results/" + bc_input + "_v_errQuad_r3", v_errQuad_r3)

    np.save("./convergence_results/" + bc_input + "_sig_errF_r1", sig_err_r1)
    np.save("./convergence_results/" + bc_input + "_sig_errInf_r1", sig_errInf_r1)
    np.save("./convergence_results/" + bc_input + "_sig_errQuad_r1", sig_errQuad_r1)

    np.save("./convergence_results/" + bc_input + "_sig_errF_r2", sig_err_r2)
    np.save("./convergence_results/" + bc_input + "_sig_errInf_r2", sig_errInf_r2)
    np.save("./convergence_results/" + bc_input + "_sig_errQuad_r2", sig_errQuad_r2)

    np.save("./convergence_results/" + bc_input + "_sig_errF_r3", sig_err_r3)
    np.save("./convergence_results/" + bc_input + "_sig_errInf_r3", sig_errInf_r3)
    np.save("./convergence_results/" + bc_input + "_sig_errQuad_r3", sig_errQuad_r3)

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