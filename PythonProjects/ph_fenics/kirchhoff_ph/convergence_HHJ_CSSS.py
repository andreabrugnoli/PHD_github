# Convergence test for HHJ

from fenics import *
import numpy as np
np.set_printoptions(threshold=np.inf)
from math import pi, floor

import matplotlib
import matplotlib.pyplot as plt
import mshr

matplotlib.rcParams['text.usetex'] = True

bc_input = 'CSSS'


def compute_err(n, r):

    h_mesh = 1/n

    E = 136 * 10**9 # Pa
    rho = 5600  # kg/m^3
    nu = 0.3
    h = 0.001

    D = E * h ** 3 / (1 - nu ** 2) / 12
    fl_rot = 12 / (E * h ** 3)

    Lx = 1
    Ly = 1

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

    mesh = RectangleMesh(Point(0, 0), Point(Lx, Ly), n, n, "right/left")

    # Domain, Subdomains, Boundary, Suboundaries
    def boundary(x, on_boundary):
        return on_boundary

    class Left(SubDomain):
        def inside(self, x, on_boundary):
            return abs(x[0] - 0.0) < DOLFIN_EPS and on_boundary

    class Right(SubDomain):
        def inside(self, x, on_boundary):
            return abs(x[0] - Lx) < DOLFIN_EPS and on_boundary

    class Lower(SubDomain):
        def inside(self, x, on_boundary):
            return abs(x[1] - 0.0) < DOLFIN_EPS and on_boundary

    class Upper(SubDomain):
        def inside(self, x, on_boundary):
            return abs(x[1] - Ly) < DOLFIN_EPS and on_boundary

    # Boundary conditions on rotations
    left = Left()
    right = Right()
    lower = Lower()
    upper = Upper()

    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    left.mark(boundaries, 1)
    lower.mark(boundaries, 2)
    right.mark(boundaries, 3)
    upper.mark(boundaries, 4)

    # Finite element defition

    CG = FiniteElement('CG', mesh.ufl_cell(), r)
    HHJ = FiniteElement('HHJ', mesh.ufl_cell(), r-1)

    Vp = FunctionSpace(mesh, CG)
    V = FunctionSpace(mesh, CG * HHJ)
    n_V = V.dim()
    print(n_V)

    v = TestFunction(V)
    v_p, v_q = split(v)

    e = TrialFunction(V)
    e_p, e_q = split(e)

    al_p = rho * h * e_p
    al_q = bending_curv(e_q)

    dx = Measure('dx')
    ds = Measure('ds', subdomain_data=boundaries)
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

    degr = 4
    t = 0
    t_fin = 1        # total simulation time

    beta = 1 # 4*pi/t_fin
    w_0 = 1

    fx = 'pow(x[0], 2)*pow(Lx/2-x[0], 2)*'
    gy = 'pow(Ly/2-x[1], 4)*'
    ht = 'sin(t)'

    dx_fx = '(pow(Lx, 2)/2*x[0] - 3*Lx*pow(x[0], 2) + 4*pow(x[0], 3))*'
    dxx_fx = '(pow(Lx, 2)/2 - 6*Lx*x[0] + 12*pow(x[0], 2))*'
    dxxx_fx = '(-6*Lx + 24*x[0])*'
    dxxxx_fx = '24*'

    dy_gy = '(-4*pow(Ly/2-x[1], 3))*'
    dyy_gy = '(12*pow(Ly/2-x[1], 2))*'
    dyyy_gy = '(-24*(Ly/2-x[1]))*'
    dyyyy_gy = '24*'

    dt_ht = 'cos(t)'
    dtt_ht = '(-sin(t))'

    str_w = fx + gy + ht
    str_wx = dx_fx + gy + ht
    str_wy = fx + dy_gy + ht

    str_v = fx + gy + dt_ht
    str_vx = dx_fx + gy + dt_ht
    str_vy = fx + dy_gy + dt_ht

    str_wxx = dxx_fx + gy + ht
    str_wyy = fx + dyy_gy + ht
    str_wxy = dx_fx + dy_gy + ht

    str_wtt = fx + gy + dtt_ht

    str_wxxxx = dxxxx_fx + gy + ht
    str_wyyyy = fx + dyyyy_gy + ht
    str_wxxyy = dxx_fx + dyy_gy + ht

    str_sxx = 'D*' + str_wxx + '+ D*nu*' + str_wyy
    str_syy = 'D*' + str_wyy + '+ D*nu*' + str_wxx
    str_sxy = 'D*(1-nu)*' + str_wxy

    str_f = 'rho*h*' + str_wtt + ' + D*' + str_wxxxx + ' + 2*D*' + str_wxxyy + '+ D*' + str_wyyyy

    w_exact = Expression(str_w, \
                         degree=degr, w_0=w_0, Lx=Lx, Ly=Ly, beta=beta, t=t)

    v_exact = Expression(str_v, \
                         degree=degr, w_0=w_0, Lx=Lx, Ly=Ly, beta=beta, t=t)

    grad_wex = Expression((str_wx, str_wy), \
                         degree=degr, w_0=w_0, Lx=Lx, Ly=Ly, beta=beta, t=t)

    grad_vex = Expression((str_vx, str_vy), \
                          degree=degr, w_0=w_0, Lx=Lx, Ly=Ly, beta=beta, t=t)

    sig_exact = Expression(((str_sxx, str_sxy),
                           (str_sxy, str_syy)),
                          degree=degr, w_0=w_0, Lx=Lx, Ly=Ly, beta=beta, D=D, nu=nu, t=t)


    force = Expression(str_f, \
                          degree=degr, w_0=w_0, Lx=Lx, Ly=Ly, beta=beta, D=D, rho=rho, h=h, t=t)
    force1 = Expression(str_f, \
                          degree=degr, w_0=w_0, Lx=Lx, Ly=Ly, beta=beta, D=D, rho=rho, h=h, t=t)

    bc_1, bc_2, bc_3, bc_4 = bc_input

    bc_dict = {1: bc_1, 2: bc_2, 3: bc_3, 4: bc_4}
    loc_dict = {1: left, 2: lower, 3: right, 4: upper}

    # bcs = []
    # for key, val in bc_dict.items():
    #
    #     if val == 'C':
    #         bcs.append(DirichletBC(V.sub(0), Constant(0.0), loc_dict[key]))
    #     elif val == 'S':
    #         bcs.append(DirichletBC(V.sub(0), v_exact, loc_dict[key]))
    #         bcs.append(DirichletBC(V.sub(1), sig_exact, loc_dict[key]))
    #     elif val == 'F':
    #         bcs.append(DirichletBC(V.sub(1), sig_exact, loc_dict[key]))

    bcs = []
    bcs.append(DirichletBC(V.sub(0), v_exact, boundary))
    bcs.append(DirichletBC(V.sub(1), sig_exact, boundary))

    f_form = v_p*force*dx
    f_form1 = v_p*force1*dx

    # J, M = PETScMatrix(), PETScMatrix()

    J = assemble(j_form)
    M = assemble(m_form)

    dt = 0.01*h_mesh
    theta = 0.5

    A = M - dt * theta * J

    e_n1 = Function(V)
    e_n = Function(V)
    w_n1 = Function(Vp)
    w_n = Function(Vp)

    e_n.assign(Expression((str_v, '0', '0', '0', '0'),
                          degree=degr, w_0=w_0, Lx=Lx, Ly=Ly, beta=beta, t=t))

    ep_n, eq_n = e_n.split(True)

    w_n.assign(Constant(0.0))

    n_t = int(floor(t_fin/dt) + 1)

    v_err_H1 = np.zeros((n_t,))
    sig_err_L2 = np.zeros((n_t,))

    w_atP = np.zeros((n_t,))
    v_atP = np.zeros((n_t,))
    Ppoint = (Lx/3, Ly/3)
    v_atP[0] = ep_n(Ppoint[0], Ppoint[1])

    w_exact.t = t
    grad_wex.t = t

    v_exact.t = t
    grad_vex.t = t

    sig_exact.t = t

    # err_H1[0] = np.sqrt(assemble(dot(w_n-w_exact, w_n-w_exact) *dx
    #                      + dot(grad(w_n) - grad_wex, grad(w_n) - grad_wex) * dx))
    v_err_H1[0] = np.sqrt(assemble(dot(ep_n - v_exact, ep_n - v_exact) * dx
                       + dot(grad(ep_n) - grad_vex, grad(ep_n) - grad_vex) * dx))

    sig_err_L2[0] = np.sqrt(assemble(inner(eq_n - sig_exact, eq_n - sig_exact) * dx))

    t_vec = np.linspace(0, t_fin, num=n_t)

    for i in range(1, n_t):

        # dtt_wex.t = t
        # dxxxx_wex.t = t
        # dyyyy_wex.t = t
        # dxxyy_wex.t = t
        # f_n = assemble(f_form)
        # [bc.apply(f_n) for bc in bcs]
        #
        # dtt_wex1.t = t + dt
        # dxxxx_wex1.t = t + dt
        # dyyyy_wex1.t = t + dt
        # dxxyy_wex1.t = t + dt

        # f_n1 = assemble(f_form1)
        # [bc.apply(f_n1) for bc in bcs]
        #
        # b = (M + dt*(1-theta)*J)*e_n.vector() + dt*(theta*f_n1 + (1-theta)*f_n)

        force.t = t
        force1.t = t + dt

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

        b = assemble(rhs)

        v_exact.t = t
        sig_exact.t = t

        [bc.apply(A, b) for bc in bcs]
        # [bc.apply(A) for bc in bcs]

        solve(A, e_n1.vector(), b)

        t += dt

        ep_n, eq_n = e_n.split(True)
        ep_n1, eq_n1 = e_n1.split(True)

        w_n1.vector()[:] = w_n.vector() + dt/2*(ep_n.vector() + ep_n1.vector())
        w_n.assign(w_n1)

        e_n.assign(e_n1)

        w_atP[i] = w_n1(Ppoint[0], Ppoint[1])
        v_atP[i] = ep_n1(Ppoint[0], Ppoint[1])

        w_exact.t = t
        grad_wex.t = t

        v_exact.t = t
        grad_vex.t = t

        # dxx_wex.t = t
        # dyy_wex.t = t
        # dxy_wex.t = t

        sig_exact.t = t


        # v_err_H1[i] = np.sqrt(assemble(dot(w_n1-w_exact, w_n1-w_exact) * dx
        #                      + dot(grad(w_n1)-grad_wex, grad(w_n1)-grad_wex) * dx))
        v_err_H1[i] = np.sqrt(assemble(dot(ep_n1 - v_exact, ep_n1 - v_exact) * dx
                         + dot(grad(ep_n1) - grad_vex, grad(ep_n1) - grad_vex) * dx))

        sig_err_L2[i] = np.sqrt(assemble(inner(eq_n1 - sig_exact, eq_n1 - sig_exact) * dx))

    plt.figure()
    # plt.plot(t_vec, w_atP, 'r-', label=r'approx $w$')
    # plt.plot(t_vec, w_0/(Lx*Ly)**4*(Ppoint[0]*(Lx/2-Ppoint[0]))**2*(Ly/2-Ppoint[1])**4*np.sin(beta*t), 'b-', label=r'exact $v$')
    plt.plot(t_vec, v_atP, 'r-', label=r'approx $v$')
    plt.plot(t_vec, (Ppoint[0]*(Lx/2-Ppoint[0]))**2*(Ly/2-Ppoint[1])**4*np.cos(t_vec), 'b-', label=r'exact $v$')
    plt.xlabel(r'Time [s]')
    plt.title(r'Displacement at ' + str(Ppoint))
    plt.legend()
    plt.show()

    v_err_last = v_err_H1[-1]
    v_err_max = max(v_err_H1)
    v_err_quad = np.sqrt(np.sum(dt * np.power(v_err_H1, 2)))

    sig_err_last = sig_err_L2[-1]
    sig_err_max = max(sig_err_L2)
    sig_err_quad = np.sqrt(np.sum(dt * np.power(sig_err_L2, 2)))

    return v_err_last, v_err_max, v_err_quad, sig_err_last, sig_err_max, sig_err_quad


n_h = 2
n1_vec = np.array([2**(i+2) for i in range(n_h)])
n2_vec = np.array([2**(i) for i in range(n_h)])
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

# np.save("./convergence_results/" + bc_input + "_h1", h1_vec)
# np.save("./convergence_results/" + bc_input + "_h2", h1_vec)
# np.save("./convergence_results/" + bc_input + "_h3", h2_vec)
#
# np.save("./convergence_results/" + bc_input + "_v_errF_r1", v_err_r1)
# np.save("./convergence_results/" + bc_input + "_v_errInf_r1", v_errInf_r1)
# np.save("./convergence_results/" + bc_input + "_v_errQuad_r1", v_errQuad_r1)
#
# np.save("./convergence_results/" + bc_input + "_v_errF_r2", v_err_r2)
# np.save("./convergence_results/" + bc_input + "_v_errInf_r2", v_errInf_r2)
# np.save("./convergence_results/" + bc_input + "_v_errQuad_r2", v_errQuad_r2)
#
# np.save("./convergence_results/" + bc_input + "_v_errF_r3", v_err_r3)
# np.save("./convergence_results/" + bc_input + "_v_errInf_r3", v_errInf_r3)
# np.save("./convergence_results/" + bc_input + "_v_errQuad_r3", v_errQuad_r3)
#
# np.save("./convergence_results/" + bc_input + "_sig_errF_r1", sig_err_r1)
# np.save("./convergence_results/" + bc_input + "_sig_errInf_r1", sig_errInf_r1)
# np.save("./convergence_results/" + bc_input + "_sig_errQuad_r1", sig_errQuad_r1)
#
# np.save("./convergence_results/" + bc_input + "_sig_errF_r2", sig_err_r2)
# np.save("./convergence_results/" + bc_input + "_sig_errInf_r2", sig_errInf_r2)
# np.save("./convergence_results/" + bc_input + "_sig_errQuad_r2", sig_errQuad_r2)
#
# np.save("./convergence_results/" + bc_input + "_sig_errF_r3", sig_err_r3)
# np.save("./convergence_results/" + bc_input + "_sig_errInf_r3", sig_errInf_r3)
# np.save("./convergence_results/" + bc_input + "_sig_errQuad_r3", sig_errQuad_r3)

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
path_fig = "/home/a.brugnoli/Plots_Videos/Python/Plots/Kirchhoff_plots/Convergence/"
# plt.savefig(path_fig  + bc_input + "_vel.eps", format="eps")

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
# plt.savefig(path_fig + bc_input + "_sigma.eps", format="eps")
plt.show()



# w_exact = Expression('w_0/(pow(Lx*Ly, 4))*pow(x[0]*(Lx/2-x[0]), 2)*pow(Ly/2-x[1], 4)*sin(beta*t)',\
#                          degree=degr, w_0=w_0, Lx=Lx, Ly=Ly, beta=beta, t=t)
#     grad_wex = Expression(('w_0/(pow(Lx*Ly, 4))*(pow(Lx,2)/2*x[0] - 3*Lx*pow(x[0], 2)'
#                            ' + 4*pow(x[0],3))*pow(Ly/2-x[1], 4)*sin(beta*t)',
#                            '-4*w_0/(pow(Lx*Ly, 4))*pow(x[0]*(Lx/2-x[0]), 2)*pow(Ly/2-x[1], 3)*sin(beta*t)'), \
#                           degree=degr, w_0=w_0, Lx=Lx, Ly=Ly, beta=beta, t=t)
#
#     v_exact = Expression("w_0*beta/(pow(Lx*Ly, 4))*pow(x[0]*(Lx/2-x[0]), 2)*pow(Ly/2-x[1], 4)*cos(beta*t)", \
#                          degree=degr, w_0=w_0, Lx=Lx, Ly=Ly, beta=beta, t=t)
#
#     grad_vex = Expression(('beta*w_0/(pow(Lx*Ly, 4))*(pow(Lx,2)/2*x[0] - 3*Lx*pow(x[0], 2)'
#                            ' + 4*pow(x[0],3))*pow(Ly/2-x[1], 4)*cos(beta*t)',
#                            '-4*beta*w_0/(pow(Lx*Ly, 4))*pow(x[0]*(Lx/2-x[0]), 2)*pow(Ly/2-x[1], 3)*cos(beta*t)'), \
#                           degree=degr, w_0=w_0, Lx=Lx, Ly=Ly, beta=beta, t=t)
#
#     dxx_wex = Expression('w_0/(pow(Lx*Ly, 4))*(pow(Lx,2)/2 - 6*Lx*x[0]'
#                            ' + 12*pow(x[0],2))*pow(Ly/2-x[1], 4)*sin(beta*t)',
#                          degree=degr, w_0=w_0, Lx=Lx, Ly=Ly, beta=beta, t=t)
#     dyy_wex = Expression('12*w_0/(pow(Lx*Ly, 4))*pow(x[0]*(Lx/2-x[0]), 2)*pow(Ly/2-x[1], 2)*sin(beta*t)', \
#                          degree=degr, w_0=w_0, Lx=Lx, Ly=Ly, beta=beta, t=t)
#
#     dxy_wex = Expression('-4*w_0/(pow(Lx*Ly, 4))*(pow(Lx,2)/2*x[0] - 3*Lx*pow(x[0], 2)'
#                            ' + 4*pow(x[0],3))*pow(Ly/2-x[1], 3)*sin(beta*t)', \
#                          degree=degr, w_0=w_0, Lx=Lx, Ly=Ly, beta=beta, t=t)
#
#     sigma_ex = as_tensor([[D*(dxx_wex + nu*dyy_wex), D*(1-nu)*dxy_wex],
#                            [D*(1-nu)*dxy_wex, D*(dyy_wex + nu*dxx_wex)]])
#
#     dtt_wex = Expression('-pow(beta, 2)*w_0/(pow(Lx*Ly, 4))*pow(x[0]*(Lx/2-x[0]), 2)*pow(Ly/2-x[1], 4)*sin(beta*t)',\
#                          degree=degr, w_0=w_0, Lx=Lx, Ly=Ly, beta=beta, t=t)
#
#     dxxxx_wex = Expression('24*w_0/(pow(Lx*Ly, 4))*pow(Ly/2-x[1], 4)*sin(beta*t)',\
#                          degree=degr, w_0=w_0, Lx=Lx, Ly=Ly, beta=beta, t=t)
#
#     dyyyy_wex = Expression('24*w_0/(pow(Lx*Ly, 4))*pow(x[0]*(Lx/2-x[0]), 2)*sin(beta*t)', \
#                           degree=degr, w_0=w_0, Lx=Lx, Ly=Ly, beta=beta, t=t)
#
#     dxxyy_wex = Expression('12*w_0/(pow(Lx*Ly, 4))*(pow(Lx,2)/2 - 6*Lx*x[0]'
#                            ' + 12*pow(x[0],2))*pow(Ly/2-x[1], 2)*sin(beta*t)', \
#                            degree=degr, w_0=w_0, Lx=Lx, Ly=Ly, beta=beta, t=t)
#
#     dtt_wex1 = Expression('-pow(beta, 2)*w_0/(pow(Lx*Ly, 4))*pow(x[0]*(Lx/2-x[0]), 2)*pow(Ly/2-x[1], 4)*sin(beta*t)', \
#                          degree=degr, w_0=w_0, Lx=Lx, Ly=Ly, beta=beta, t=t)
#
#     dxxxx_wex1 = Expression('24*w_0/(pow(Lx*Ly, 4))*pow(Ly/2-x[1], 4)*sin(beta*t)', \
#                            degree=degr, w_0=w_0, Lx=Lx, Ly=Ly, beta=beta, t=t)
#
#     dyyyy_wex1 = Expression('24*w_0/(pow(Lx*Ly, 4))*pow(x[0]*(Lx/2-x[0]), 2)*sin(beta*t)', \
#                            degree=degr, w_0=w_0, Lx=Lx, Ly=Ly, beta=beta, t=t)
#
#     dxxyy_wex1 = Expression('12*w_0/(pow(Lx*Ly, 4))*(pow(Lx,2)/2 - 6*Lx*x[0]'
#                            ' + 12*pow(x[0],2))*pow(Ly/2-x[1], 2)*sin(beta*t)', \
#                            degree=degr, w_0=w_0, Lx=Lx, Ly=Ly, beta=beta, t=t)
#
#     str_sxx = 'D*sin(beta*t)*w_0/pow(Lx*Ly, 4)*( (pow(Lx,2)/2\
#      - 6*Lx*x[0]+ 12*pow(x[0],2))*pow(Ly/2-x[1], 4)+ nu*12*pow(x[0]*(Lx/2-x[0]), 2)*pow(Ly/2-x[1], 2))'
#
#     str_syy = 'D*sin(beta*t)*w_0/(pow(Lx*Ly, 4))*(12*pow(x[0]*(Lx/2-x[0]), 2)*pow(Ly/2-x[1], 2)\
#                                    +nu*(pow(Lx,2)/2 - 6*Lx*x[0]+ 12*pow(x[0],2))*pow(Ly/2-x[1], 4) )'
#
#     str_sxy = '-4*D*(1-nu)*w_0/(pow(Lx*Ly, 4))*(pow(Lx,2)/2*x[0] - 3*Lx*pow(x[0], 2)\
#                                 + 4*pow(x[0],3))*pow(Ly/2-x[1], 3)*sin(beta*t)'
#
#     Exsigma_ex = Expression(((str_sxx, str_sxy),
#                              (str_sxy, str_syy)),
#                             degree=degr, w_0=w_0, Lx=Lx, Ly=Ly, beta=beta, D=D, nu=nu, t=t)