# Convergence test for HHJ

from fenics import *
import numpy as np
np.set_printoptions(threshold=np.inf)
from math import pi, floor

import matplotlib
import matplotlib.pyplot as plt


matplotlib.rcParams['text.usetex'] = True

def compute_err(n, r):

    h_mesh = 1/n

    E = 136 * 10**9 # Pa
    rho = 5600  # kg/m^3
    nu = 0.3
    h = 0.001
    bc_input = 'SSSS'

    Lx = 1
    Ly = 1

    D = E * h ** 3 / (1 - nu ** 2) / 12
    fl_rot = 12 / (E * h ** 3)
    # Useful Matrices

    # Operators and functions
    def gradSym(u):
        return 0.5 * (nabla_grad(u) + nabla_grad(u).T)
        # return sym(nabla_grad(u))

    def bending_curv(momenta):
        kappa = fl_rot * ((1+nu)*momenta - nu * Identity(2) * tr(momenta))
        return kappa

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

    # bc_1, bc_2, bc_3, bc_4 = bc_input
    #
    # bc_dict = {1: bc_1, 2: bc_2, 3: bc_3, 4: bc_4}
    # loc_dict = {1: left, 2: lower, 3: right, 4: upper}
    #
    # bcs = []
    # for key, val in bc_dict.items():
    #
    #     if val == 'C':
    #         bcs.append(DirichletBC(V.sub(0), Constant(0.0), loc_dict[key]))
    #     elif val == 'S':
    #         bcs.append(DirichletBC(V.sub(0), Constant(0.0), loc_dict[key]))
    #         bcs.append(DirichletBC(V.sub(1), Constant(((0.0, 0.0), (0.0, 0.0))), loc_dict[key]))
    #     elif val == 'F':
    #         bcs.append(DirichletBC(V.sub(1), Constant(((0.0, 0.0), (0.0, 0.0))), loc_dict[key]))

    bcs = []
    bcs.append(DirichletBC(V.sub(0), Constant(0.0), boundary))
    bcs.append(DirichletBC(V.sub(1), Constant(((0.0, 0.0), (0.0, 0.0))), boundary))

    t = 0
    t_fin = 1        # total simulation time

    beta = 4*pi/t_fin
    w_exact = Expression("sin(pi*x[0]/a)*sin(pi*x[1]/b)*sin(beta*t)", degree=2, beta=beta, a=Lx, b=Ly, t=t)
    grad_wex = Expression(('pi/a*cos(pi*x[0]/a)*sin(pi*x[1]/b)*sin(beta*t)',
                           'pi/b*sin(pi*x[0]/a)*cos(pi*x[1]/b)*sin(beta*t)'), degree=2, beta=beta, a=Lx, b=Ly, t=t)

    v_exact = Expression("beta * sin(pi*x[0]/a)*sin(pi*x[1]/b)*cos(beta*t)", degree=2, beta=beta, a=Lx, b=Ly, t=t)
    grad_vex = Expression(('beta*pi/a*cos(pi*x[0]/a)*sin(pi*x[1]/b)*cos(beta*t)',
                           'beta*pi/b*sin(pi*x[0]/a)*cos(pi*x[1]/b)*cos(beta*t)'), degree=2, beta=beta, a=Lx, b=Ly, t=t)

    force = Expression("sin(pi*x[0]/a)*sin(pi*x[1]/b)*sin(beta*t)*"
                       "(D*pow( pow(pi/a, 2) + pow(pi/b, 2) , 2) - pow(beta,2)*rho*h)",
                       degree=2, beta=beta, a=Lx, b=Ly, t=t, D=D, rho=rho, h=h)

    b_form = v_p*force*dx

    J, M = PETScMatrix(), PETScMatrix()
    B = PETScVector

    J = assemble(j_form)
    M = assemble(m_form)

    # Apply boundary conditions to M, J
    [bc.apply(J) for bc in bcs]
    [bc.apply(M) for bc in bcs]

    dt = 0.01
    theta = 0.5

    e_n1 = Function(V)
    e_n = Function(V)
    w_n1 = Function(Vp)
    w_n = Function(Vp)

    e_n.assign(Expression(('beta*sin(pi*x[0]/a)*sin(pi*x[1]/b)*cos(beta*t)', '0', '0', '0', '0'),
                          degree=2, beta=beta, a=Lx, b=Ly, t=t))

    ep_n, eq_n = e_n.split(True)

    w_n.assign(Constant(0.0))

    n_t = int(floor(t_fin/dt) + 1)

    err_L2 = np.zeros((n_t,))
    w_atP = np.zeros((n_t,))
    v_atP = np.zeros((n_t,))
    Ppoint = (Lx/3, Ly/3)
    v_atP[0] = ep_n(Ppoint[0], Ppoint[1])

    w_exact.t = t
    # err_L2[0] = np.sqrt(assemble(dot(w_n-w_exact, w_n-w_exact) *dx
    #                      + dot(grad(w_n) - grad_wex, grad(w_n) - grad_wex) * dx))
    err_L2[0] = np.sqrt(assemble(dot(ep_n - v_exact, ep_n - v_exact) * dx
                         + dot(grad(ep_n) - grad_vex, grad(ep_n) - grad_vex) * dx))

    t_vec = np.linspace(0, t_fin, num=n_t)

    for i in range(1, n_t):

        force.t = t
        f_n = assemble(b_form)
        [bc.apply(f_n) for bc in bcs]

        t += dt

        force.t = t
        f_n1 = assemble(b_form)
        [bc.apply(f_n1) for bc in bcs]

        b = (M + dt*(1-theta)*J)*e_n.vector() + dt*(theta*f_n1 + (1-theta)*f_n)

        A = M - dt*theta*J

        solve(A, e_n1.vector(), b)

        ep_n, eq_n = e_n.split(True)
        ep_n1, eq_n1 = e_n1.split(True)

        w_n1.vector()[:] = w_n.vector() + dt/2*(ep_n.vector() + ep_n1.vector())

        e_n.assign(e_n1)
        w_n.assign(w_n1)

        w_atP[i] = w_n1(Ppoint[0], Ppoint[1])

        v_atP[i] = ep_n1(Ppoint[0], Ppoint[1])

        w_exact.t = t
        # err_L2[i] = np.sqrt(assemble(dot(w_n1-w_exact, w_n1-w_exact) * dx
        #                      + dot(grad(w_n1)-grad_wex, grad(w_n1)-grad_wex) * dx))
        err_L2[i] = np.sqrt(assemble(dot(ep_n1 - v_exact, ep_n1 - v_exact) * dx
                         + dot(grad(ep_n1) - grad_vex, grad(ep_n1) - grad_vex) * dx))

    # plt.figure()
    # # plt.plot(t_vec, w_atP, 'r-', label=r'approx $w$')
    # # plt.plot(t_vec, np.sin(pi*Ppoint[0]/Lx)*np.sin(pi*Ppoint[1]/Ly)*np.sin(beta*t_vec), 'b-', label=r'exact $w$')
    # plt.plot(t_vec, v_atP, 'r-', label=r'approx $v$')
    # plt.plot(t_vec, beta * np.sin(pi*Ppoint[0]/Lx)*np.sin(pi*Ppoint[1]/Ly) * np.cos(beta * t_vec), 'b-', label=r'exact $v$')
    # plt.xlabel(r'Time [s]')
    # plt.title(r'Displacement at $(L_x/2, L_y/2)$')
    # plt.legend()
    # plt.show()

    err_quad = np.sqrt(np.sum(dt*np.power(err_L2, 2)))
    err_max = max(err_L2)
    err_last = err_L2[-1]
    return err_last


n_h = 5
n_vec = np.array([2**(i+1) for i in range(n_h)])
h_vec = 1./n_vec
err_vec_r1 = np.zeros((n_h,))
err_vec_r2 = np.zeros((n_h,))

r1_ext = np.zeros((n_h-1,))
r2_ext = np.zeros((n_h-1,))

for i in range(n_h):
    err_vec_r1[i] = compute_err(n_vec[i], 1)
    err_vec_r2[i] = compute_err(n_vec[i], 2)

    if i>0:
        r1_ext[i-1] = np.log(err_vec_r1[i]/err_vec_r1[i-1])/np.log(h_vec[i]/h_vec[i-1])
        r2_ext[i-1] = np.log(err_vec_r2[i]/err_vec_r2[i-1])/np.log(h_vec[i]/h_vec[i-1])

order_r1 = np.polyfit(np.log(h_vec), np.log(err_vec_r1), 1)[0]
order_r2 = np.polyfit(np.log(h_vec), np.log(err_vec_r2), 1)[0]
print("Estimated order of convergence: " + str(r1_ext))
print("Estimated order of convergence: " + str(r2_ext))
print("Interpolated order of convergence: " + str(order_r1))
print("Interpolated order of convergence: " + str(order_r2))
plt.figure()
plt.plot(np.log(h_vec), np.log(err_vec_r1), '-m', label='HHJ 1')
plt.plot(np.log(h_vec), np.log(err_vec_r2), '-r', label='HHJ 2')
plt.plot(np.log(h_vec), np.log(h_vec), '-g', label=r'$h$')
plt.plot(np.log(h_vec), np.log(h_vec**2), '-b', label=r'$h^2$')
plt.xlabel(r'Mesh size')
plt.title(r'Error at $T_f$')
plt.legend()
plt.show()
