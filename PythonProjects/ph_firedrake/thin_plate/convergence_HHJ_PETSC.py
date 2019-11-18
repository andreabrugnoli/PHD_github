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

    # The unit square mesh is divided in :math:`N\times N` quadrilaterals::

    mesh = RectangleMesh(n, n, Lx, Lx, quadrilateral=False)

    # Domain, Subdomains, Boundary, Suboundaries

    # Finite element defition

    Vp = FunctionSpace(mesh, 'CG', r + 1)
    Vq = FunctionSpace(mesh, 'HHJ', r)
    V = Vp * Vq

    Vgradp = VectorFunctionSpace(mesh, 'CG', r + 1)


    n_Vp = V.sub(0).dim()
    n_Vq = V.sub(1).dim()
    n_V = V.dim()
    print(n_V, n_Vp)

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

    bc_1, bc_2, bc_3, bc_4 = bc_input
    bc_dict = {1: bc_1, 3: bc_2, 2: bc_3, 4: bc_4}

    # The boundary edges in this mesh are numbered as follows:

    # 1: plane x == 0
    # 2: plane x == 1
    # 3: plane y == 0
    # 4: plane y == 1

    bcs = []
    # bcs_q = []
    boundary_dofs = []

    for key, val in bc_dict.items():

        if val == 'C':
            bc_p = DirichletBC(V.sub(0), Constant(0.0), key)
            for node in bc_p.nodes:
                boundary_dofs.append(node)
            bcs.append(bc_p)

        elif val == 'S':
            bc_p = DirichletBC(V.sub(0), Constant(0.0), key)
            bc_q = DirichletBC(V.sub(1), Constant(((0.0, 0.0), (0.0, 0.0))), key)

            for node in bc_p.nodes:
                boundary_dofs.append(node)
            bcs.append(bc_p)

            for node in bc_q.nodes:
                boundary_dofs.append(n_Vp + node)
            bcs.append(bc_q)

        elif val == 'F':
            bc_q = DirichletBC(V.sub(1), Constant(((0.0, 0.0), (0.0, 0.0))), key)
            for node in bc_q.nodes:
                boundary_dofs.append(n_Vp + node)
            bcs.append(bc_q)

    boundary_dofs = sorted(boundary_dofs)
    n_lmb = len(boundary_dofs)

    G = np.zeros((n_V, n_lmb))
    for (i, j) in enumerate(boundary_dofs):
        G[j, i] = 1

    degr = 4
    t = 0.
    t_ = Constant(t)
    t_fin = 1        # total simulation time
    x = mesh.coordinates

    beta = 4*pi/t_fin
    w_exact = sin(pi*x[0]/Lx)*sin(pi*x[1]/Ly)*sin(beta*t_)
    grad_wex = as_vector([pi/Lx*cos(pi*x[0]/Ly)*sin(pi*x[1]/Ly)*sin(beta*t_),
                           pi/Ly*sin(pi*x[0]/Lx)*cos(pi*x[1]/Ly)*sin(beta*t_)])

    v_exact = beta * sin(pi*x[0]/Lx)*sin(pi*x[1]/Ly)*cos(beta*t_)
    grad_vex = as_vector([pi / Lx * cos(pi * x[0] / Ly) * sin(pi * x[1] / Ly) * cos(beta * t_),
                          pi / Ly * sin(pi * x[0] / Lx) * cos(pi * x[1] / Ly) * cos(beta * t_)])

    force = sin(pi*x[0]/Lx)*sin(pi*x[1]/Ly)*sin(beta*t_)*(D *((pi/Lx)**2 + (pi/Ly)**2)**2 - rho*h*beta**2)


    f_form = v_p*force*dx

    J = assemble(j_form, bcs=bcs).M.handle
    M = assemble(m_form, bcs=bcs).M.handle

    # Apply boundary conditions to M, J
    # [bc.apply(J) for bc in bcs]
    # [bc.apply(M) for bc in bcs]

    dt = 0.01
    theta = 0.5

    e_n1 = Function(V, name="e next")
    e_n = Function(V,  name="e old")
    w_n1 = Function(Vp, name="w old")
    w_n = Function(Vp, name="w next")

    fw_ex = Function(Vp, name="w exact")
    fv_ex = Function(Vp, name="v exact")

    fgradw_ex = Function(Vgradp, name="grad w exact")
    fgradv_ex = Function(Vgradp, name="grad v exact")

    e_n.sub(0).assign(interpolate(v_exact, Vp))

    ep_n, eq_n = e_n.split()

    w_n.assign(Constant(0.0))

    n_t = int(floor(t_fin/dt) + 1)

    err_L2 = np.zeros((n_t,))
    w_atP = np.zeros((n_t,))
    v_atP = np.zeros((n_t,))
    Ppoint = (Lx/3, Ly/3)
    v_atP[0] = ep_n.at(Ppoint)

    # w_interpolator = Interpolator(w_exact, fw_ex)
    # v_interpolator = Interpolator(v_exact, fv_ex)
    #
    # gradw_interpolator = Interpolator(grad_wex, fgradw_ex)
    # gradv_interpolator = Interpolator(grad_vex, fgradv_ex)

    fw_ex.interpolate(w_exact)
    fv_ex.interpolate(v_exact)
    fgradw_ex.interpolate(grad_wex)
    fgradv_ex.interpolate(grad_vex)

    t_.assign(t)
    # err_L2[0] = np.sqrt(assemble(dot(w_n-w_exact, w_n-w_exact) *dx
    #                      + dot(grad(w_n) - grad_wex, grad(w_n) - grad_wex) * dx))
    err_L2[0] = np.sqrt(assemble(dot(ep_n - fv_ex, ep_n - fv_ex) * dx
                         + dot(grad(ep_n) - fgradv_ex, grad(ep_n) - fgradv_ex) * dx))

    t_vec = np.linspace(0, t_fin, num=n_t)

    b_ = Function(V)
    ksp = petsc4py.PETSc.KSP().create()
    A = M - dt * theta * J
    ksp.setOperators(A)
    # print(e_n.vector().get_local())
    for i in range(1, n_t):

        t_.assign(t)

        f_n = assemble(f_form, bcs=bcs)
        # [bc.apply(f_n) for bc in bcs]

        t += dt

        t_.assign(t)
        f_n1 = assemble(f_form, bcs=bcs)
        # [bc.apply(f_n1) for bc in bcs]


        # ksp.setUp()

        with e_n.dat.vec as en_vec:
            with e_n1.dat.vec as en1_vec:
                with b_.dat.vec as b_vec:
                    with f_n.dat.vec as fn_vec:
                        with f_n1.dat.vec as fn1_vec:
                                (M + dt * (1 - theta) * J).mult(en_vec, b_vec)
                                b_vec.__add__(dt*(theta*fn1_vec + (1-theta)*fn_vec))
                                ksp.solve(b_vec, en1_vec)
                                values = en1_vec.getValues(list(range(n_V)))
                                e_n1.sub(0).vector()[:] = values[:n_Vp]
                                e_n1.sub(1).vector()[:] = values[n_Vp:]

        ep_n = e_n.sub(0)
        ep_n1 = e_n1.sub(0)

        w_n1.vector()[:] = w_n.vector() + dt/2*(ep_n.vector() + ep_n1.vector())

        e_n.sub(0).assign(e_n1.sub(0))
        e_n.sub(1).assign(e_n1.sub(1))
        w_n.assign(w_n1)

        w_atP[i] = w_n1.at(Ppoint)

        v_atP[i] = ep_n1.at(Ppoint)



        # err_L2[i] = np.sqrt(assemble(dot(w_n1-w_exact, w_n1-w_exact) * dx
        #                      + dot(grad(w_n1)-grad_wex, grad(w_n1)-grad_wex) * dx))
        err_L2[i] = np.sqrt(assemble(dot(ep_n - fv_ex, ep_n - fv_ex) * dx
                         + dot(grad(ep_n) - fgradv_ex, grad(ep_n) - fgradv_ex) * dx))

    plt.figure()
    plt.plot(t_vec, w_atP, 'r-', label=r'approx $w$')
    plt.plot(t_vec, np.sin(pi*Ppoint[0]/Lx)*np.sin(pi*Ppoint[1]/Ly)*np.sin(beta*t_vec), 'b-', label=r'exact $w$')
    # plt.plot(t_vec, v_atP, 'r-', label=r'approx $v$')
    # plt.plot(t_vec, beta * np.sin(pi*Ppoint[0]/Lx)*np.sin(pi*Ppoint[1]/Ly) * np.cos(beta * t_vec), 'b-', label=r'exact $v$')
    plt.xlabel(r'Time [s]')
    plt.title(r'Displacement at $(L_x/2, L_y/2)$')
    plt.legend()
    plt.show()

    err_quad = np.sqrt(np.sum(dt*np.power(err_L2, 2)))
    err_max = max(err_L2)
    err_last = err_L2[-1]
    return err_last, err_max


n_h = 3
n1_vec = np.array([2**(i+1) for i in range(n_h)])
n2_vec = np.array([2**(i) for i in range(n_h)])
h1_vec = 1./n1_vec
h2_vec = 1./n2_vec
err_vec_r1 = np.zeros((n_h,))
err_vec_r2 = np.zeros((n_h,))
errInf_vec_r1 = np.zeros((n_h,))
errInf_vec_r2 = np.zeros((n_h,))

r1_ext = np.zeros((n_h-1,))
r2_ext = np.zeros((n_h-1,))

for i in range(n_h):
    err_vec_r1[i], errInf_vec_r1[i] = compute_err(n1_vec[i], 1)
    err_vec_r2[i], errInf_vec_r2[i] = compute_err(n2_vec[i], 2)

    if i>0:
        r1_ext[i-1] = np.log(err_vec_r1[i]/err_vec_r1[i-1])/np.log(h1_vec[i]/h1_vec[i-1])
        r2_ext[i-1] = np.log(err_vec_r2[i]/err_vec_r2[i-1])/np.log(h2_vec[i]/h2_vec[i-1])

order_r1 = np.polyfit(np.log(h1_vec), np.log(err_vec_r1), 1)[0]
order_r2 = np.polyfit(np.log(h1_vec), np.log(err_vec_r2), 1)[0]
print("Estimated order of convergence: " + str(r1_ext))
print("Estimated order of convergence: " + str(r2_ext))
print("Interpolated order of convergence: " + str(order_r1))
print("Interpolated order of convergence: " + str(order_r2))
plt.figure()
plt.plot(np.log(h1_vec), np.log(err_vec_r1), '-m', label='HHJ 1')
plt.plot(np.log(h2_vec), np.log(err_vec_r2), '-r', label='HHJ 2')
# plt.plot(np.log(h1_vec), np.log(errInf_vec_r1), '-y', label='HHJ 1 $L^\infty$')
# plt.plot(np.log(h2_vec), np.log(errInf_vec_r2), '-o', label='HHJ 2 $L^\infty$')
plt.plot(np.log(h1_vec), np.log(h1_vec), '-g', label=r'$h$')
plt.plot(np.log(h2_vec), np.log(h2_vec**2), '-b', label=r'$h^2$')
plt.xlabel(r'Mesh size')
plt.title(r'Error at $T_f$')
plt.legend()
plt.show()