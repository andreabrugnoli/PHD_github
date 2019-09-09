from firedrake import *
import numpy as np
import scipy.linalg as la
from modules_phdae.classes_phsystem import SysPhdaeRig, check_positive_matrix
import warnings
np.set_printoptions(threshold=np.inf)
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from firedrake.plot import _two_dimension_triangle_func_val
from scipy import integrate
from tools_plotting.animate_2surf import animate2D
import matplotlib.animation as animation
from assimulo.solvers import IDA
from assimulo.implicit_ode import Implicit_Problem
from mpl_toolkits.mplot3d import Axes3D
from math import pi
plt.rc('text', usetex=True)

def print_modes(sys, Vp1, Vp2, n_modes):

    eigenvalues, eigvectors = la.eig(sys.J_f, sys.M_f)
    omega_all = np.imag(eigenvalues)

    index = omega_all > 0

    omega = omega_all[index]
    eigvec_omega = eigvectors[:, index]
    perm = np.argsort(omega)
    eigvec_omega = eigvec_omega[:, perm]

    omega.sort()

    fntsize = 15

    n_Vp1 = Vp1.dim()
    n_Vp2 = Vp2.dim()

    for i in range(int(n_modes)):
        print("Eigenvalue num " + str(i+1) + ":" + str(omega[i]))
        eig_real_p1 = Function(Vp1)
        eig_imag_p1 = Function(Vp1)

        eig_real_p1.vector()[:] = np.real(eigvec_omega[:n_Vp1, i])
        eig_imag_p1.vector()[:] = np.imag(eigvec_omega[:n_Vp1, i])

        eig_real_p2 = Function(Vp2)
        eig_imag_p2 = Function(Vp2)

        eig_real_p2.vector()[:] = np.real(eigvec_omega[n_Vp1:n_Vp1 + n_Vp2, i])
        eig_imag_p2.vector()[:] = np.imag(eigvec_omega[n_Vp1:n_Vp1 + n_Vp2, i])

        norm_real_eig = np.linalg.norm(np.real(eigvec_omega[:n_Vp1+n_Vp2, i]))
        norm_imag_eig = np.linalg.norm(np.imag(eigvec_omega[:n_Vp1+n_Vp2, i]))

        figure = plt.figure()
        ax = figure.add_subplot(111, projection="3d")

        if norm_imag_eig > norm_real_eig:
            # plot(eig_imag_p1, axes=ax, plot3d=True)
            # plot(eig_imag_p2, axes=ax, plot3d=True)
            triangulation1, z1_goodeig = _two_dimension_triangle_func_val(eig_imag_p1, 10)
            triangulation2, z2_goodeig = _two_dimension_triangle_func_val(eig_imag_p2, 10)
        else:
            # plot(eig_real_p1, axes=ax, plot3d=True)
            # plot(eig_real_p2, axes=ax, plot3d=True)
            triangulation1, z1_goodeig = _two_dimension_triangle_func_val(eig_real_p1, 10)
            triangulation2, z2_goodeig = _two_dimension_triangle_func_val(eig_real_p2, 10)

        ax.set_xlabel('$x [m]$', fontsize=fntsize)
        ax.set_ylabel('$y [m]$', fontsize=fntsize)
        ax.set_title('Eigenvector num ' + str(i + 1), fontsize=fntsize)

        ax.w_zaxis.set_major_locator(LinearLocator(10))
        ax.w_zaxis.set_major_formatter(FormatStrFormatter('%1.2g'))

        plot_eig1 = ax.plot_trisurf(triangulation1, z1_goodeig, cmap=cm.jet)
        plot_eig1._facecolors2d = plot_eig1._facecolors3d
        plot_eig1._edgecolors2d = plot_eig1._edgecolors3d

        plot_eig2 = ax.plot_trisurf(triangulation2, z2_goodeig, cmap=cm.jet)
        plot_eig2._facecolors2d = plot_eig2._facecolors3d
        plot_eig2._edgecolors2d = plot_eig2._edgecolors3d

        ax.legend(("mesh 1", "mesh2"))

        path_figs = "/home/a.brugnoli/Plots_Videos/Python/Plots/Waves/"
        # plt.savefig(path_figs + "Eig_n" + str(i) + ".eps")

    plt.show()


def create_sys(mesh1, mesh2, deg_p1, deg_q1, deg_p2, deg_q2):
    x1, y1 = SpatialCoordinate(mesh1)

    tab_coord1 = mesh1.coordinates.dat.data
    x1_cor = tab_coord1[:, 0]
    y1_cor = tab_coord1[:, 1]

    Vp1 = FunctionSpace(mesh1, "CG", deg_p1)
    Vq1 = FunctionSpace(mesh1, "RT", deg_q1)

    V1 = Vp1 * Vq1

    n_1 = V1.dim()
    np_1 = Vp1.dim()
    nq_1 = Vq1.dim()

    v1 = TestFunction(V1)
    v_p1, v_q1 = split(v1)

    e1 = TrialFunction(V1)
    e_p1, e_q1 = split(e1)

    rho = 0.1
    T = as_tensor([[10, 0], [0, 10]])

    al_p1 = rho * e_p1
    al_q1 = dot(inv(T), e_q1)

    dx1 = Measure('dx', domain=mesh1)
    ds1 = Measure('ds', domain=mesh1)

    m_p1 = v_p1 * al_p1 * dx1
    m_q1 = dot(v_q1, al_q1) * dx1

    m_form1 = m_p1 + m_q1

    j_div = dot(v_p1, div(e_q1)) * dx1
    j_divIP = -dot(div(v_q1), e_p1) * dx1

    j_form1 = j_div + j_divIP
    petsc_j1 = assemble(j_form1, mat_type='aij').M.handle
    petsc_m1 = assemble(m_form1, mat_type='aij').M.handle

    J1 = np.array(petsc_j1.convert("dense").getDenseArray())
    M1 = np.array(petsc_m1.convert("dense").getDenseArray())

    Vf1 = FunctionSpace(mesh1, "CG", 1)
    f_D = TrialFunction(Vf1)
    v_D = TestFunction(Vf1)

    m_delta = v_D * f_D * ds1

    petsc_md = assemble(m_delta, mat_type='aij').M.handle
    Mdelta = np.array(petsc_md.convert("dense").getDenseArray())

    n_ver = FacetNormal(mesh1)

    u_Dxy1 = pow(x1, 2) - pow(y1, 2)
    u_N1 = pow(x1, 2) + pow(y1, 2)

    ray1 = sqrt(pow(x1, 2) + pow(y1, 2))
    is_D = conditional(lt(ray1, 1.25), 1, 0)
    is_int1 = conditional(gt(ray1, 1.25), 1, 0)

    b_Dxy = dot(v_q1, n_ver) * u_Dxy1 * is_D * ds1
    b_Dt = dot(v_q1, n_ver) * is_D * ds1

    B_Dxy = np.reshape(assemble(b_Dxy).vector().get_local(), (-1, 1))
    B_Dt = np.reshape(assemble(b_Dt).vector().get_local(), (-1, 1))

    b_int1 = dot(v_q1, n_ver) * f_D * is_int1 * ds1

    petsc_bint1 = assemble(b_int1, mat_type='aij').M.handle
    B_int1 = np.array(petsc_bint1.convert("dense").getDenseArray())

    th_cor = np.arctan2(y1_cor, x1_cor)
    perm_th = np.argsort(th_cor)
    B_int1 = B_int1[:, perm_th]

    Mdelta = Mdelta[:, perm_th]
    Mdelta = Mdelta[perm_th, :]

    int_dofs1 = np.where(B_int1.any(axis=0))[0]

    Mdelta = Mdelta[:, int_dofs1]
    Mdelta = Mdelta[int_dofs1, :]

    B_int1 = B_int1[:, int_dofs1]

    sys1 = SysPhdaeRig(n_1, 0, 0, np_1, nq_1, E=M1, J=J1, B=np.concatenate((B_int1, B_Dxy, B_Dt), axis=1))

    x2, y2 = SpatialCoordinate(mesh2)

    tab_coord2 = mesh2.coordinates.dat.data
    x2_cor = tab_coord2[:, 0]
    y2_cor = tab_coord2[:, 1]

    Vp2 = FunctionSpace(mesh2, "CG", deg_p2)
    Vq2 = FunctionSpace(mesh2, "RT", deg_q2)

    V2 = Vp2 * Vq2

    n_2 = V2.dim()
    np_2 = Vp2.dim()
    nq_2 = Vq2.dim()

    v2 = TestFunction(V2)
    v_p2, v_q2 = split(v2)

    e2 = TrialFunction(V2)
    e_p2, e_q2 = split(e2)

    al_p2 = rho * e_p2
    al_q2 = dot(inv(T), e_q2)

    dx2 = Measure('dx', domain=mesh2)
    ds2 = Measure('ds', domain=mesh2)

    m_p2 = v_p2 * al_p2 * dx2
    m_q2 = dot(v_q2, al_q2) * dx2
    m_form2 = m_p2 + m_q2

    j_grad = dot(v_q2, grad(e_p2)) * dx2
    j_gradIP = -dot(grad(v_p2), e_q2) * dx2
    j_form2 = j_grad + j_gradIP

    petsc_j2 = assemble(j_form2, mat_type='aij').M.handle
    petsc_m2 = assemble(m_form2, mat_type='aij').M.handle
    J2 = np.array(petsc_j2.convert("dense").getDenseArray())
    M2 = np.array(petsc_m2.convert("dense").getDenseArray())

    Vf2 = FunctionSpace(mesh2, "CG", 1)
    f_N = TrialFunction(Vf2)

    ray2 = sqrt(pow(x2, 2) + pow(y2, 2))
    is_N = conditional(gt(ray2, 1.75), 1, 0)
    is_int2 = conditional(lt(ray2, 1.75), 1, 0)

    u_Dxy2 = pow(x2, 2) - pow(y2, 2)
    u_N2 = pow(x2, 2) + pow(y2, 2)

    b_N = v_p2 * u_N2 * is_N * ds2

    B_N = np.reshape(assemble(b_N).vector().get_local(), (-1, 1))

    b_int2 = v_p2 * f_N * is_int2 * ds2

    petsc_bint2 = assemble(b_int2, mat_type='aij').M.handle
    B_int2 = np.array(petsc_bint2.convert("dense").getDenseArray())

    th_cor = np.arctan2(y2_cor, x2_cor)
    perm_th = np.argsort(th_cor)
    B_int2 = B_int2[:, perm_th]

    int_dofs2 = np.where(B_int2.any(axis=0))[0]

    B_int2 = B_int2[:, int_dofs2]

    sys2 = SysPhdaeRig(n_2, 0, 0, np_2, nq_2, E=M2, J=J2, B=np.concatenate((B_int2, B_N), axis=1))

    m1 = sys1.m
    m2 = sys2.m
    sys_DN = SysPhdaeRig.gyrator_ordered(sys1, sys2, list(range(m1 - 2)), list(range(m2 - 1)), la.inv(Mdelta))

    ep1_0 = project(u_Dxy1, Vp1).vector().get_local()
    eq1_0 = project(as_vector([u_N1 * x1 / ray1, u_N1 * y1 / ray1]), Vq1).vector().get_local()

    ep2_0 = project(u_Dxy2, Vp2).vector().get_local()
    eq2_0 = project(as_vector([u_N2 * x2 / ray2, u_N2 * y2 / ray2]), Vq2).vector().get_local()

    ep_0 = np.concatenate((ep1_0, ep2_0))
    eq_0 = np.concatenate((eq1_0, eq2_0))

    return sys_DN, Vp1, Vp2, ep_0, eq_0


ind = 9
path_mesh = "/home/a.brugnoli/GitProjects/PythonProjects/ph_firedrake/waves/meshes/"
mesh1 = Mesh(path_mesh + "circle1_" + str(ind) + ".msh")
mesh2 = Mesh(path_mesh + "circle2_" + str(ind) + ".msh")


figure = plt.figure()
ax = figure.add_subplot(111)
plot(mesh1, axes=ax)
plot(mesh2, axes=ax)
plt.show()

degp1 = 1
degq1 = 2

degp2 = 1
degq2 = 2

sys_DN, Vp1, Vp2, ep_0, eq_0 = create_sys(mesh1, mesh2, degp1, degq1, degp2, degq2)
JJ = sys_DN.J
MM = sys_DN.E
BB = sys_DN.B

n_p = len(ep_0)
n_q = len(eq_0)
n_e = n_p + n_q
print(n_e)


B_Dxy = np.reshape(BB[:, 0], (-1, ))
B_Dt = np.reshape(BB[:, 1], (-1, ))
B_N = np.reshape(BB[:, 2], (-1, ))

t_final = 1
om_D = 2*pi/t_final
om_N = om_D/8


def ode_closed_phs(t, y, yd):

    res = MM @ yd - JJ @ y - B_Dxy - B_Dt * np.sin(om_D*t) - B_N * (1 + np.sin(om_N * t))

    return res


order = []


def handle_result(solver, t, y, yd):

    order.append(solver.get_last_order())

    solver.t_sol.extend([t])
    solver.y_sol.extend([y])
    solver.yd_sol.extend([yd])


y0 = np.concatenate((ep_0, eq_0))

yd0 = la.solve(MM, JJ @ y0 + B_Dxy + B_N) # Initial conditions

# Create an Assimulo implicit problem
imp_mod = Implicit_Problem(ode_closed_phs, y0, yd0, name='dae_closed_pHs')
imp_mod.handle_result = handle_result

# Set the algebraic components
imp_mod.algvar = list(np.ones(n_e))

# Create an Assimulo implicit solver (IDA)
imp_sim = IDA(imp_mod)  # Create a IDA solver

# Sets the paramters
imp_sim.atol = 1e-6  # Default 1e-6
imp_sim.rtol = 1e-6  # Default 1e-6
imp_sim.suppress_alg = True  # Suppress the algebraic variables on the error test
imp_sim.report_continuously = True
# imp_sim.maxh = 1e-6

# Let Sundials find consistent initial conditions by use of 'IDA_YA_YDP_INIT'
imp_sim.make_consistent('IDA_YA_YDP_INIT')

# Simulate
n_ev = 100
t_ev = np.linspace(0, t_final, n_ev)
t_sol, y_sol, yd_sol = imp_sim.simulate(t_final, 0, t_ev)

y_sol = y_sol.T
yd_sol = yd_sol.T

n_ev = len(t_sol)
dt_vec = np.diff(t_sol)


n_p1 = Vp1.dim()
n_p2 = Vp2.dim()

ep1_sol = y_sol[:n_p1, :]
ep2_sol = y_sol[n_p1:n_p1+n_p2, :]
ep_sol = y_sol[:n_p1+n_p2, :]

e_sol = y_sol

H_vec = np.zeros((n_ev,))
for i in range(n_ev):
    H_vec[i] = 0.5 * (e_sol[:, i].T @ MM @ e_sol[:, i])

# np.save("t_ode.npy", t_sol)
# np.save("H_ode.npy", H_vec)

fig = plt.figure()
plt.plot(t_ev, H_vec, 'b-')
plt.xlabel(r'{Time} (s)', fontsize=16)
plt.ylabel(r'{Hamiltonian} (J)', fontsize=16)
plt.title(r"Hamiltonian trend",
          fontsize=16)
plt.legend(loc='upper left')

w1_sol = np.zeros(ep1_sol.shape)
w0_1 = np.zeros((n_p1,))
w1_sol[:, 0] = w0_1
w1_old = w0_1

w2_sol = np.zeros(ep2_sol.shape)
w0_2 = np.zeros((n_p2,))
w2_sol[:, 0] = w0_2
w2_old = w0_2

dt_vec = np.diff(t_ev)

for i in range(1, n_ev):
    w1_sol[:, i] = w1_old + 0.5 * (ep1_sol[:, i - 1] + ep1_sol[:, i]) * dt_vec[i-1]
    w_pl1_old = w1_sol[:, i]

    w2_sol[:, i] = w2_old + 0.5 * (ep2_sol[:, i - 1] + ep2_sol[:, i]) * dt_vec[i-1]
    w2_old = w2_sol[:, i]

w1_fun = Function(Vp1)
w2_fun = Function(Vp2)
w1fun_vec = []
w2fun_vec = []

maxZ = np.max(ep_sol)
minZ = np.min(ep_sol)

for i in range(n_ev):
    w1_fun.vector()[:] = ep1_sol[:, i]
    w1fun_vec.append(interpolate(w1_fun, Vp1))

    w2_fun.vector()[:] = ep2_sol[:, i]
    w2fun_vec.append(interpolate(w2_fun, Vp2))

anim = animate2D(minZ, maxZ, w1fun_vec, w2fun_vec, t_ev, xlabel = '$x[m]$', ylabel = '$y [m]$', \
                         zlabel = '$w [mm]$', title = 'Vertical Displacement')

plt.show()

Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
pathout = './'
anim.save(pathout + 'wave_ode.mp4', writer=writer)


