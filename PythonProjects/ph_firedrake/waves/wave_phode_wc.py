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

q_1 = 0.8163  # m^3 kg ^-1   1/mu_0
mu_0 = 1 / q_1

q_2 = 1.4161 * 10 ** 5  # Pa   1/xsi
xi_s = 1 / q_2
c_0 = 340  # 340 m/s
ind = 15


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
    x1, r1 = SpatialCoordinate(mesh1)
    R_ext = 1
    L_duct = 2
    tol_geo = 1e-9

    tab_coord1 = mesh1.coordinates.dat.data
    x1_cor = tab_coord1[:, 0]
    r1_cor = tab_coord1[:, 1]

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

    al_p1 = xi_s * e_p1
    al_q1 = mu_0 * e_q1

    dx1 = Measure('dx', domain=mesh1)
    ds1 = Measure('ds', domain=mesh1)

    m_p1 = v_p1 * al_p1 * r1 * dx1
    m_q1 = dot(v_q1, al_q1) * r1 * dx1

    m_form1 = m_p1 + m_q1

    j_div = v_p1 * (div(e_q1) + e_q1[1]/r1) * r1 * dx1
    j_divIP = -(div(v_q1) + v_q1[1]/r1) * e_p1 * r1 * dx1

    j_form1 = j_div + j_divIP

    petsc_j1 = assemble(j_form1, mat_type='aij').M.handle
    petsc_m1 = assemble(m_form1, mat_type='aij').M.handle

    J1 = np.array(petsc_j1.convert("dense").getDenseArray())
    M1 = np.array(petsc_m1.convert("dense").getDenseArray())

    Vf1 = FunctionSpace(mesh1, "CG", 1)
    f_D = TrialFunction(Vf1)
    v_D = TestFunction(Vf1)
    u_D = TrialFunction(Vf1)

    m_delta = v_D * f_D * r1 * ds1

    petsc_md = assemble(m_delta, mat_type='aij').M.handle
    Mdelta = np.array(petsc_md.convert("dense").getDenseArray())

    n_ver1 = FacetNormal(mesh1)

    is_D = conditional(gt(r1, R_ext - tol_geo), 1, 0)
    is_int1 = conditional(lt(r1, R_ext - tol_geo), 1, 0)
    # is_int1 = conditional(ne(r1, R_ext), 1, 0)

    is_uD = conditional(And(gt(r1, R_ext - tol_geo), And(gt(x1, L_duct / 3), lt(x1, 2 * L_duct / 3))), 1, 0)

    b_D = dot(v_q1, n_ver1) * u_D * is_uD * r1 * ds1

    petsc_bD = assemble(b_D, mat_type='aij').M.handle

    B_D = np.array(petsc_bD.convert("dense").getDenseArray())
    controlD_dofs = np.where(B_D.any(axis=0))[0]

    B_D = B_D[:, controlD_dofs]

    nu_D = len(controlD_dofs)

    b_int1 = dot(v_q1, n_ver1) * f_D * is_int1 * r1 * ds1

    petsc_bint1 = assemble(b_int1, mat_type='aij').M.handle
    B_int1 = np.array(petsc_bint1.convert("dense").getDenseArray())

    perm_x1 = np.argsort(x1_cor)
    B_int1 = B_int1[:, perm_x1]

    Mdelta = Mdelta[:, perm_x1]
    Mdelta = Mdelta[perm_x1, :]

    int_dofs1 = np.where(B_int1.any(axis=0))[0]

    Mdelta = Mdelta[:, int_dofs1]
    Mdelta = Mdelta[int_dofs1, :]

    B_int1 = B_int1[:, int_dofs1]

    nint_1 = len(int_dofs1)

    sys1 = SysPhdaeRig(n_1, 0, 0, np_1, nq_1, E=M1, J=J1, B=np.concatenate((B_int1, B_D), axis=1))

    x2, r2 = SpatialCoordinate(mesh2)

    tab_coord2 = mesh2.coordinates.dat.data
    x2_cor = tab_coord2[:, 0]
    r2_cor = tab_coord2[:, 1]

    # ind_x = np.where(np.isclose(x2_cor, 1), np.isclose(r2_cor, 1))[0][0]
    # print(ind_x, x2_cor[ind_x], r2_cor[ind_x])
    # x12_cor = np.concatenate((x1_cor, x2_cor))
    # r12_cor = np.concatenate((r1_cor, r2_cor))
    ind_x = np_1 + np.where(np.logical_and(np.isclose(x2_cor, 0), \
                                           np.isclose(r2_cor, 0)))[0][0]
    # print(x12_cor[ind_x], r12_cor[ind_x])

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

    al_p2 = xi_s * e_p2
    al_q2 = mu_0 * e_q2

    dx2 = Measure('dx', domain=mesh2)
    ds2 = Measure('ds', domain=mesh2)

    m_p2 = v_p2 * al_p2 * r2 * dx2
    m_q2 = dot(v_q2, al_q2) * r2 * dx2
    m_form2 = m_p2 + m_q2

    # j_div2 = v_p2 * (div(e_q2) + e_q2[1] / r2) * r2 * dx2
    # j_divIP2 = -(div(v_q2) + v_q2[1] / r2) * e_p2 * r2 * dx2
    # j_form2 = j_div2 + j_divIP2

    j_grad = dot(v_q2, grad(e_p2)) * r2 * dx2
    j_gradIP = -dot(grad(v_p2), e_q2) * r2 * dx2
    j_form2 = j_grad + j_gradIP

    petsc_j2 = assemble(j_form2, mat_type='aij').M.handle
    petsc_m2 = assemble(m_form2, mat_type='aij').M.handle
    J2 = np.array(petsc_j2.convert("dense").getDenseArray())
    M2 = np.array(petsc_m2.convert("dense").getDenseArray())

    Vf2 = FunctionSpace(mesh2, "CG", 1)
    f_N = TrialFunction(Vf2)
    v_N = TestFunction(Vf1)
    u_N = TrialFunction(Vf1)

    isL_uN = conditional(lt(x2, tol_geo), 1, 0)
    isR_uN = conditional(gt(x2, L_duct - tol_geo), 1, 0)

    is_N = conditional(Or(Or(lt(x2, tol_geo), gt(x2, L_duct - tol_geo)), lt(r2, tol_geo)), 1, 0)
    is_int2 = conditional(And(And(gt(x2, tol_geo), lt(x2, L_duct-tol_geo)), gt(r2, R_ext/2-tol_geo)), 1, 0)

    ux_N1 = 1 - r1**2/R_ext**2
    ux_N2 = 1 - r2**2/R_ext**2

    uy_N1 = 16*r1**2*(R_ext-r1)**2
    uy_N2 = 16*r2**2*(R_ext-r2)**2

    b_N = -v_p2 * ux_N2 * isL_uN * r2 * ds2 + v_p2 * ux_N2 * isR_uN * r2 * ds2

    B_N = np.reshape(assemble(b_N).vector().get_local(), (-1, 1))

    b_int2 = v_p2 * f_N * is_int2 * r2 * ds2

    petsc_bint2 = assemble(b_int2, mat_type='aij').M.handle
    B_int2 = np.array(petsc_bint2.convert("dense").getDenseArray())

    perm_x2 = np.argsort(x2_cor)
    B_int2 = B_int2[:, perm_x2]

    int_dofs2 = np.where(B_int2.any(axis=0))[0]

    B_int2 = B_int2[:, int_dofs2]
    #
    # x1_cor = x1_cor[perm_x1]
    # r1_cor = r1_cor[perm_x1]
    # x2_cor = x2_cor[perm_x2]
    # r2_cor = r2_cor[perm_x2]
    #
    # plt.plot(x1_cor[:], r1_cor[:], 'bo')
    # plt.plot(x1_cor[int_dofs1], r1_cor[int_dofs1], 'r*')
    # plt.show()
    #
    # plt.plot(x2_cor[:], r2_cor[:], 'bo')
    # plt.plot(x2_cor[int_dofs2], r2_cor[int_dofs2], 'r*')
    # plt.show()

    nint_2 = len(int_dofs2)
    # print(nint_1, nint_2)

    sys2 = SysPhdaeRig(n_2, 0, 0, np_2, nq_2, E=M2, J=J2, B=np.concatenate((B_int2, B_N), axis=1))

    m1 = sys1.m
    m2 = sys2.m
    sys_DN = SysPhdaeRig.gyrator_ordered(sys1, sys2, list(range(nint_1)), list(range(nint_2)), la.inv(Mdelta))

    ep1_0 = project(Constant(0), Vp1).vector().get_local()
    eq1_0 = project(as_vector([ux_N1, uy_N1]), Vq1).vector().get_local()

    ep2_0 = project(Constant(0), Vp2).vector().get_local()
    eq2_0 = project(as_vector([ux_N2, uy_N2]), Vq2).vector().get_local()

    ep_0 = np.concatenate((ep1_0, ep2_0))
    eq_0 = np.concatenate((eq1_0, eq2_0))

    return sys_DN, Vp1, Vp2, ep_0, eq_0, ind_x


path_mesh = "/home/a.brugnoli/GitProjects/PythonProjects/ph_firedrake/waves/meshes_ifacwc/"
mesh1 = Mesh(path_mesh + "duct_dom1_" + str(ind) + ".msh")
mesh2 = Mesh(path_mesh + "duct_dom2_" + str(ind) + ".msh")

figure = plt.figure()
ax = figure.add_subplot(111)
colors1 = {"colors": 'b'}
colors2 = {"colors": 'r'}
plot(mesh1, axes=ax, **colors1)
plot(mesh2, axes=ax, **colors2)
# plt.xlabel(r"x")
# plt.ylabel(r"r")
plt.show()

path_figs = "/home/a.brugnoli/Plots_Videos/Python/Plots/Waves/IFAC_WC2020/"
plt.savefig(path_figs + "Mesh_" + str(ind) + ".eps", format="eps")
degp1 = 1
degq1 = 2

degp2 = 1
degq2 = 2

sys_DN, Vp1, Vp2, ep_0, eq_0, ind_x = create_sys(mesh1, mesh2, degp1, degq1, degp2, degq2)
JJ = sys_DN.J
MM = sys_DN.E
BB = sys_DN.B

print_modes(sys_DN, Vp1, Vp2, 10)

m_sysDN = sys_DN.m
n_p = len(ep_0)
n_q = len(eq_0)
n_e = n_p + n_q
print(n_e)


B_D = BB[:, :-1]
B_N = BB[:, -1]
print(m_sysDN, B_D.shape, B_N.shape)

t_final = 0.1
Z = c_0 * mu_0
t_diss = 0.2*t_final
tau_imp = t_final/100

invMM = la.inv(MM)
RR = Z * B_D @ B_D.T

def ode_closed_phs(t, y, yd):

    print(t/t_final*100)

    ft_imp = (t>t_diss) # * (1 - np.exp((t - t_diss)/tau_imp))
    ft_ctrl = 1 # (t<t_diss)
    res = yd - invMM @ ((JJ - RR * ft_imp) @ y + B_N * ft_ctrl)

    return res


order = []


def handle_result(solver, t, y, yd):

    order.append(solver.get_last_order())

    solver.t_sol.extend([t])
    solver.y_sol.extend([y])
    solver.yd_sol.extend([yd])


y0 = np.concatenate((ep_0, eq_0))

yd0 = la.solve(MM, JJ @ y0 + B_N) # Initial conditions

# Create an Assimulo implicit problem
imp_mod = Implicit_Problem(ode_closed_phs, y0, yd0, name='ode_closed_pHs')
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
n_ev = 500
t_ev = np.linspace(0, t_final, n_ev)
t_sol, y_sol, yd_sol = imp_sim.simulate(t_final, 0, t_ev)

y_sol = y_sol.T
yd_sol = yd_sol.T

n_ev = len(t_sol)
dt_vec = np.diff(t_sol)


n_p1 = Vp1.dim()
n_p2 = Vp2.dim()

e_sol = y_sol

ep_sol = e_sol[:n_p1+n_p2, :]
eq_sol = e_sol[n_p1+n_p2:, :]
ep1_sol = ep_sol[:n_p1, :]
ep2_sol = ep_sol[n_p1:, :]

MMp = MM[:n_p1 + n_p2, :n_p1 + n_p2]
MMq = MM[n_p1 + n_p2:, n_p1 + n_p2:]

H_vec = np.zeros((n_ev,))
Hp_vec = np.zeros((n_ev,))
Hq_vec = np.zeros((n_ev,))

for i in range(n_ev):
    H_vec[i] = 0.5 * (e_sol[:, i].T @ MM @ e_sol[:, i])
    Hp_vec[i] = 0.5 * (ep_sol[:, i].T @ MMp @ ep_sol[:, i])
    Hq_vec[i] = 0.5 * (eq_sol[:, i].T @ MMq @ eq_sol[:, i])

path_results = "/home/a.brugnoli/GitProjects/PythonProjects/ph_firedrake/waves/results_ifacwc/"
np.save(path_results + "t_ode_" + str(ind) + ".npy", t_sol)
np.save(path_results + "H_ode_" + str(ind) + ".npy", H_vec)
np.save(path_results + "Hp_ode_" + str(ind) + ".npy", Hp_vec)
np.save(path_results + "Hq_ode_" + str(ind) + ".npy", Hq_vec)

fntsize = 16

fig = plt.figure()
plt.plot(t_ev, H_vec, 'b-', label= "H")
plt.plot(t_ev, Hp_vec, 'r-', label= "Hp")
plt.plot(t_ev, Hq_vec, 'g-', label= "Hq")

plt.xlabel(r'{Time} (s)', fontsize=fntsize)
plt.ylabel(r'{Hamiltonian} (J)', fontsize=fntsize)
plt.title(r"Hamiltonian trend",
          fontsize=fntsize)
plt.legend(loc='upper right')

path_figs = "/home/a.brugnoli/Plots_Videos/Python/Plots/Waves/IFAC_WC2020/"
plt.savefig(path_figs + "H_ode" + str(ind) + ".eps", format="eps")

dt_vec = np.diff(t_ev)

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

anim = animate2D(minZ, maxZ, w1fun_vec, w2fun_vec, t_ev, xlabel = '$x[m]$', ylabel = '$r [m]$', \
                         zlabel = '$p [Pa]$', title = 'Pressure')

Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
path_videos = "/home/a.brugnoli/Plots_Videos/Python/Videos/Waves/IFAC_WC2020/"
anim.save(path_videos + 'wave_ode' + str(ind) + '.mp4', writer=writer)

plt.show()

