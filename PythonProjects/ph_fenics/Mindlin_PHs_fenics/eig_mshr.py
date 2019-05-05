# Mindlin plate written with the port Hamiltonian approach

from fenics import *
import numpy as np
np.set_printoptions(threshold=np.inf)
import mshr
import scipy.linalg as la
from Mindlin_PHs_fenics.parameters import *
from modules_phdae.classes_phsystem import SysPhdaeRig
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
plt.rc('text', usetex=True)

deg = 2

L = 1

D = E * h ** 3 / (1 - nu ** 2) / 12.
G = E / 2 / (1 + nu)
F = G * h * k

# I_w = 1. / (rho * h)
pho_rot = (rho * h ** 3)/12.

# Useful Matrices

D_b = as_tensor([
  [D, D * nu, 0],
  [D * nu, D, 0],
  [0, 0, D * (1 - nu) / 2]
])

fl_rot = 12. / (E * h ** 3)
C_b = as_tensor([
    [fl_rot, -nu * fl_rot, 0],
    [-nu * fl_rot, fl_rot, 0],
    [0, 0, fl_rot * (1 + nu) / 2]
])


# Operators and functions
def gradSym(u):
    return 0.5 * (nabla_grad(u) + nabla_grad(u).T)
    # return sym(nabla_grad(u))


def strain2voigt(eps):
    return as_vector([eps[0, 0], eps[1, 1], 2 * eps[0, 1]])


def voigt2stress(S):
    return as_tensor([[S[0], S[2]], [S[2], S[1]]])


def bending_moment(u):
    return voigt2stress(dot(D_b, strain2voigt(u)))


def bending_curv(u):
    return voigt2stress(dot(C_b, strain2voigt(u)))

# The unit square mesh is divided in :math:`N\times N` quadrilaterals::


rect = mshr.Rectangle(Point(0, 0), Point(Lx, Ly))
hole = mshr.Circle(Point(0, 0), r)

domain = rect - hole
mesh = mshr.generate_mesh(domain, n)

plot(mesh)
plt.show()

# Domain, Subdomains, Boundary, Suboundaries
class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0]**2 + x[1]**2 - r**2) < 1e-2 and\
               x[0] <= r and x[1] <= r and on_boundary
# Boundary conditions on rotations


bottom = Bottom()
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
bottom.mark(boundaries, 1)

# Finite element defition

P_pw = FiniteElement('CG', triangle, deg)
P_pth = VectorElement('CG', triangle, deg)
P_qth = TensorElement('CG', triangle, deg, shape=(2, 2), symmetry=True)
P_qw = VectorElement('CG', triangle, deg) # FiniteElement('RT', triangle, deg) #


element = MixedElement([P_pw, P_pth, P_qth, P_qw])
V = FunctionSpace(mesh, element)

v = TestFunction(V)
v_pw, v_pth, v_qth, v_qw = split(v)

e_v = TrialFunction(V)
e_pw, e_pth, e_qth, e_qw = split(e_v)

al_pw = rho * h * e_pw
al_pth = (rho * h ** 3) / 12. * e_pth
al_qth = bending_curv(e_qth)
al_qw = 1. / F * e_qw

dx = Measure('dx')
ds = Measure('ds', subdomain_data=boundaries)

m = inner(v_pw, al_pw) * dx + inner(v_pth, al_pth) * dx + inner(v_qth, al_qth) * dx + inner(v_qw, al_qw) * dx

j_grad = dot(v_qw, grad(e_pw)) * dx
j_gradIP = -dot(grad(v_pw), e_qw) * dx

j_gradSym = inner(v_qth, gradSym(e_pth)) * dx
j_gradSymIP = -inner(gradSym(v_pth), e_qth) * dx

j_Id = dot(v_pth, e_qw) * dx
j_IdIP = -dot(v_qw, e_pth) * dx

j_allgrad = j_grad + j_gradIP + j_gradSym + j_gradSymIP + j_Id + j_IdIP

j = j_allgrad

n = FacetNormal(mesh)
s = as_vector([ -n[1], n[0] ])

P_qn = FiniteElement('CG', triangle, 1)
P_Mnn = FiniteElement('CG', triangle, 1)
P_Mns = FiniteElement('CG', triangle, 1)

element_u = MixedElement([P_qn, P_Mnn, P_Mns])

Vu = FunctionSpace(mesh, element_u)

q_n, M_nn, M_ns = TrialFunction(Vu)

v_omn = dot(v_pth, n)
v_oms = dot(v_pth, s)

b_form = v_pw * q_n * ds(1) + v_omn * M_nn * ds(1) + v_oms * M_ns * ds(1)

JJ = assemble(j).array()
MM = assemble(m).array()
B_in = assemble(b_form).array()

boundary_dofs = np.where(B_in.any(axis=0))[0]
B_in = B_in[:, boundary_dofs]

n_tot = V.dim()
n_u = len(boundary_dofs)
# print(N_u)

Z_u = np.zeros((n_u, n_u))

J_aug = np.vstack([np.hstack([JJ, B_in]),
                    np.hstack([-B_in.T, Z_u])])


M_aug = la.block_diag(MM, Z_u)
tol = 10**(-9)

eigenvalues, eigvectors = la.eig(J_aug, M_aug)
omega_all = np.imag(eigenvalues)

index = omega_all > tol

omega = omega_all[index]
eigvec_omega = eigvectors[:, index]
perm = np.argsort(omega)
eigvec_omega = eigvec_omega[:, perm]

omega.sort()

nconv = len(omega)

print("Eigenvalues full system")
for i in range(10):
    print(omega[i]/(2*pi))

d = mesh.geometry().dim()
dofV_x = V.tabulate_dof_coordinates().reshape((-1, d))

dofs_Vpw = V.sub(0).dofmap().dofs()
dofs_Vpth = V.sub(1).dofmap().dofs()
dofs_Vqth = V.sub(2).dofmap().dofs()
dofs_Vqw = V.sub(3).dofmap().dofs()

dofVpw_x = dofV_x[dofs_Vpw]

x = dofVpw_x[:, 0]
y = dofVpw_x[:, 1]

eigvec_w = eigvec_omega[dofs_Vpw, :]
eigvec_w_real = np.real(eigvec_w)
eigvec_w_imag = np.imag(eigvec_w)

dofs_Vq = dofs_Vqth + dofs_Vqw

n_pw = len(dofs_Vpw)
n_pth = len(dofs_Vpth)
n_p = n_pw + n_pth
n_q = len(dofs_Vq)

n_fl = n_p + n_q

Mpw = MM[:, dofs_Vpw]
Mpw = Mpw[dofs_Vpw, :]

Mpth = MM[:, dofs_Vpth]
Mpth = Mpth[dofs_Vpth, :]

Mq = MM[:, dofs_Vq]
Mq = Mq[dofs_Vq, :]

Dqw = JJ[:, dofs_Vq]
Dqw = Dqw[dofs_Vpw, :]

Dqth = JJ[:, dofs_Vq]
Dqth = Dqth[dofs_Vpth, :]

Dpw = JJ[:, dofs_Vpw]
Dpw = Dpw[dofs_Vq, :]

Dpth = JJ[:, dofs_Vpth]
Dpth = Dpth[dofs_Vq, :]

Bpw = B_in[dofs_Vpw]
Bpth = B_in[dofs_Vpth]
Bq = B_in[dofs_Vq]

M_ord = la.block_diag(Mpw, Mpth, Mq)
J_ord = np.zeros((n_fl, n_fl))

J_ord[:n_pw, n_p:] = Dqw
J_ord[n_pw:n_p, n_p:] = Dqth
J_ord[n_p:, :n_pw] = Dpw
J_ord[n_p:, n_pw:n_p] = Dpth

B_ord = np.concatenate((Bpw, Bpth, Bq), axis=0)

# from IPython import embed; embed()


plate = SysPhdaeRig(n_fl, 0, 0, n_p, n_q, E=M_ord, J=J_ord, B=B_ord)

n_red = 5
s0 = 0.001

plate_red, V_red = plate.reduce_system(s0, n_red)
Jred = plate_red.J_f
Mred = plate_red.M_f
Bred = plate_red.B_f

np_red = plate_red.n_p
nf_red = plate_red.n_f
npw_red = int(np_red/3)

Jred_aug = np.vstack([np.hstack([Jred, Bred]),
                    np.hstack([-Bred.T, Z_u])])

Mred_aug = la.block_diag(Mred, Z_u)

tol = 10**(-9)

eigenvalues, eigvectors = la.eig(Jred_aug, Mred_aug)
omega_all = np.imag(eigenvalues)

index = omega_all > tol

omega = omega_all[index]
eigvec_omega = eigvectors[:, index]
perm = np.argsort(omega)
eigvec_omega = eigvec_omega[:, perm]

omega.sort()

print("Eigenvalues reduced system")
for i in range(5):
    print(omega[i]/(2*pi))


n_fig = 5
plot_eigenvector = 'y'

if plot_eigenvector == 'y':

    for i in range(n_fig):

        eigvec_i = V_red @ eigvec_omega[:nf_red, i]
        eigvec_w = eigvec_i[:n_pw]
        eigvec_w_real = np.real(eigvec_w)
        eigvec_w_imag = np.imag(eigvec_w)

        z_real = eigvec_w_real
        z_imag = eigvec_w_imag

        tol = 1e-6
        fntsize = 20

        zreal_norm = np.linalg.norm(z_real)
        zimag_norm = np.linalg.norm(z_imag)

        if zreal_norm > zimag_norm:
            z1 = z_real
        else:
            z1 = z_imag

        minZ1 = min(z1)
        maxZ1 = max(z1)

        if minZ1 != maxZ1:

            fig1 = plt.figure()

            ax1 = fig1.add_subplot(111, projection='3d')
            # ax1.zaxis._axinfo['label']['space_factor'] = 20

            ax1.set_xbound(min(x) - tol, max(x) + tol)
            ax1.set_xlabel('$x$', fontsize=fntsize)

            ax1.set_ybound(min(y) - tol, max(y) + tol)
            ax1.set_ylabel('$y$', fontsize=fntsize)

            ax1.set_title('$v_{e_{p,w}}$', fontsize=fntsize)

            ax1.set_zlim3d(minZ1 - 0.01*abs(minZ1), maxZ1 + 0.01*abs(maxZ1))
            ax1.w_zaxis.set_major_locator(LinearLocator(10))
            ax1.w_zaxis.set_major_formatter(FormatStrFormatter('%.04f'))

            ax1.plot_trisurf(x, y, z1, cmap=cm.jet, linewidth=0, antialiased=False)

            # path_out1 = "/home/a.brugnoli/PycharmProjects/Mindlin_Phs_fenics/Figures_Eig_Min/RealEig/"
            # plt.savefig(path_out1 + "Case" + case_study + "_el" + str(n) + "_deg" + str(deg) + "_thick_" + \
            #             str(thick) + "_eig_" + str(i+1) + ".eps", format="eps")

plt.show()
