# Mindlin plate written with the port Hamiltonian approach
# with weak symmetry

from fenics import *
import numpy as np
np.set_printoptions(threshold=np.inf)
import mshr
import matplotlib.pyplot as plt

import scipy.linalg as la

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

matplotlib.rcParams['text.usetex'] = True

n = 20 #int(input("Number of elements for side: "))
deg = 2 #int(input('Degree for FE: '))
nreq = 10

E = 1e12
nu = 0.3

rho = 2600

L = 1

thick = 'y'
if thick == 'y':
    h = 0.1
else:
    h = 0.01
plot_eigenvector = 'y'

bc_input = input('Select Boundary Condition: ')
#bc_input = 'CCCC'

if bc_input == 'CCCC' or  bc_input == 'CCCF':
    k = 0.8601 # 5./6. #
elif bc_input == 'SSSS':
    k = 0.8333
elif bc_input == 'SCSC':
    k = 0.822
else: k = 0.8601

G = E / 2 / (1 + nu)
F = G * h * k


# Useful Matrices
D = E * h ** 3 / (1 - nu ** 2) / 12.
fl_rot = 12. / (E * h ** 3)

# The unit square mesh is divided in :math:`N\times N` quadrilaterals::

L = 1
l_x = L
l_y = L

n_x, n_y = n, n
mesh = RectangleMesh(Point(0, 0), Point(L, L), n_x, n_y, "right/left")
d = mesh.geometry().dim()


# domain = mshr.Rectangle(Point(0, 0), Point(l_x, l_y))
# mesh = mshr.generate_mesh(domain, n, "cgal")

# plot(mesh)
# plt.show()

# Operators and functions
def gradSym(u):
    return 0.5 * (nabla_grad(u) + nabla_grad(u).T)
    # return sym(nabla_grad(u))

def bending_moment(kappa):
    momenta = D * ((1-nu) * kappa + nu * Identity(d) * tr(kappa))
    return momenta

def bending_curv(momenta):
    kappa = fl_rot * ((1+nu)*momenta - nu * Identity(d) * tr(momenta))
    return kappa

# Domain, Subdomains, Boundary, Suboundaries
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0] - 0.0) < DOLFIN_EPS and on_boundary

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0] - l_x) < DOLFIN_EPS and on_boundary

class Lower(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[1] - 0.0) < DOLFIN_EPS and on_boundary

class Upper(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[1] - l_y) < DOLFIN_EPS and on_boundary

# Boundary conditions on rotations
left = Left()
right = Right()
lower = Lower()
upper = Upper()

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(5)
left.mark(boundaries, 1)
lower.mark(boundaries, 2)
right.mark(boundaries, 3)
upper.mark(boundaries, 4)


# Finite element defition

P_pw = FiniteElement('CG', triangle, deg)
P_pth = VectorElement('CG', triangle, deg)
P_qth = VectorElement('DG', triangle, deg-1, dim=3)
P_qw = VectorElement('DG', triangle, deg-1)

element = MixedElement([P_pw, P_pth, P_qth, P_qw])
V = FunctionSpace(mesh, element)

n_V = V.dim()
print(n_V)

v = TestFunction(V)
v_pw, v_pth, v_qth, v_qw = split(v)

e = TrialFunction(V)
e_pw, e_pth, e_qth, e_qw = split(e)

v_qth = as_tensor([[v_qth[0], v_qth[1]],
                    [v_qth[1], v_qth[2]]
                   ])

e_qth = as_tensor([[e_qth[0], e_qth[1]],
                    [e_qth[1], e_qth[2]]
                   ])

al_pw = rho * h * e_pw
al_pth = (rho * h ** 3) / 12. * e_pth
al_qth = bending_curv(e_qth)
al_qw = 1. / F * e_qw

# v_skw = skew(v_skw)
# al_skw = skew(e_skw)


dx = Measure('dx')
ds = Measure('ds', subdomain_data=boundaries)

m_form = v_pw * al_pw * dx \
    + dot(v_pth, al_pth) * dx \
    + inner(v_qth, al_qth) * dx \
    + dot(v_qw, al_qw) * dx 

j_grad = dot(v_qw, grad(e_pw)) * dx
j_gradIP = -dot(grad(v_pw), e_qw) * dx

j_gradSym = inner(v_qth, gradSym(e_pth)) * dx
j_gradSymIP = -inner(gradSym(v_pth), e_qth) * dx

j_Id = dot(v_pth, e_qw) * dx
j_IdIP = -dot(v_qw, e_pth) * dx

j_allgrad = j_grad + j_gradIP + j_gradSym + j_gradSymIP + j_Id + j_IdIP

j_form = j_allgrad

bc_1, bc_2, bc_3, bc_4 = bc_input

bc_dict = {left: bc_1, lower: bc_2, right: bc_3, upper: bc_4}

bcs = []
for key,val in bc_dict.items():
    if val == 'C':
        bcs.append(DirichletBC(V.sub(0), Constant(0.0), key))
        bcs.append(DirichletBC(V.sub(1), Constant((0.0, 0.0)), key))

    if val == 'S':
        bcs.append(DirichletBC(V.sub(0), Constant(0.0), key))  
        
        if key == left or key==right:
            bcs.append(DirichletBC(V.sub(1).sub(1), Constant(0.0), key))
            
        if key == upper or key==lower:
            bcs.append(DirichletBC(V.sub(1).sub(0), Constant(0.0), key))
        
J = PETScMatrix()
M = PETScMatrix()

dummy = v_pw*dx
assemble_system(j_form, dummy, bcs, A_tensor=J)
assemble_system(m_form, dummy, bcs, A_tensor=M)

#[bc.zero(M) for bc in bcs]
       
shift = 1/(L*((2*(1+nu)*rho)/E)**0.5)


solver = SLEPcEigenSolver(J, M)
solver.parameters["solver"] = "krylov-schur"
solver.parameters["problem_type"] = "pos_gen_non_hermitian"

solver.parameters["spectrum"] = "target imaginary"
solver.parameters["spectral_transform"] = "shift-and-invert"
solver.parameters["spectral_shift"] = shift
neigs = 15
solver.solve(neigs)

nconv = solver.get_number_converged()
if nconv==0:
    print('no eig converged')
else: print(str(nconv) + ' eigenvalues converged')

computed_real_eigenvalues = []
computed_imag_eigenvalues = []

dofV_x = V.tabulate_dof_coordinates().reshape((-1, d))

dofs_Vpw = V.sub(0).dofmap().dofs()
dofs_Vpth = V.sub(1).dofmap().dofs()
dofs_Vqth = V.sub(2).dofmap().dofs()
dofs_Vqw = V.sub(3).dofmap().dofs()

dofVpw_x = dofV_x[dofs_Vpw]

x = dofVpw_x[:, 0]
y = dofVpw_x[:, 1]

n_pw = len(dofs_Vpw)

eigvec_w_real = []
eigvec_w_imag = []
n_eig_stocked = 0

tol = 1e-9

for i in range(nconv):
    r, c, rx, cx = solver.get_eigenpair(i) # ignore the imaginary part
    
#    print(i , r, c)
    
    if c>tol:
        computed_real_eigenvalues.append(r)
        computed_imag_eigenvalues.append(c)
    
        eigvec_w_real.append(rx[dofs_Vpw])
        eigvec_w_imag.append(cx[dofs_Vpw])
    
        n_eig_stocked += 1
    
#   eigenmode = Function(Vpw,name="Eigenvector "+str(i))
#    eigenmode.vector()[:] = rx

    
omega_til_imag = np.array(computed_imag_eigenvalues)*L*((2*(1+nu)*rho)/E)**0.5
omega_til_real = np.array(computed_real_eigenvalues)*L*((2*(1+nu)*rho)/E)**0.5

print('Imaginary part tilde')
for i in range(n_eig_stocked):
    print('Eig n ' + str(i) + ': ' + str(np.sort(np.array(omega_til_imag))[i]))

if plot_eigenvector == 'y':

    for i in range(min(4, n_eig_stocked)):
        z_real = eigvec_w_real[i]
        z_imag = eigvec_w_imag[i]

        tol = 1e-6
        fntsize = 20

        if np.linalg.norm(z_real) > np.linalg.norm(z_imag):
            z = z_real
        else:
            z = z_imag

        minZ = min(z)
        maxZ = max(z)

        if minZ != maxZ:

            fig = plt.figure()

            ax = fig.add_subplot(111, projection='3d')

            ax.set_xbound(min(x) - tol, max(x) + tol)
            ax.set_xlabel('$x$', fontsize=fntsize)

            ax.set_ybound(min(y) - tol, max(y) + tol)
            ax.set_ylabel('$y$', fontsize=fntsize)

            ax.set_title('$v_{e_{p,w}}$', fontsize=fntsize)

            ax.set_zlim3d(minZ - 0.01*abs(minZ), maxZ + 0.01*abs(maxZ))
            ax.w_zaxis.set_major_locator(LinearLocator(10))
            ax.w_zaxis.set_major_formatter(FormatStrFormatter('%.2g'))

            ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0, antialiased=False)

            # path_out = "/home/a.brugnoli/PycharmProjects/Mindlin_Phs_fenics/Figures_Eig_Min/RealEig/"
            # plt.savefig(path_out1 + "Case" + case_study + "_el" + str(n) + "_deg" + str(deg) + "_thick_" + \
            #             str(thick) + "_eig_" + str(i+1) + ".eps", format="eps")

plt.show()

