# Mindlin plate written with the port Hamiltonian approach
# with strong symmetry
from firedrake import *
import numpy as np
np.set_printoptions(threshold=np.inf)
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from firedrake.plot import _two_dimension_triangle_func_val
from mpl_toolkits.mplot3d import Axes3D
plt.rc('text', usetex=True)

from firedrake.petsc import PETSc
try:
    from slepc4py import SLEPc
except ImportError:
    import sys
    warning("Unable to import SLEPc, eigenvalue computation not possible (try firedrake-update --slepc)")
    sys.exit(0)


n_el = 10
deg = 2

rho = 2700
E = 1e12
nu = 0.3

L = 1

thick = 'n'
if thick == 'y':
    h = 0.1
else:
    h = 0.01
bc_input = input('Select Boundary Condition: ')

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
fl_rot = 12 / (E * h ** 3)

# The unit square mesh is divided in :math:`N\times N` quadrilaterals::

L = 1

n_x, n_y = n_el, n_el
L_x, L_y = L, L
# mesh = RectangleMesh(n_x, n_y, L_x, L_y, quadrilateral=True)

mesh_int = IntervalMesh(n_el, L)
mesh = ExtrudedMesh(mesh_int, n_el)

# domain = mshr.Rectangle(Point(0, 0), Point(l_x, l_y))
# mesh = mshr.generate_mesh(domain, n, "cgal")

# plot(mesh)
# plt.show()


# Operators and functions
def gradSym(u):
    return 0.5 * (nabla_grad(u) + nabla_grad(u).T)
    # return sym(nabla_grad(u))

def bending_moment(kappa):
    momenta = D * ((1-nu) * kappa + nu * Identity(2) * tr(kappa))
    return momenta

def bending_curv(momenta):
    kappa = fl_rot * ((1+nu)*momenta - nu * Identity(2) * tr(momenta))
    return kappa
#
# def strain2voigt(eps):
#     return as_vector([eps[0, 0], eps[1, 1], 2 * eps[0, 1]])
#
# def voigt2stress(S):
#     return as_tensor([[S[0], S[2]], [S[2], S[1]]])
#
# def bending_moment(u):
#     return voigt2stress(dot(D_b, strain2voigt(u)))
#
# def bending_curv(u):
#     return voigt2stress(dot(C_b, strain2voigt(u)))


# Finite element defition

CG_deg1 = FiniteElement("CG", interval, deg)
DG_deg = FiniteElement("DG", interval, deg-1)
DG_deg1 = FiniteElement("DG", interval, deg)

P_CG1_DG = TensorProductElement(CG_deg1, DG_deg)
P_DG_CG1 = TensorProductElement(DG_deg, CG_deg1)

RT_horiz = HDivElement(P_CG1_DG)
RT_vert = HDivElement(P_DG_CG1)
RT_quad = RT_horiz + RT_vert

P_CG1_DG1 = TensorProductElement(CG_deg1, DG_deg1)
P_DG1_CG1 = TensorProductElement(DG_deg1, CG_deg1)

BDM_horiz = HDivElement(P_CG1_DG1)
BDM_vert = HDivElement(P_DG1_CG1)
BDM_quad = BDM_horiz + BDM_vert

V_pw = FunctionSpace(mesh, "DG", deg-1)
V_pth = VectorFunctionSpace(mesh, "DG", deg-1)
V_qthD = FunctionSpace(mesh, BDM_quad)
V_qth12 = FunctionSpace(mesh, "CG", deg)
V_qw = FunctionSpace(mesh, BDM_quad)

V = MixedFunctionSpace([V_pw, V_pth, V_qthD, V_qth12, V_qw])

n_V = V.dim()
print(n_V)

v = TestFunction(V)
v_pw, v_pth, v_qthD, v_qth12, v_qw = split(v)

e = TrialFunction(V)
e_pw, e_pth, e_qthD, e_qth12, e_qw = split(e)

v_qth = as_tensor([[v_qthD[0], v_qth12],
                    [v_qth12, v_qthD[1]]
                   ])

e_qth = as_tensor([[e_qthD[0], e_qth12],
                    [e_qth12, e_qthD[1]]
                   ])

al_pw = rho * h * e_pw
al_pth = (rho * h ** 3) / 12. * e_pth
al_qth = bending_curv(e_qth)
al_qw = 1. / F * e_qw


# v_skw = skew(v_skw)
# al_skw = skew(e_skw)

dx = Measure('dx')
ds = Measure('ds')

m_form = v_pw * al_pw * dx \
    + dot(v_pth, al_pth) * dx \
    + inner(v_qth, al_qth) * dx \
    + dot(v_qw, al_qw) * dx

j_div = v_pw * div(e_qw) * dx
j_divIP = -div(v_qw) * e_pw * dx

j_divSym = dot(v_pth, div(e_qth)) * dx
j_divSymIP = -dot(div(v_qth), e_pth) * dx

# j_grad = dot(v_qw, grad(e_pw)) * dx
# j_gradIP = -dot(grad(v_pw), e_qw) * dx
#
# j_gradSym = inner(v_qth, gradSym(e_pth)) * dx
# j_gradSymIP = -inner(gradSym(v_pth), e_qth) * dx

j_Id = dot(v_pth, e_qw) * dx
j_IdIP = -dot(v_qw, e_pth) * dx

j_alldiv = j_div + j_divIP + j_divSym + j_divSymIP + j_Id + j_IdIP
# j_allgrad = j_grad + j_gradIP + j_gradSym + j_gradSymIP + j_Id + j_IdIP

j_form = j_alldiv

# The boundary edges in this mesh are numbered as follows:

# 1: plane x == 0
# 2: plane x == 1
# 3: plane y == 0
# 4: plane y == 1

bc_1, bc_2, bc_3, bc_4 = bc_input

bc_dict = {1: bc_1, 3: bc_2, 2: bc_3, 4: bc_4}

bcs = []
for key,val in bc_dict.items():
    if val == 'F':
        if key == 1 or key == 2:
            bcs.append(DirichletBC(V.sub(2), Constant((0.0, 0.0)), key))
            bcs.append(DirichletBC(V.sub(3), Constant(0.0), key))
            bcs.append(DirichletBC(V.sub(4), Constant((0.0, 0.0)), key))
        elif key == 3:
            bcs.append(DirichletBC(V.sub(2), Constant((0.0, 0.0)), "bottom"))
            bcs.append(DirichletBC(V.sub(3), Constant(0.0), "bottom"))
            bcs.append(DirichletBC(V.sub(4), Constant((0.0, 0.0)), "bottom"))
        else:
            bcs.append(DirichletBC(V.sub(2), Constant((0.0, 0.0)), "top"))
            bcs.append(DirichletBC(V.sub(3), Constant(0.0), "top"))
            bcs.append(DirichletBC(V.sub(4), Constant((0.0, 0.0)), "top"))
    elif val == 'S':
        if key == 1 or key == 2:
            bcs.append(DirichletBC(V.sub(2), Constant((0.0, 0.0)), key))
        elif key == 3:
            bcs.append(DirichletBC(V.sub(2), Constant((0.0, 0.0)), "bottom"))
        else:
            bcs.append(DirichletBC(V.sub(2), Constant((0.0, 0.0)), "top"))


M = assemble(m_form, bcs=bcs, mat_type='aij')
J = assemble(j_form, bcs=bcs, mat_type='aij')

petsc_j = J.M.handle
petsc_m = M.M.handle

tol = 10**(-9)

# plt.spy(J_aug); plt.show()

num_eigenvalues = 10

target = 1/(L*((2*(1+nu)*rho)/E)**0.5)

opts = PETSc.Options()
opts.setValue("pos_gen_non_hermitian", None)
opts.setValue("st_pc_factor_shift_type", "NONZERO")
opts.setValue("eps_type", "krylovschur")
opts.setValue("eps_tol", 1e-10)
opts.setValue("st_type", "sinvert")
# opts.setValue("eps_target_imaginary", None)
opts.setValue("st_shift", target)
opts.setValue("eps_target", target)


es = SLEPc.EPS().create(comm=COMM_WORLD)
# st = es.getST()
# st.setShift(0)
# st.setType("sinvert")
# es.setST(st)
# es.setWhichEigenpairs(1)
es.setDimensions(num_eigenvalues)
es.setOperators(petsc_j, petsc_m)
es.setFromOptions()
es.solve()

nconv = es.getConverged()

if nconv == 0:
    import sys
    warning("Did not converge any eigenvalues")
    sys.exit(0)

vr, vi = petsc_j.getVecs()

tol = 1e-6
lamda_vec = np.zeros((nconv))

n_stocked_eig = 0

omega_tilde = []
eig_real_w_vec = []
eig_imag_w_vec = []

eig_real_w = Function(V_pw)
eig_imag_w = Function(V_pw)
fntsize = 15

n_pw = V_pw.dim()
for i in range(nconv):
    lam = es.getEigenpair(i, vr, vi)


    lam_r = np.real(lam)
    lam_c = np.imag(lam)

    if lam_c > tol:

        print(n_stocked_eig)

        omega_tilde.append(lam_c*L*((2*(1+nu)*rho)/E)**0.5)

        vr_w = vr.getArray().copy()[:n_pw]
        vi_w = vi.getArray().copy()[:n_pw]

        eig_real_w_vec.append(vr_w)
        eig_imag_w_vec.append(vi_w)

        n_stocked_eig += 1

for i in range(n_stocked_eig):
    print("Eigenvalue num " + str(i + 1) + ":" + str(omega_tilde[i]))

n_fig = min(n_stocked_eig, 4)

# for i in range(n_fig):
#     norm_real_eig = np.linalg.norm(eig_real_w_vec[i])
#     norm_imag_eig = np.linalg.norm(eig_imag_w_vec[i])
#
#     eig_real_w.vector()[:] = eig_real_w_vec[i]
#     eig_imag_w.vector()[:] = eig_imag_w_vec[i]
#
#     figure = plt.figure()
#     ax = figure.add_subplot(111, projection="3d")
#
#     ax.set_xlabel('$x [m]$', fontsize=fntsize)
#     ax.set_ylabel('$y [m]$', fontsize=fntsize)
#     ax.set_title('Eigenvector num ' + str(i + 1), fontsize=fntsize)
#
#     ax.w_zaxis.set_major_locator(LinearLocator(10))
#     ax.w_zaxis.set_major_formatter(FormatStrFormatter('%1.2g'))
#
#     if norm_imag_eig > norm_real_eig:
#         plot(eig_imag_w, axes=ax, plot3d=True)
#     else:
#         plot(eig_real_w, axes=ax, plot3d=True)
#
# plt.show()