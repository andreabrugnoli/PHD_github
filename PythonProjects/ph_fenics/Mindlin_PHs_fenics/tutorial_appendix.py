from fenics import *
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import vstack, hstack, block_diag
from scipy.sparse.linalg import eigs

n_el  = 40
mesh  = UnitSquareMesh(n_el, n_el)
P_w   = FiniteElement('CG', triangle, 1)   	# vertical velocity
P_th  = VectorElement('CG', triangle, 1)   	# angular velocity
P_kap = VectorElement('DG', triangle, 0, dim=3)	# bending momenta			
P_gam = VectorElement('DG', triangle, 0)	# shear stress 
elem  = MixedElement([P_w, P_th, P_kap, P_gam])
V     = FunctionSpace(mesh, elem)

# Physical parameters 
E   = 1e12
nu  = 0.3	
rho = 2600
h   = 0.1
k   = 5/6	
G   = E / 2 / (1 + nu)
F   = G * h * k

# Definition of the bending curvature operator
def bending_curv(mom):
	kappa = 12. / (E * h ** 3) * ((1+nu)*mom - nu * Identity(2) * tr(mom))
	return kappa
	
# Test variables
v = TestFunction(V)
v_w, v_th, v_kap, v_gam = split(v)	

# Co-energy variables
e = TrialFunction(V)
e_w, e_th, e_kap, e_gam = split(e)

# Convert the R^3 vector to a symmetric tensor
v_kap = as_tensor([[v_kap[0], v_kap[1]],
				   [v_kap[1], v_kap[2]]])
e_kap = as_tensor([[e_kap[0], e_kap[1]],
				   [e_kap[1], e_kap[2]]])
			
# Energy variables   
a_w   = rho * h * e_w
a_th  = (rho * h ** 3) / 12. * e_th
a_kap = bending_curv(e_kap)
a_gam = 1. / F * e_gam

# Mass bilinear form 
m_form = v_w * a_w * dx + dot(v_th, a_th) * dx + \
		 inner(v_kap, a_kap) * dx + dot(v_gam, a_gam) * dx 

# Interconnection bilinear form
j_form = dot(v_gam, grad(e_w)) * dx - dot(grad(v_w), e_gam) * dx + \
		 inner(v_kap, sym(grad(e_th))) * dx - \
		 inner(sym(grad(v_th)), e_kap) * dx + \
		 dot(v_th, e_gam) * dx - dot(v_gam, e_th) * dx


# Boundary conditions
bcs = []
#bcs.append(DirichletBC(V.sub(0), Constant(0.0), "on_boundary"))
#bcs.append(DirichletBC(V.sub(1), Constant((0.0, 0.0)), "on_boundary"))

#J, M = PETScMatrix(), PETScMatrix()
#dummy = v_w * dx
#assemble_system(j_form, dummy, bcs, A_tensor=J)
#assemble_system(m_form, dummy, bcs, A_tensor=M)
#
#solver = SLEPcEigenSolver(J, M)
#solver.parameters["solver"] = "krylov-schur"
## Set the problem type: the J matrix is not hermitian nor positive.
#solver.parameters["problem_type"] = "pos_gen_non_hermitian"
## We look for eigenvalues on the imaginary axis.
#solver.parameters["spectrum"] = "target imaginary"
#solver.parameters["spectral_transform"] = "shift-and-invert"
#solver.parameters["spectral_shift"] = 1/((2*(1+nu)*rho)/E)**0.5
#solver.solve(40)
#n_conv = solver.get_number_converged()
#
#omega_tilde = []
#for i in range(n_conv):
#	lam_r, lam_i, psi_r, psi_i = solver.get_eigenpair(i)
#	
#	# Discard the zero eigenvalues due to the bcs.
#	if lam_i > 1e-5:
#		omega_tilde.append(lam_i *((2 * (1 + nu) * rho) / E) ** 0.5)

#print(omega_tilde)
        
P_qn  = FiniteElement('CG', triangle, 1)   # shear force
P_Mnn = FiniteElement('CG', triangle, 1)   # flexural momentum
P_Mns = FiniteElement('CG', triangle, 1)   # torsional momentum
elem  = MixedElement([P_qn, P_Mnn, P_Mns])
 
V_u   = FunctionSpace(mesh, elem)
q_n, M_nn, M_ns = TrialFunction(V_u)

n_ver = FacetNormal(mesh)  
s_ver = as_vector([-n_ver[1], n_ver[0]])

b_form = v_w * q_n * ds + dot(v_th, n_ver) * M_nn * ds + \
		 dot(v_th, s_ver) * M_ns * ds	 
B_petsc = PETScMatrix()
assemble(b_form, B_petsc)
B = B_petsc.mat()

# Conversion to CSR scipy format
B_scipy_cols = csr_matrix(B.getValuesCSR()[::-1])

# Non zero rows and columns
rows, cols = csr_matrix.nonzero(B_scipy_cols)

# Indexes of non zero columns
set_cols = np.array(list(set(cols)))

# Number of non zero columns (i.e. input number)
n_u = len(set(cols))

# Initialization of the final matrix in lil folmat
# for efficient incremental construction.
B_scipy = lil_matrix((V.dim(), n_u))

for r, c in zip(rows, cols):
	# Column index in the final matrix
	ind_col = np.where(set_cols == c)[0]
	
	# Fill the matrix with the values
	B_scipy[r, ind_col] = B_scipy_cols[r, c]
	
# Convert to csr format
B_scipy.tocsr()	

# Conversion to scipy CSR matrices 
# Important: no boundary conditions imposed
J_scipy = csr_matrix(J.mat().getValuesCSR()[::-1])  # for fenics J.mat() 
M_scipy = csr_matrix(M.mat().getValuesCSR()[::-1])  # for fenics M.mat() 

Z_mat = csr_matrix((n_u, n_u))
J_aug = vstack((hstack((J_scipy, B_scipy)), hstack((-B_scipy.T, Z_mat))))
M_aug = block_diag((M_scipy, Z_mat))

# Shift value
shift = 1/(((2*(1+nu)*rho)/E)**0.5)
eigenvalues, eigvectors = eigs(J_aug, k=40, M=M_aug,\
	 sigma=shift, which='LM', tol=1e-6, maxiter=5000)

omega_all = np.imag(eigenvalues)
index = omega_all >= 1e-5
omega = omega_all[index]
omega.sort()

omega_tilde = omega*((2*(1+nu)*rho)/E)**0.5