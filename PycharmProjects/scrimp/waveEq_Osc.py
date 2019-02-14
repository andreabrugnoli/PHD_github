# -*- coding: utf-8 -*-

from fenics import *
import numpy as np
import scipy.linalg as la
    
def MatricesString(n, deg, T = 1, rho =1, L=1):
    
    
    mesh = IntervalMesh(n, 0, L)
    
    
    d = mesh.geometry().dim()
     
    class Left(SubDomain):
        def inside(self, x, on_boundary):
            return abs(x[0] - 0.0) < DOLFIN_EPS and on_boundary
    
    class Right(SubDomain):
        def inside(self, x, on_boundary):
            return abs(x[0] - L) < DOLFIN_EPS and on_boundary
    
    # Boundary conditions on rotations
    left = Left()
    right = Right()
    
    
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    left.mark(boundaries, 1)
    right.mark(boundaries, 2)
    
    
    dx = Measure('dx')
    ds = Measure('ds', subdomain_data= boundaries)
    
    # Finite element defition
    
    Pp = FiniteElement('CG', mesh.ufl_cell(), deg)
    Pq = FiniteElement('CG', mesh.ufl_cell(), deg)
    
    V = FunctionSpace(mesh, MixedElement([Pp, Pq]))
    Vp = V.sub(0)
    Vq = V.sub(1)
    
    n_Vp = Vp.dim()
    n_Vq = Vq.dim()
    n_V = V.dim()
    
    dofV_x = V.tabulate_dof_coordinates().reshape((-1, d))
    
      
    #    vertex_x = mesh.coordinates().reshape((-1, d))
    
    dofs_Vp = Vp.dofmap().dofs()
    dofs_Vq = Vq.dofmap().dofs()
    
    dofVp_x = dofV_x[dofs_Vp]
    dofVq_x = dofV_x[dofs_Vq]
    
    v_p, v_q = TestFunction(V)
    
    e_p, e_q = TrialFunction(V)
    
    al_p = rho * e_p
    al_q = 1./T * e_q
    
    
    m_p = inner(v_p, al_p) * dx
    m_q = inner(v_q, al_q) * dx
    
    #    j_div = v_p * e_q.dx(0) * dx # v_p * div(e_q) * dx
    #    j_divIP = -v_q.dx(0) * e_p *dx # -div(v_q) * e_p * dx
    
    j_grad = v_q * e_p.dx(0) * dx # dot(v_q, grad(e_p)) * dx
    j_gradIP = -v_p.dx(0) * e_q *dx # dot(-grad(v_p), e_q) * dx
    
    m = m_p + m_q
    j_allgrad = j_grad + j_gradIP
       
    
    # Assemble the interconnection matrix and the mass matrix.
    J, M = PETScMatrix(), PETScMatrix()
    
    J = assemble(j_allgrad).array()   
    M = assemble(m).array()
    
    G_l, G_r = PETScVector(), PETScVector()
    
    G_l = assemble(-v_p * ds(1)).get_local()
    G_r = assemble(+v_p * ds(2)).get_local()
    
    G = np.array([G_l, G_r]).T

    B_e = np.zeros((n_V,))
    B_lambda = np.array([0, 1])
    
    B = np.concatenate((B_e, B_lambda), axis = 0).reshape((-1, 1))
    
    Mall = la.block_diag(M, np.zeros((2,2)) )
    
    Jall = np.zeros((n_V + 2, n_V + 2))
    Jall[:n_V, :n_V] = J
    Jall[:n_V, n_V:] = +G
    Jall[n_V:, :n_V] = -G.T
    
    return Mall, Jall, B


Mst, Jst, Bst = MatricesString(1, 1)
n_st, n_st = Mst.shape

Qst = np.eye(n_st)
Rst = np.zeros((n_st, n_st))
# Linear Oscillator    

m_rod = 1
k_rod = 1

Mrod = np.array([[m_rod, 0],[0, 1./k_rod]])
Jrod = np.array([[0, -1],[1, 0]])
Brod = np.array([1, 0]).reshape((-1, 1))

n_rod = 2

Qrod = np.eye(n_rod)
Rrod = np.zeros((n_rod, n_rod))


Mlambda = np.zeros((2,2))
M = la.block_diag(Mst, Mrod)


J = la.block_diag(Jst, Jrod)
J[n_st, n_st - 1] = -1
J[n_st - 1, n_st] = +1

from phdae_system_manager import System_DAE
from phdae_subsystem_manager import Subsystem_DAE
# ns: dimension of the state space of the linear oscillator
# Here ns is equal to 2d since we consider a d-dimensional linear oscillator

# nr: dimension of the resistive space
nr = 0

# nc: dimension of the input/control space
nc = 1

# ni: dimension of the interconnection space 
ni = 0

# nm: dimension of the constraint space (unconstrained case so nm = 0)
nm = 2

Sub_L1 = Subsystem_DAE(n_st,nr,nc,ni)

Sub_L1.Set_E_matrix(Mst)

Sub_L1.Set_J_matrix(Jst)
    
Sub_L1.Set_B_matrix(Bst)

Sub_L1.Set_R_matrix(Rst)
    
Sub_L1.Set_D_matrix( np.array([0]).reshape((-1, 1)) )
 
Sub_L1.Set_Q_matrix(Qst)    
 
Sub_L1.Set_S_matrix()


#
# Second subsystem related to Lo_2
#

Sub_L2 = Subsystem_DAE(n_rod,nr,nc,ni)

Sub_L2.Set_E_matrix(Mrod)

Sub_L2.Set_J_matrix(Jrod)
    
Sub_L2.Set_B_matrix(Brod)

Sub_L2.Set_R_matrix(Rrod)
    
Sub_L2.Set_D_matrix( np.array([0]).reshape((-1, 1)) )
 
Sub_L2.Set_Q_matrix(Qrod)    
 
Sub_L2.Set_S_matrix()

#
# We check that the two pHDAEs have been built successfully
#

print("The Sub_L1 object has been created successfully: ", bool(Sub_L1.istatus_S))

print("The Sub_L2 object has been created successfully: ", bool(Sub_L2.istatus_S))

#
# We are now ready to interconnect the two pHDAEs.
#

#
# Create a Python list of subsystems defining the global system
#

Subsystem_List = []
Subsystem_List.append(Sub_L1)
Subsystem_List.append(Sub_L2)

#
# Define the coupling matrix related to the two subsystems
#

Coupling_Lo = -1

#
# Specify that the global port-Hamiltonian system named pHs_Lo is
# made of the two pHs Sub_L1 and Sub_L2
#

pHs_Lo = System_DAE(Subsystem_List,Coupling_Lo,connection_type="Gyrator")

assert (pHs_Lo.E).all == M.all
# assert (pHs_Lo.J).all == J.all

assert Sub_L1.E.all == Mst.all
assert Sub_L2.E.all == Mrod.all

assert Sub_L1.J.all == Jst.all
assert Sub_L2.J.all == Jrod.all

    
    
    
    
    
    
    