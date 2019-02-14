# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 14:33:34 2017

@author: vasseur
"""
## @package 
#  Documentation for this module.
#
#  

## Documentation for a class.
#
#  Class to handle a subsystem of port-Hamiltonian descriptor system type (phDAE).
#
#  A standard port-Hamiltonian system is defined as
#  \f$\Sigma = (\Xi, H, \cal R, \cal C, \cal I, \cal D) \f$
#
#  \f$\Xi \f$: state space manifold
#  \f$\cal H \f$: Hamiltonian:  \f$\Xi \rightarrow \mathcal{R} \f$ corresponding to the energy storage port \f$\cal S \f$
#  \f$\cal R \f$: resistive port 
#  \f$\cal C \f$: control port
#  \f$\cal I \f$: interconnection port
#  \f$\cal D \f$: total Dirac structure 
#
#  See 
# 
#  van der Schaft A.J. (2013) Port-Hamiltonian Differential-Algebraic Systems. 
#  In: Ilchmann A., Reis T. (eds) Surveys in Differential-Algebraic Equations I. Differential-Algebraic Equations 
#  Forum. Springer, Berlin, Heidelberg
#
#  and 
#  
#  A.J. van der Schaft, "Port-Hamiltonian systems: an introductory survey", 
#  Proceedings of the International Congress of Mathematicians, Volume III, Invited Lectures, 
#  eds. Marta Sanz-Sole, Javier Soria, Juan Luis Verona, Joan Verdura, Madrid, Spain, pp. 1339-1365, 2006.
#
#  A port-Hamiltonian descriptor system (phDAE) is defined in 
#
#  [BMXZ17] C. Beattie, V. Mehrmann, H. Xu and H. Zwart, "Port-Hamiltonian descriptor systems", 
#       arXiv:1705.09081v2
#
#  In the linear case the subsytem may be written as (Def. 5 of [BMXZ17])  :
#
#                              E xdot = [(J-R)Q - E K] x + B u   + s_sys
#                                   y =        B^TQ x    + D u   + s_o
# 
# Note that E may be singular depending on the application and that EK is introduced to 
#      accommodate a time-varying change of basis. 
# Note that the P and N contributions in (Def. 5 of [BMXZ17]) have been discarded here.
# Consequently D corresponds to the S matrix in Def. 5 of [BMXZ17] and is supposed to be 
# symmetric.
#
#
# NB: We first consider the linear case ! (see Sections 2 and 3 of [BMXZ17]).
#
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class Subsystem_DAE(object):
    """ Define the general settings for a subsystem of Port-Hamiltonian system type with differential algebraic equations (DAE)"""

    ## The constructor of the Subsystem_DAE class.
    #  @param self The object pointer.
    #  @param  ns  Dimension of the state space manifold with port variables (fs, es)
    #  @param  nr  Dimension of the resistive space with port variables (fr, er)
    #  @param  ni  Dimension of the interconnection space with port variables (fi, ei)
    #  @param  nc  Dimension of the input/output control space with port variables (fc, ec)
    #  @param  mode Short description related to the Port-Hamiltonian DAE system  [by default "User defined"]

    def __init__(self,ns,nr,nc,ni,mode="User defined"):
        """
        Constructor for the Subsystem_DAE class.
        """
        ## @var ns
        #  dimension of the state space [integer]
        #  @var nr
        #  dimension of the resistive space [integer]
        #  @var nc
        #  dimension of the input/control space [integer]
        #  @var ni   
        #  dimension of the interconnection space [integer]
        #  @var nm   
        #  dimension of the constraint space [integer]
        #  @var mode
        #  description related to the subsystem [string]


        # Include the definition of a default automatic subsystem mainly for checking purpose 
        if mode != "User defined":
            self.mode = "Automatic" 
        else: 
            self.mode = mode
            
        assert ns >= 0, "The dimension of the state space (energy storage space) should be positive"       
        assert nr >= 0, "The dimension of the resistive space should be positive"       
        assert nc >= 0, "The dimension of the input/control space should be positive"       
        assert ni >= 0, "The dimension of the interconnection port space should be positive"    
        
        self.ns  = ns
        self.nr  = nr
        self.nc  = nc
        self.ni  = ni
        self.ncc = nr+nc+ni
        self.nt  = ns+nr+nc+ni
        
        ## @var J
        #  Structure matrix [Numpy array]
        #  @var Q
        #  Quadratic part of the Hamiltonian if applicable [Numpy array]
        #  @var R
        #  Dissipation matrix [Numpy array]
        #  @var S
        #  Global matrix of the system  [Numpy array]
        #  @var D
        #  Matrix related to the feedthrough part (if a linear model is applicable) [Numpy array]
       
        self.J  = np.zeros(shape=(self.ns,self.ns)) 
        self.Q  = np.eye(self.ns)
        self.R  = np.zeros(shape=(self.ns,self.ns))
        self.A  = np.zeros(shape=(self.ns,self.ns))
        
        
        self.S  = np.zeros(shape=(self.nt,self.nt))
        self.W  = np.zeros(shape=(self.nt,self.nt))
        
        # NB: Zero as a shape dimension is allowed in Numpy
        self.D  = np.zeros(shape=(self.ncc,self.ncc))
        self.B  = np.zeros(shape=(self.ns,self.ncc))
        self.C  = np.zeros(shape=(self.ncc,self.ns))
        
        #
        # @var E [Numpy array]
        # Matrix related to the left-hand side of the system
        #
        self.E  = np.eye(self.ns)
        
        #
        # @var K [Numpy array]
        # Matrix related to the time variation of the basis)
        #  
        self.K = np.zeros(shape=(self.ns,self.ns))
        
        ## @var x
        #  Global effort vector of the system [Numpy array]
        #  @var y 
        #  Global flow vector of the system [Numpy array]
        #  @var s
        #  Global source vector of the system if any [Numpy array]

        self.x  = np.random.rand(self.nt)
        self.y  = np.zeros(shape=(self.nt))
        self.s  = np.zeros(shape=(self.nt))
        
        ## @var istatus_J
        #  Flag related to the J matrix [integer]
        #  @var istatus_Q
        #  Flag related to the Q matrix [integer]
        #  @var istatus_R
        #  Flag related to the R matrix [integer]
        #  @var istatus_S
        #  Flag related to the S matrix [integer]
        #  @var istatus_D
        #  Flag related to the D matrix [integer]
        #  @var istatus_B
        #  Flag related to the B matrix [integer]
        #  @var istatus_G_sys
        #  Flag related to the G_sys matrix [integer]
        #  @var istatus_E
        #  Flag related to the E matrix [integer]
        

        self.istatus_J = 0
        self.istatus_B = 0
        self.istatus_S = 0
        self.istatus_W = 0
        
        #
        # NB: We do impose default matrices for all these components 
        #
        
        self.istatus_E = 1
        self.istatus_Q = 1
        self.istatus_R = 1
        self.istatus_D = 1
        self.istatus_K = 1
        
        ## @var istatus_source_term
        #  Flag related to the source term [integer]
        #  @var status_diff_mode
        #  String related to the type of differentiation mode [string]
        #  @var type
        #  String related to the type of port-Hamiltonian system [string]
        #  @var status
        #  Flag to check if the system has been set
        
        self.istatus_source_term = 0

        self.status_diff_mode = "None"
        
        if self.nt == self.ns and self.mode=="User defined":
        	self.type = "Port Hamiltonian DAE system"	

        if self.ncc > 0 and self.mode=="User defined":
        	self.type = "Input-state-output port-Hamiltonian DAE system"	

        if self.ni > 0 and self.mode=="User defined":
        	self.type = "Port-Hamiltonian DAE system with interconnection"	
  
        if self.ni == 0 and self.nc== 0 and self.mode=="User defined":
        	self.type = "Autonomous port-Hamiltonian DAE system"	
      
         
        # To describe later
        self.istatus = 0 
        
        # Create an automatic object for checking/debugging
        
        if self.mode == "Automatic":
            #
            # Define the different matrices related to the subsystem
            #
            E = np.random.rand(self.ns,self.ns)
            J = np.random.rand(self.ns,self.ns)
            J = 0.5*(J-np.transpose(J))
            R = np.eye(self.ns)
            Q = np.eye(self.ns)
            K = np.zeros(shape=(self.ns,self.ns))
            B = np.random.rand(self.ns,self.nr+self.nc+self.ni) 
            D = np.random.rand(self.ncc,self.ncc)
            D = 0.5*(D+np.transpose(D))
            #
            # Create the global components of the system 
            #
            self.Set_E_matrix(E)
            self.Set_J_matrix(J)
            self.Set_R_matrix(R)
            self.Set_Q_matrix(Q)
            self.Set_K_matrix(K)
            self.Set_B_matrix(B)
            self.Set_D_matrix(D)
            self.Set_S_matrix()
   

    ## Method.
    #  @param self The object pointer.

    def Get_port_storage_flow(self):
        """
        Get the vector related to the output flow variables 
        """
        return self.y[0:self.ns]
     
    ## Method.
    #  @param self The object pointer.
     
    def Get_port_resistive_flow(self):
        """
        Get the vector related to the output dissipation variables
        """
        assert self.nr != 0, "Operation not allowed since there is no resistive space in the subsystem"
        
        return self.y[self.ns:self.ns+self.nr]
        
    ## Method.
    #  @param self The object pointer.
     
    def Get_port_control_flow(self): 
        """
        Get the vector related to the output variables
        """

        assert self.nc != 0, "Operation not allowed since there is no control space in the subsystem"       
        
        return self.y[self.ns+self.nr:self.ns+self.nr+self.nc]       

    ## Method.
    #  @param self The object pointer.
     
    def Get_port_interconnection_flow(self): 
        """
        Get the vector related to the interconnection variables
        """

        assert self.ni != 0, "Operation not allowed since there is no interconnection space in the subsystem"       
        
        return self.y[self.ns+self.nr+self.nc:self.ns+self.ncc]
  
    ## Method.
    #  @param self The object pointer.
       
    def Get_port_storage_effort(self):
        """
        Get the vector related to the input effort variables
        """
        return self.x[0:self.ns]
  
    ## Method.
    #  @param self The object pointer.
       
    def Get_port_resistive_effort(self):
        """
        Get the vector related to the input dissipation variables
        """
        assert self.nr != 0, "Operation not allowed since there is no resistive space in the subsystem"
          
        return self.x[self.ns:self.ns+self.nr]
        
    ## Method.
    #  @param self The object pointer.
   
    def Get_port_control_effort(self):
        """
        Get the vector related to the input control variables
        """
        assert self.nc != 0, "Operation not allowed since there is no control space in the subsystem"       
            
        return self.x[self.ns+self.nr:self.ns+self.nr+self.nc] 
        
    ## Method.
    #  @param self The object pointer.
     
    def Get_port_interconnection_effort(self): 
        """
        Get the vector related to the interconnection variables
        """

        assert self.ni != 0, "Operation not allowed since there is no interconnection space in the subsystem"       
        
        return self.x[self.ns+self.nr+self.nc:self.ns+self.ncc]

    ## Method.
    #  @param self The object pointer.

    def Set_source_term(self,s):
        """
        Set the source term of the PHS system 
        """
        assert self.nt == s.shape[0], "Wrong size in the source term"
        
        self.s = s
        self.istatus_source_term = 1  
     
    ## Method.
    #  @param self The object pointer.
    #  @param E Mass matrix.

    def Set_E_matrix(self,E):
        """
        Set the E matrix of the subsystem. 
        Note that the E matrix can be singular in case of systems with constraints.
        """
        assert self.E.shape[0] == E.shape[0], "Wrong row size in E"
        assert self.E.shape[1] == E.shape[1], "Wrong column size in E"
        
        self.E         = E
        self.istatus_E = 1
        
        return self.istatus_E
    
    ## Method.
    #  @param self The object pointer.
    #  @param E Mass matrix.

    def Set_K_matrix(self,K):
        """
        Set the K matrix of the subsystem. 
        """
        assert self.K.shape[0] == K.shape[0], "Wrong row size in k"
        assert self.K.shape[1] == K.shape[1], "Wrong column size in K"
        
        self.K         = K
        self.istatus_K = 1
        
        return self.istatus_K

    ## Method.
    #  @param self The object pointer.
    #  @param R    Dissipation matrix.

    def Set_R_matrix(self,R):
        """
        Set the dissipation matrix of the subsystem
        """
        assert self.R.shape[0] == R.shape[0], "Wrong row size in R"
        assert self.R.shape[1] == R.shape[1], "Wrong column size in R"
        
        self.R = R
        self.istatus_R = 1
        
        return self.istatus_R
     
    ## Method.
    #  @param self The object pointer.
    #  @param J    Structure matrix.
    
    def Set_J_matrix(self,J):
        """
        Set the structure matrix of the subsystem
        """
        assert self.J.shape[0] == J.shape[0], "Wrong row size in J"
        assert self.J.shape[1] == J.shape[1], "Wrong column size in J"
        
        self.J = J
        self.istatus_J = 1
        
        return self.istatus_J
      
    ## Method.
    #  @param self The object pointer.
    #  @param Q    Q matrix of the pHDAE system.
   
    def Set_Q_matrix(self,Q):
        """
        Set the Q matrix component of the pHDAE system
        """
        assert self.Q.shape[0] == Q.shape[0], "Wrong row size in Q"
        assert self.Q.shape[1] == Q.shape[1], "Wrong column size in Q"
        
        self.Q = Q
        self.istatus_Q = 1
        
        return self.istatus_Q
        
    ## Method.
    #  @param self The object pointer.
    #  @param B    input matrix.
    
    def Set_B_matrix(self,B):
        """
        Set the input matrix of the subsystem
        """
        assert self.B.shape[0] == B.shape[0], "Wrong row size in B"
        assert self.B.shape[1] == B.shape[1], "Wrong column size in B"
        
        self.B = B
        self.istatus_B = 1
        
        # Extract the correct blocks if available 
        
        if self.nr != 0:
            self.Br = self.B[0:self.ns,0:self.nr]
            self.istatus_Br = 1
        else:
            self.Br = []
            self.istatus_Br = 0
        
        if self.nc !=0:
            self.Bc = self.B[0:self.ns,self.nr:self.nr+self.nc]
            self.istatus_Bc = 1
        else:
            self.Bc = []
            self.istatus_Bc = 0
            
        if self.ni !=0:
            self.Bi = self.B[0:self.ns,self.nr+self.nc:self.nr+self.nc+self.ni]
            self.istatus_Bi = 1
        else:
            self.Bi = []
            self.istatus_Bi = 0
        
        return self.istatus_B
      
    def Set_D_matrix(self,D):
        """
        Set the feedthrough matrix of the subsystem
        """
        assert self.D.shape[0] == D.shape[0], "Wrong row size in D"
        assert self.D.shape[1] == D.shape[1], "Wrong column size in D"
        
        self.D = D
        self.istatus_D = 1
        
        
        # Extract the correct blocks if available [TO CHECK:D is assumed to have a block diagonal structure]
        
        if self.nr != 0:
            self.Dr = self.D[0:self.nr,0:self.nr]
            self.istatus_Dr = 1
        else:
            self.Dr = []
            self.istatus_Dr = 0
            
        if self.nc != 0:    
            self.Dc = self.D[self.nr:self.nr+self.nc,self.nr:self.nr+self.nc]
            self.istatus_Dc = 1
        else:
            self.Dc = []
            self.istatus_Dc = 0
            
        if self.ni != 0:    
            self.Di = self.D[self.nr+self.nc:self.nr+self.nc+self.ni,self.nr+self.nc:self.nr+self.nc+self.ni]
            self.istatus_Di = 1
        else:
            self.Di = []
            self.istatus_Di = 0
            
        return self.istatus_D 
    
    ## Method.
    #  @param self The object pointer.
       
    def Set_constraint_matrix(self,G_sys):
        """
        Set the constraint matrix of the subsystem
        Note: this is not needed any longer since the constraints are directly included in 
              the state variable
        """
        pass 
      
    ## Method.
    #  @param self The object pointer.
    
    def Set_S_matrix(self):
        """
        Set the global system matrix of the subsystem.  
        """
        assert self.istatus_J == 1, "The structure matrix (J) must be set before."
        
        assert self.istatus_R == 1, "The dissipation matrix (R) must be set before."
        
        assert self.istatus_E == 1, "The E matrix (E) must be set before."
        
        assert self.istatus_K == 1, "The K matrix (K) must be set before."
        
        assert self.istatus_Q == 1, "The Q matrix (Q) must be set before."
       
        if (self.ncc > 0):
            assert self.istatus_B == 1, "The input matrix (B) must be set before." 
            assert self.istatus_D == 1, "The feed-through matrix (D) must be set before." 
                 
        self.S[0:self.ns,0:self.ns] = np.dot(self.J - self.R,self.Q)-np.dot(self.E,self.K)
        
        if self.istatus_B == 1:
            self.S[0:self.ns,self.ns:self.ns+self.ncc] = self.B
            self.S[self.ns:self.ns+self.ncc,0:self.ns] = np.dot(np.transpose(self.B),self.Q)
        
        if self.istatus_D == 1:
            self.S[self.ns:self.ns+self.ncc,self.ns:self.ns+self.ncc] = self.D
            
       
        self.istatus_S = 1
        
        return self.istatus_S
 
    ## Method.
    #  @param self The object pointer.
     
    def Define_Dynamical_system(self):
        """
        Retrieve the different matrices related to the dynamical system.
        This corresponds to the classical (E,A,B,C,D) formulation of dynamical system theory.  
        """
        assert self.istatus_S == 1, "The global matrix (S) must be set before."

        # Note that we only need to define the A and C blocks since S has been set before.
        # Hence E, B anc D are known. 
        
        self.A = self.S[0:self.ns,0:self.ns]
        self.C = self.S[self.ns:self.ns+self.ncc,0:self.ns]

        self.istatus = 1
        
        return self.istatus
    
    ## Method.
    #  @param self The object pointer.
       
    def Set_W_matrix(self):
        """
        Set the W matrix of the global subsystem. This matrix should be positive semidefinite.
        """
        assert self.istatus_R == 1, "The dissipation matrix (R) must be set before."
        assert self.istatus_Q == 1, "The Q matrix (Q) must be set before."
        assert self.istatus_D == 1, "The D matrix (D) must be set before."
        
        self.W[0:self.ns,0:self.ns] = np.linalg.multi_dot([self.Q.transpose(),self.R,self.Q])
        self.W[self.ns:self.ns+self.ncc,self.ns:self.ns+self.ncc] = 0.5*(self.D+self.D.transpose())
        
        self.istatus_W = 1
        
        return self.istatus_W

    ## Method.
    #  @param self The object pointer.
    #  @param x 
    
    def Set_system_effort_vector(self,x):
        """
        Set an effort vector for the global system
        """
        assert self.nt == x.shape[0], "Wrong dimension in the effort vector"
        self.x[0:self.nt] = x
     
        
    ## Method.
    #  @param self The object pointer.
    #  @param x 

    def Set_status_diff_mode(self,type_diff_mode):
        """
        Set the status of the differentiation mode of the Hamiltonian 
        """
        self.status_diff_mode = type_diff_mode
        
      
#    ## Method.
#    #  @param self The object pointer.
#    #  @param x 
#   
#    def Compute_system_flow_vector(self,y):
#        """
#        Compute an effort vector for the global system as self.y = S y 
#        """
#        assert self.nt == y.shape[0], "Wrong dimension in the flow vector"
#        
#        assert self.istatus_S == 1, "The global system matrix (S) must be set before"
#        
#        self.y[0:self.nt] = numpy.matmul(self.S,y)
# 
#
#    ## Method.
#    #  @param self The object pointer.
#    
#    def Compute_storage_power(self):
#        """
#        Compute the power due to the storage variables [SIGN OK ?]
#        """
#        return np.dot(self.y[0:self.ns],self.x[0:self.ns])
#     
#
#    ## Method.
#    #  @param self The object pointer.
#    
#    def Compute_resistive_power(self):
#        """
#        Compute the dissipated power due to the dissipative terms
#        """
#        power = 0.0
#        
#        if self.nr != 0:
#            return np.dot(self.y[self.ns:self.ns+self.nr],self.x[self.ns:self.ns+self.nr]) 
#        else:
#            return power
#
#    ## Method.
#    #  @param self The object pointer.
#
#    def Compute_interconnection_power(self):
#        """
#        Compute the dissipated power due to the interconnection terms
#        """
#        power = 0.0
#        
#        if self.ni != 0:
#            return np.dot(self.y[self.ns+self.nr+self.nc:self.ns+self.ncc],self.x[self.ns+self.nr+self.nc:self.ns+self.ncc])        
#        else:
#            return power  
#     
#    ## Method.
#    #  @param self The object pointer.
#   
#    def Compute_provided_power(self):
#        """
#        Compute the provided power due to the control/source terms
#        """
#        power = 0.0
#        
#        if self.nc != 0:
#            return np.dot(self.y[self.ns+self.nr:self.ns+self.nr+self.nc],self.x[self.ns+self.nr:self.ns+self.nr+self.nc])
#        else:
#            return power
#     
#    ## Method.
#    #  @param self The object pointer.
#   
#    def Compute_power_balance(self):
#        """
#        Compute the power balance of the subsystem
#        """
#        
#        power_balance  = self.Compute_storage_power()
#        
#        power_balance += self.Compute_resistive_power()
#        
#        power_balance += self.Compute_provided_power()
#        
#        power_balance += self.Compute_interconnection_power()
#        
#        return power_balance
  
    ## Method.
    #  @param self The object pointer.
       
    def Apply_E(self,state_vector):
        """
        Apply the E matrix to a state vector
        """
        assert self.ns == state_vector.shape[0], "Wrong dimension in the state vector"
        assert self.istatus_E == 1, "The E matrix must be set before."        
        
        return np.matmul(self.E,state_vector)   
 
    ## Method.
    #  @param self The object pointer.
       
    def Apply_J(self,state_vector):
        """
        Apply the J matrix to a state vector
        """
        assert self.ns == state_vector.shape[0], "Wrong dimension in the state vector"
        assert self.istatus_J == 1, "The J matrix must be set before."        
        
        return np.matmul(self.J,state_vector)
  
    ## Method.
    #  @param self The object pointer.
       
    def Apply_R(self,state_vector):
        """
        Apply the R matrix to a state vector
        """
        assert self.ns == state_vector.shape[0], "Wrong dimension in the state vector"
        assert self.istatus_R == 1, "The R matrix must be set before."        
        
        return np.matmul(self.R,state_vector)
    
    ## Method.
    #  @param self The object pointer.
       
    def Apply_Q(self,state_vector):
        """
        Apply the Q matrix to a state vector
        """
        assert self.ns == state_vector.shape[0], "Wrong dimension in the state vector"
        assert self.istatus_Q == 1, "The Q matrix must be set before."        
        
        return np.matmul(self.Q,state_vector)   
    
    ## Method.
    #  @param self The object pointer.
       
    def Apply_K(self,state_vector):
        """
        Apply the K matrix to a state vector
        """
        assert self.ns == state_vector.shape[0], "Wrong dimension in the state vector"
        assert self.istatus_K == 1, "The K matrix must be set before."        
        
        return np.matmul(self.K,state_vector)  
 
    ## Method.
    #  @param self The object pointer.
       
    def Apply_B(self,input_vector):
        """
        Apply the B matrix to an input vector
        """
        assert self.ncc == input_vector.shape[0], "Wrong dimension in the input vector"
        assert self.istatus_B == 1, "The B matrix must be set before."        
        
        return np.matmul(self.B,input_vector) 

    ## Method.
    #  @param self The object pointer.
       
    def Apply_B_transpose(self,state_vector):
        """
        Apply the transpose of the B matrix to a state vector
        """
        assert self.ns == state_vector.shape[0], "Wrong dimension in the input vector"
        assert self.istatus_B == 1, "The B matrix must be set before."        
        
        return np.matmul(self.B.transpose(),state_vector)
       
    ## Method.
    #  @param self The object pointer.
       
    def Apply_D(self,input_vector):
        """
        Apply the D matrix to an input vector
        """
        assert self.ncc == input_vector.shape[0], "Wrong dimension in the input vector"
        assert self.istatus_D == 1, "The D matrix must be set before."        
        
        return np.matmul(self.D,input_vector) 

    ## Method.
    #  @param self The object pointer.
       
    def Apply_D_transpose(self,input_vector):
        """
        Apply the transpose of the D matrix to an input vector
        """
        assert self.ncc == input_vector.shape[0], "Wrong dimension in the input vector"
        assert self.istatus_D == 1, "The D matrix must be set before."        
        
        return np.matmul(self.D.transpose(),input_vector) 

    ## Method.
    #  @param self The object pointer.
       
    def Apply_S(self,global_vector):
        """
        Apply the global matrix S to a global vector
        """
        assert self.nt == global_vector.shape[0], "Wrong dimension in the vector"
        assert self.istatus_S == 1, "The S matrix must be set before."        
        
        return np.matmul(self.S,global_vector)

    
    ## Method.
    #  @param self The object pointer.
       
    def Compute_W_inner_product(self,state_vector,input_vector):
        """
        Compute the W inner product applied to a global vector. 
        Note that it is assumed that W is positive semidefinite. 
        Here we do not intend to use the W matrix explicitly. 
        """
        assert self.ns  == state_vector.shape[0], "Wrong dimension in the state vector"
        assert self.ncc == input_vector.shape[0], "Wrong dimension in the input vector"
        assert self.istatus_D == 1, "The D matrix must be set before." 
        assert self.istatus_R == 1, "The R matrix must be set before."        
        assert self.istatus_Q == 1, "The Q matrix must be set before."               
        
        s = np.dot(self.Apply_Q(state_vector).transpose(),self.Apply_R(self.Apply_Q(state_vector)))
        t = np.dot((input_vector).transpose(),self.Apply_D(input_vector))
        t = 0.5*(t + np.dot((input_vector).transpose(),self.Apply_D_transpose(input_vector)))
        
        return s+t 
    
    ## Method.
    #  @param self The object pointer.
       
    def Compute_Time_Derivative_Hamiltonian(self,state_vector,input_vector,output_vector):
        """
        Compute the time derivative of the Hamiltonian
        see Theorem 11 in [BMXZ17]
        """
        assert self.ns  == state_vector.shape[0], "Wrong dimension in the state vector"
        assert self.ncc == input_vector.shape[0], "Wrong dimension in the input vector"
        assert self.ncc == output_vector.shape[0], "Wrong dimension in the output vector"
        assert self.istatus_D == 1, "The D matrix must be set before." 
        assert self.istatus_R == 1, "The R matrix must be set before."        
        assert self.istatus_Q == 1, "The Q matrix must be set before."               
        
        s = np.dot(input_vector.transpose(),output_vector)
        s = s - self.Compute_W_inner_product(state_vector,input_vector)
        
        return s 
  
    ## Method.
    #  @param self The object pointer.
    
    def Get_Hamiltonian(self,state_vector):
        """
        Compute the Hamiltonian of the system at the state vector. 
        Note that E^T Q must be positive semidefinite [Lemma 6, BMXZ17].
        """ 
        assert self.ns == state_vector.shape[0], "Wrong dimension in the state vector"
        assert self.istatus_E == 1, "The E matrix must be set before."   
        assert self.istatus_Q == 1, "The Q matrix must be set before."             
        
        return 0.5*np.dot(self.Apply_E(state_vector).transpose(),self.Apply_Q(state_vector))        
    

    ## Method.
    #  @param self The object pointer.
    
    def Apply_gradient_Hamiltonian(self,state_vector):
        """
        TO DO
        """ 
        pass
    

    def Apply_Transformation_DAE(self,U,V):
        """
        Given U and V, matrices of change of basis and scaling supposed to be invertible, obtain the matrices of the new DAE system in the new basis
        Note that the time variation of the basis V has been discarded when computing K. 
        This transformation is notably required if one intends to obtain a classical pHS when 
        either E or Q is invertible (Remark 14 in [BMXZ17]).
        Note that the new state variable is given by $\tilde{x} = V^{-1}x$.
        """
        assert self.ns == U.shape[0], "Wrong row dimension in U"
        assert self.ns == U.shape[1], "Wrong column dimension in U"
        assert self.ns == V.shape[0], "Wrong row dimension in V"
        assert self.ns == V.shape[1], "Wrong column dimension in V"  
        
        # Computation of the different matrices defining the new phDAE system
        # These matrices will be used to define the new phDAE object
        
        E = np.linalg.multi_dot([U.transpose(),self.E,V])
        J = np.linalg.multi_dot([U.transpose(),self.J,U])
        R = np.linalg.multi_dot([U.transpose(),self.R,U])
        
        Q = np.linalg.solve(U,np.dot(self.Q,V))
        K = np.linalg.solve(V,np.dot(self.K,V))
        B = np.dot(U.transpose(),self.B)
        D = self.D
        
        return (E,J,R,Q,K,B,D)
        
        
    ## Method.
    #  @param self The object pointer.
     
    def Evaluate_resistive_function(self,resistive_vector):
        """
        Evaluate the resistive function to a vector related to the dissipative variables
        """
        assert self.ncc == resistive_vector.shape[0], "Wrong dimension in the resistive vector"
        
        return np.matmul(self.D,resistive_vector)
        
    ## Method.
    #  @param self The object pointer.
  
    def Set_input_vector(self,input_vector):
        """
        Set the input control vector of the system
        """
        assert self.nc == input_vector.shape[0], "Wrong dimension in the input control vector"

        return input_vector
    
#    def Get_coenergy_variable(self,energy_vector,component_string):
#        """
#        Get the coenergy variable corresponding to a given component of the global system
#        """
#        pass

       
if __name__ == '__main__':
    
    
    # Set the parameters for the subsystem 
    
    ns  = 4
    nr  = 3
    nc  = 2 
    ni  = 1 
    
    # Generate a random state vector
    
    x   = np.random.rand(ns,1)
    
    #
    # Test the automatic initialization of the subsystem 
    #
    
    Sub_auto = Subsystem_DAE(ns,nr,nc,ni,mode="auto")
    
    print(Sub_auto.J)
    
    print(Sub_auto.E)
  
    print(Sub_auto.R)
    
    print(Sub_auto.K)
   
    print(Sub_auto.S)
    
    print(Sub_auto.D)
    
    print(Sub_auto.B)

    print(Sub_auto.mode)
   
    print(Sub_auto.Get_Hamiltonian(x))
    
    #
    # Define U and V as random and assumed to be invertible
    #
    
    U = np.random.rand(ns,ns)
    V = np.random.rand(ns,ns)
   
    (E,J,R,Q,K,B,D) = Sub_auto.Apply_Transformation_DAE(U,V)
    
    Sub_auto_transf = Subsystem_DAE(ns,nr,nc,ni)
    
    Sub_auto_transf.Set_E_matrix(E)
    Sub_auto_transf.Set_J_matrix(J)
    Sub_auto_transf.Set_R_matrix(R)
    Sub_auto_transf.Set_Q_matrix(Q)
    Sub_auto_transf.Set_K_matrix(K)
    Sub_auto_transf.Set_B_matrix(B)
    Sub_auto_transf.Set_D_matrix(D)
    Sub_auto_transf.Set_S_matrix()
    
    print("Difference between the Hamiltonian at x in the original and transformed basis")
    print(Sub_auto_transf.Get_Hamiltonian(np.linalg.solve(V,x))-Sub_auto.Get_Hamiltonian(x))       
 
    y = np.random.rand(Sub_auto.ncc,1)
    u = np.random.rand(Sub_auto.ncc,1)
    z = np.vstack((x,u))
    
    print("Computation of the time derivative of the Hamiltonian \n")
    print("...with Compute_Time_Derivative_Hamiltonian \n")
    print(Sub_auto.Compute_Time_Derivative_Hamiltonian(x,u,y))
    
    Sub_auto.Set_W_matrix()
    
    print("...with Compute_W_inner_product \n")
    print(np.dot(u.transpose(),y)-Sub_auto.Compute_W_inner_product(x,u))
    
    print("...with the explicit W matrix \n")
    print(np.dot(u.transpose(),y)-np.dot(z.transpose(),np.matmul(Sub_auto.W,z)))
    