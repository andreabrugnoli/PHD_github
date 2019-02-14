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
#  Class to handle a subsystem of port-Hamiltonian type.
#
#  A port-Hamiltonian system is defined as
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
#  The global subsytem may be written as (with the additional possibility of a constraint matrix G_sys)
#
#                      M_sys xdot = (J-R)     GradH + B u + G_sys lambda  + s_sys
#                            y    = B^T       GradH + D u                 + s_o
#                            0    = G_sys^T   GradH
# 
#
# The role of M_sys has to be detailed later, up to now M_sys is equal to Identity. 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class Subsystem(object):
    """ Define the general settings for a subsystem of Port-Hamiltonian system type """

    ## The constructor of the Subsystem class.
    #  @param self The object pointer.
    #  @param  ns  Dimension of the state space manifold with port variables (fs, es)
    #  @param  nr  Dimension of the resistive space with port variables (fr, er)
    #  @param  ni  Dimension of the interconnection space with port variables (fi, ei)
    #  @param  nc  Dimension of the input/output control space with port variables (fc, ec)
    #  @param  nm  Number of constraints (Lagrange multipliers) if any [by default 0] in case of a constrainted state space \Xi_c
    #  @param  mode Short description related to the Port-Hamiltonian system  [by default "User defined"]

    def __init__(self,ns,nr,nc,ni,nm=0,mode="User defined"):
        """
        Constructor for the Subsystem class.
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
        assert nm >= 0, "The dimension of the constraint space (in case of a constrained subsystem) should be positive" 

        self.ns  = ns
        self.nr  = nr
        self.nc  = nc
        self.ni  = ni
        self.nm  = nm
        self.ncc = nr+nc+ni
        self.nt  = ns+nr+nc+ni+nm
        
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
        self.Q  = np.zeros(shape=(self.ns,self.ns))
        self.R  = np.zeros(shape=(self.ns,self.ns))
        self.S  = np.zeros(shape=(self.nt,self.nt))
        
        # NB: Zero as a shape dimension is allowed in Numpy
        self.D      = np.zeros(shape=(self.ncc,self.ncc))
        self.B      = np.zeros(shape=(self.ns,self.ncc))
        self.G_sys  = np.zeros(shape=(self.ns,self.nm))
        
        self.M_sys  = np.eye(self.ns)
        
        
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
        #  @var istatus_M_sys
        #  Flag related to the M_sys matrix [integer]
        
        self.istatus_J = 0
        self.istatus_Q = 0
        self.istatus_R = 0
        self.istatus_S = 0
        self.istatus_D = 0
        self.istatus_B = 0
        self.istatus_G_sys = 0
        self.istatus_M_sys = 0
        
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
        	self.type = "Classical Hamiltonian system"	

        if self.ncc > 0 and self.mode=="User defined":
        	self.type = "Input-state-output port-Hamiltonian system"	

        if self.ni > 0 and self.mode=="User defined":
        	self.type = "Port-Hamiltonian system with interconnection"	
  
        if self.ni == 0 and self.nc== 0 and self.mode=="User defined":
        	self.type = "Autonomous port-Hamiltonian system"	

        if self.ni == 0 and self.nc== 0 and self.nr == 0 and self.mode=="User defined":
        	self.type = "Autonomous port-Hamiltonian system without energy dissipation"	
      
        if self.nm > 0 and self.mode=="User defined":
            self.type = self.type + "with constraints"
         
        # To describe later
        self.status = 0 
        
        # Finalize the automatic object
        # This needs to be finalized [Q quadratic part, R semidefinite positive]
        
        if self.mode == "Automatic":
            # Define the different matrices related to the subsystem
            J = np.random.rand(self.ns,self.ns)
            J = 0.5*(J-np.transpose(J)) 
            #R = np.random.rand(self.ns,self.ns)
            #R = 0.5*(R+np.transpose(R))
            R = np.eye(self.ns)
            B = np.random.rand(self.ns,self.nr+self.nc+self.ni)
            #D = np.random.rand(self.nr+self.nc+self.ni,self.nr+self.nc+self.ni) 
            D = np.zeros(shape=(self.ncc,self.ncc))
            Q = np.eye(self.ns)
            # Create the global system matrix
            self.Set_structure_matrix(J)
            self.Set_input_matrix(B)
            self.Set_dissipation_matrix(R)
            self.Set_feedthrough_matrix(D)
            self.Set_Hamiltonian_quadratic_component(Q)
            if self.nm != 0:
                G_sys = np.random.rand(self.ns,self.nm)
                self.Set_constraint_matrix(G_sys)
            self.Set_system_matrix()
   

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
    #  @param Mass Mass matrix.

    def Set_mass_matrix(self,M_sys):
        """
        Set the mass matrix of the subsystem. 
        By default this matrix is set to the identity matrix.
        Note that the mass matrix can be singular in case of systems with constraints.
        """
        assert self.M_sys.shape[0] == M_sys.shape[0], "Wrong row size in M_sys"
        assert self.M_sys.shape[1] == M_sys.shape[1], "Wrong column size in M_sys"
        
        self.M_sys         = M_sys
        self.istatus_M_sys = 1
        
        return self.istatus_M_sys



    ## Method.
    #  @param self The object pointer.
    #  @param R    Dissipation matrix.

    def Set_dissipation_matrix(self,R):
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
    
    def Set_structure_matrix(self,J):
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
    #  @param Q    Quadratic part of the Hamiltonian.
   
    def Set_Hamiltonian_quadratic_component(self,Q):
        """
        Set the quadratic part of the Hamiltonian
        """
        assert self.Q.shape[0] == Q.shape[0], "Wrong row size in Q"
        assert self.Q.shape[1] == Q.shape[1], "Wrong column size in Q"
        
        self.Q = Q
        self.istatus_Q = 1
        
        return self.istatus_Q
        
    ## Method.
    #  @param self The object pointer.
    #  @param B    input matrix.
    
    def Set_input_matrix(self,B):
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
      
    def Set_feedthrough_matrix(self,D):
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
        
    def Set_constraint_matrix(self,G_sys):
        """
        Set the constraint matrix of the subsystem
        """
        assert self.G_sys.shape[0] == G_sys.shape[0], "Wrong row size in G_sys"
        assert self.G_sys.shape[1] == G_sys.shape[1], "Wrong column size in G_sys"
        
        self.G_sys         = G_sys
        self.istatus_G_sys = 1
        
        return self.istatus_G_sys 
      
    ## Method.
    #  @param self The object pointer.
    
    def Set_system_matrix(self):
        """
        Set the global system matrix of the subsystem
        """
        assert self.istatus_J == 1, "The structure matrix (J) must be set before."
        
        assert self.istatus_R == 1, "The dissipation matrix (R) must be set before."
        
        if (self.ncc > 0):
            assert self.istatus_B == 1, "The input matrix (B) must be set before." 
            assert self.istatus_D == 1, "The feed-through matrix (D) must be set before." 
           
        if (self.nm > 0):
            assert self.istatus_G_sys == 1, "The constraint matrix (G_sys) must be set before." 
       
        self.S[0:self.ns,0:self.ns]                = self.J - self.R 
        
        if self.istatus_B == 1:
            self.S[0:self.ns,self.ns:self.ns+self.ncc] = self.B
            self.S[self.ns:self.ns+self.ncc,0:self.ns] = np.transpose(self.B)
        
        if self.istatus_D == 1:
            self.S[self.ns:self.ns+self.ncc,self.ns:self.ns+self.ncc] = self.D
            
        if self.istatus_G_sys == 1:
            self.S[0:self.ns,self.ns+self.ncc:self.nt] = self.G_sys
            self.S[self.ns+self.ncc:self.nt,0:self.ns] = np.transpose(self.G_sys)
        
        self.istatus_S = 1
        
        return self.istatus_S
      

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
   
    def Compute_system_flow_vector(self,y):
        """
        Compute an effort vector for the global system as self.y = S y 
        """
        assert self.nt == y.shape[0], "Wrong dimension in the flow vector"
        
        assert self.istatus_S == 1, "The global system matrix (S) must be set before"
        
        self.y[0:self.nt] = numpy.matmul(self.S,y)
 
    ## Method.
    #  @param self The object pointer.
    #  @param x 

    def Set_status_diff_mode(self,type_diff_mode):
        """
        Set the status of the differentiation mode of the Hamiltonian 
        """
        self.status_diff_mode = type_diff_mode
 

    ## Method.
    #  @param self The object pointer.
    
    def Compute_storage_power(self):
        """
        Compute the power due to the storage variables [SIGN OK ?]
        """
        return np.dot(self.y[0:self.ns],self.x[0:self.ns])
     

    ## Method.
    #  @param self The object pointer.
    
    def Compute_resistive_power(self):
        """
        Compute the dissipated power due to the dissipative terms
        """
        power = 0.0
        
        if self.nr != 0:
            return np.dot(self.y[self.ns:self.ns+self.nr],self.x[self.ns:self.ns+self.nr]) 
        else:
            return power

    ## Method.
    #  @param self The object pointer.

    def Compute_interconnection_power(self):
        """
        Compute the dissipated power due to the interconnection terms
        """
        power = 0.0
        
        if self.ni != 0:
            return np.dot(self.y[self.ns+self.nr+self.nc:self.ns+self.ncc],self.x[self.ns+self.nr+self.nc:self.ns+self.ncc])        
        else:
            return power  
     
    ## Method.
    #  @param self The object pointer.
   
    def Compute_provided_power(self):
        """
        Compute the provided power due to the control/source terms
        """
        power = 0.0
        
        if self.nc != 0:
            return np.dot(self.y[self.ns+self.nr:self.ns+self.nr+self.nc],self.x[self.ns+self.nr:self.ns+self.nr+self.nc])
        else:
            return power
     
    ## Method.
    #  @param self The object pointer.
   
    def Compute_power_balance(self):
        """
        Compute the power balance of the subsystem
        """
        
        power_balance  = self.Compute_storage_power()
        
        power_balance += self.Compute_resistive_power()
        
        power_balance += self.Compute_provided_power()
        
        power_balance += self.Compute_interconnection_power()
        
        return power_balance
        
    ## Method.
    #  @param self The object pointer.
       
    def Apply_gradient_Hamiltonian(self,state_vector):
        """
        Apply the gradient of the Hamiltonian to a state vector
        """
        assert self.ns == state_vector.shape[0], "Wrong dimension in the state vector"
        assert self.istatus_Q == 1, "Up to now this function assumes a quadratic Hamiltonian (to be set)."        
        
        # Call the appropriate function depending on self.diff_mode and self.Hamiltonian_quadratic

        if self.istatus_Q == 1:
            return np.matmul(self.Q,state_vector)
        
        
   
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
    
#    def Get_Hamiltonian(self,state_vector):
#        """
#        Compute the Hamiltonian of the system at the state vector
#        """  
#        pass
#    
#    def Get_coenergy_variable(self,energy_vector,component_string):
#        """
#        Get the coenergy variable corresponding to a given component of the global system
#        """
#        pass

       
if __name__ == '__main__':
    
    import numpy   
    
    # Set the parameters for the subsystem 
    
    ns  = 2
    nr  = 0
    nc  = 1 
    ni  = 1 
    nm  = 0
    
    # Define the subsystem (unconstrained case)
    
    Sub = Subsystem(ns,nr,nc,ni)
    
    J = numpy.random.rand(ns,ns)
    J = 0.5*(J-numpy.transpose(J)) 
      
    R = numpy.random.rand(ns,ns)
    R = 0.5*(R+numpy.transpose(R))

    print("J-R", J-R)    
    
    B = numpy.random.rand(ns,nr+nc+ni)
    D = numpy.random.rand(nr+nc+ni,nr+nc+ni)
    
    print("B", B)
    
    print("D", D)
    
    # Set the different blocks of the global matrix S
    
    print(Sub.Set_structure_matrix(J))
    
    print(Sub.Set_dissipation_matrix(R))
    
    print(Sub.Set_input_matrix(B))
    
    print(Sub.Set_feedthrough_matrix(D))
    
    print(Sub.Set_system_matrix())
    
    print(Sub.S)
    
    print(Sub.Bc)
    
    # Energy check
    
    print(Sub.Compute_power_balance())
    
    
    # Test the automatic initialization of the subsystem 
    
    Sub_auto = Subsystem(ns,nr,nc,ni,nm=1,mode="auto")
    
    print(Sub_auto.J)
    
    print(Sub_auto.S)
    
    print(Sub_auto.D)
    
    print(Sub_auto.B)
    
    print(Sub_auto.R)
    
    print(Sub_auto.mode)
    
    print(Sub_auto.nm)
    