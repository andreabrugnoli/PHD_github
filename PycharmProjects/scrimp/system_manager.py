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
#  Class to handle a system of port-Hamiltonian type.
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
#  and the following electronic reference due to A.J. van der Schaft
# 
#  https://link.springer.com/content/pdf/10.1007%2F978-3-642-03196-0_2.pdf
#
# We consider a constrained Phs of the form:
#                      M_sys xdot = (J-R)     GradH + B u + G_sys lambda  + s_sys
#                            y    = B^T       GradH + D u                 + s_o
#                            0    = G_sys^T   GradH
#
# NB: Up to now we consider the case of a system made of ONLY TWO subsystems
# By repeated applications of the constructor, the general case can be considered. 
# This will be investigated later. 
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math

from util import nullspace, blockdiag, blockstructured

class System:
    """ Define the general settings for a global system of Port-Hamiltonian system type """

    ## The constructor of the Subsystem class.
    #  @param self The object pointer.
    #  @param Subsystem_List list of subsystems that defines the global system [list]
    #  @param C connection matrix [numpy array]
    #  @param connection_type string that defines the type of coupling  [string]
 
    def __init__(self,Subsystem_List,C=None,connection_type="None"):
        """
        Constructor for the System class as a constrained PHS. 
        """
        ## @var ns
        #  dimension of the state space [integer]
        #  @var nr
        #  dimension of the resistive space [integer]
        #  @var nc
        #  dimension of the input/control space [integer]
        #  @var ni   
        #  dimension of the interaction port space [integer]
        #  @var nm   
        #  dimension of the constraint space (space of multipliers) [integer]
        #  @var nb_subsystem
        #  total number of subsystems [integer]

        self.Subsystem_List  = Subsystem_List
        self.C               = C
        self.connection_type = connection_type

        ns, nr, nc, ni, nm, nb_subsystem = 0, 0, 0, 0, 0, 0
        
        for index, Sub in enumerate(Subsystem_List):
            ns += Sub.ns
            nr += Sub.nr
            nc += Sub.nc
            ni += Sub.ni
            nm += Sub.nm
            nb_subsystem += 1
       
        assert ns >= 0, "The dimension of the global state space should be positive"       
        assert nr >= 0, "The dimension of the global resistive space should be positive"       
        assert nc >= 0, "The dimension of the global input/control space should be positive"       
        assert nm >= 0, "The dimension of the global constraint space should be positive"       
        assert nb_subsystem > 0, "The subsystem list is empty"
        
        self.nb_subsystem = nb_subsystem
        self.ns           = ns
        self.nr           = nr
        self.nc           = nc
        # We remove the ports related to the interconnection 
        self.ni           = 0
        self.ncc          = nr+nc
        
        # The case of TWO subsystems is handled here [see the matrix structure]
        
        self.nm = 0        
        
        if self.connection_type == "Gyrator":
            self.nm = 0
            
        if self.connection_type == "Transformer":    
            self.nm = C.shape[1]       
        
        # The interconnection ports have been removed here in this formula
        self.nt           = ns+nr+nc+nm
        
        ## @var J
        #  Structure matrix [Numpy array]
        #  @var Q
        #  Quadratic part of the Hamiltonian if applicable [Numpy array]
        #  @var R
        #  Dissipation matrix [Numpy array]
        #  @var S
        #  Global matrix of the system  [Numpy array]
        #  @var Z
        #  Matrix related to the resistive part (if a linear model is applicable) [Numpy array]
        #  @var G
        #  Matrix related to the constraint part [Numpy array]
        
        #  @var B
        #   matrix [Numpy array]
        #  @var D
        #   matrix   [Numpy array]
        #  @var M
        #   [Numpy array]
       

        self.J  = np.zeros(shape=(self.ns,self.ns)) 
        self.Q  = np.zeros(shape=(self.ns,self.ns))
        self.R  = np.zeros(shape=(self.ns,self.ns))
        
        self.B  = np.zeros(shape=(self.ns,self.ncc))
        self.D  = np.zeros(shape=(self.ncc,self.ncc))
        
        self.G_sys  = np.zeros(shape=(self.ns,self.nm))
    
        self.M  = np.zeros(shape=(self.ns,self.ns))
        self.S  = np.zeros(shape=(self.nt,self.nt))
        

        ## @var x
        #  Global effort vector of the system [Numpy array]
        #  @var y 
        #  Global flow vector of the system [Numpy array]
        #  @var s
        #  Source vector of the system if any [Numpy array]

        self.x  = np.random.rand(self.nt)
        self.y  = np.zeros(shape=(self.nt))
        self.z  = np.zeros(shape=(self.nt-self.nm))
        self.s  = np.zeros(shape=(self.nt))
        
        ## @var istatus_J
        #  Flag related to the J matrix [integer]
        #  @var istatus_Q
        #  Flag related to the Q matrix [integer]
        #  @var istatus_R
        #  Flag related to the R matrix [integer]
        #  @var istatus_S
        #  Flag related to the S matrix [integer]
        #  @var istatus_B
        #  Flag related to the B matrix [integer]
        #  @var istatus_D
        #  Flag related to the D matrix [integer]
        #  @var istatus_G_sys
        #  Flag related to the G_sys matrix [integer]
        #  @var istatus_M
        #  Flag related to the M matrix [integer]
        #  @var istatus_Z
        #  Flag related to the Z matrix [integer]
        
        self.istatus_J = 0
        self.istatus_Q = 0
        self.istatus_R = 0
        self.istatus_S = 0
        
        self.istatus_B = 0
        self.istatus_D = 0
        self.istatus_G_sys = 0
        
        self.istatus_M = 0
        self.istatus_Z = 0
        
        self.istatus_reduced = 0
                
        
    ## Method.
    #  @param self The object pointer.
        
    def Define_PHS(self):
        """
        Define the global PHS system based on information related to Subsystem_List and connection_type.
        This function is valid for the two types of interconnection considered. 
        """
        
        assert self.Subsystem_List[0].istatus_B == 1, "The B matrix must be set in the subsystem 0"
        assert self.Subsystem_List[0].istatus_D == 1, "The D matrix must be set in the subsystem 0"
        assert self.Subsystem_List[0].istatus_J == 1, "The J matrix must be set in the subsystem 0"
        assert self.Subsystem_List[0].istatus_R == 1, "The R matrix must be set in the subsystem 0"
      
        
        J = self.Subsystem_List[0].J
        R = self.Subsystem_List[0].R
        
        # Extract the blocks related to the control ports
        B = self.Subsystem_List[0].Bc
        D = self.Subsystem_List[0].Dc
        
        # Create the block diagonal matrices
        
        for index in range(1,self.nb_subsystem):
            assert self.Subsystem_List[index].istatus_B == 1, "The B matrix must be set in the subsystem %d" %index
            assert self.Subsystem_List[index].istatus_D == 1, "The D matrix must be set in the subsystem %d" %index
            assert self.Subsystem_List[index].istatus_J == 1, "The J matrix must be set in the subsystem %d" %index
            assert self.Subsystem_List[index].istatus_R == 1, "The R matrix must be set in the subsystem %d" %index
            B = blockdiag(B,self.Subsystem_List[index].Bc)
            D = blockdiag(D,self.Subsystem_List[index].Dc)
            J = blockdiag(J,self.Subsystem_List[index].J)
            R = blockdiag(R,self.Subsystem_List[index].R)
         
        # Set the final matrices to the object
         
        self.J = J
        self.R = R
        self.B = B
        self.D = D

        # Set the corresponding istatus variables
        
        self.istatus_J = 1
        self.istatus_R = 1       
        self.istatus_B = 1
        self.istatus_D = 1
     
        # Construct the global matrix of the system named S 
        # This part assumes that subsystems 0 and 1 are coupled by 
        # a connection of type connection_type.
        # The case of multiple systems needs to be handled later.
     
        if self.connection_type == "Gyrator":
            
            # Construct the B1i C B2i^T block and store it in E
            
            n1s = self.Subsystem_List[0].Bi.shape[0]
            n2s = self.Subsystem_List[1].Bi.shape[0]
            W = np.matmul(self.C,np.transpose(self.Subsystem_List[1].Bi))
            C = np.matmul(self.Subsystem_List[0].Bi,W)
            D = np.transpose(C)
            C = -C
            E = blockstructured(np.zeros(shape=(n1s,n1s)),np.zeros(shape=(n2s,n2s)),C,D)
            
        if  self.connection_type == "Transformer":
            
            # Construct the constraint matrix of the system 
            
            B1i = self.Subsystem_List[0].Bi
            n1s = self.Subsystem_List[0].Bi.shape[0]
            n2s = self.Subsystem_List[1].Bi.shape[0]
            self.G_sys[0:n1s,0:self.C.shape[1]]       = -np.matmul(B1i,self.C)
            self.G_sys[n1s:n1s+n2s,0:self.C.shape[1]] = self.Subsystem_List[1].Bi
            self.istatus_G_sys = 1
            
       
        self.S[0:self.ns,0:self.ns]                = self.J - self.R 
        
        if self.connection_type == "Gyrator":
            self.S[0:self.ns,0:self.ns] += E
        
        self.S[0:self.ns,self.ns:self.ns+self.ncc] = self.B
        self.S[self.ns:self.ns+self.ncc,0:self.ns] = np.transpose(self.B)
        self.S[self.ns:self.ns+self.ncc,self.ns:self.ns+self.ncc] = self.D
          
        if self.istatus_G_sys == 1:
            self.S[0:self.ns,self.ns+self.ncc:self.nt] = self.G_sys
            self.S[self.ns+self.ncc:self.nt,0:self.ns] = np.transpose(self.G_sys)
        
        self.istatus_S = 1
        
        # Check if all the subsystems are based on a Quadratic Hamiltonian formulation
     
        dummy = 0 
        for index in range(self.nb_subsystem):
            dummy += self.Subsystem_List[index].istatus_Q
            
        # If it is the case build the block diagonal quadratic part of the Hamiltonian
            
        if dummy == self.nb_subsystem:
           self.istatus_Q = 1 
           Q = self.Subsystem_List[0].Q
           for index in range(1,self.nb_subsystem):
               Q = blockdiag(Q,self.Subsystem_List[index].Q)
           self.Q = Q
           
        return self.istatus_S   
           
    ## Method.
    #  @param self The object pointer. 
           
    def Define_elimination_matrix(self):
        """
        Define the elimination matrix self.M related to the global system. 
        Needed for the Transformer connection type. 
        """
        assert self.istatus_G_sys == 1, "The constraint matrix must be set before."
        assert self.connection_type == "Transformer", "The connection type must be of Transformer type."
        assert self.nm > 0, "The dimension of the global constraint space should be strictly positive."
        
        G_orth = np.transpose(nullspace(np.transpose(self.G_sys)))
               
        # Construct the elimination matrix M  
        
        self.M[0:G_orth.shape[0],0:G_orth.shape[1]] = G_orth
        self.M[G_orth.shape[0]:G_orth.shape[0]+self.nm,0:G_orth.shape[1]] = np.linalg.solve(np.matmul(np.transpose(self.G_sys),self.G_sys),np.transpose(self.G_sys))
    
        
        # Set the corresponding istatus variable
    
        self.istatus_M = 1
        
    ## Method.
    #  @param self The object pointer. 
        
    def Define_reduced_PHS(self):
        """
        Define the reduced system of DAE/ODE after the change of basis defined by the matrix self.M. 
        Required for the Transformer connection type.    
        """
        
        assert self.connection_type == "Transformer", "The connection type must be of transformer type."
        assert self.istatus_M == 1, "The elimination matrix M should be set before."
        
                
        self.J = np.matmul(self.M,np.matmul(self.J,np.transpose(self.M)))
        self.R = np.matmul(self.M,np.matmul(self.R,np.transpose(self.M)))
        self.B = np.matmul(self.M,self.B)
           
        
        # The reduced part corresponds to the (ns-nm) dynamical state vector
        
        self.Jr = self.J[0:self.ns-self.nm,0:self.ns-self.nm]
        self.Rr = self.R[0:self.ns-self.nm,0:self.ns-self.nm]
        self.Br = self.B[0:self.ns-self.nm,0:self.B.shape[1]]
        self.Dr = self.D[0:self.B.shape[1],0:self.B.shape[1]]
        
        # Shall we constuct the complete matrix
        # Do we need to define a subsystem at this stage 
        # to define the reduced system ? [TO CHECK]
        self.Sr = self.Jr-self.Rr
        
        # In case of a quadratic Hamiltonian for the global system
        
        if self.istatus_Q == 1:
           Rhs     = np.linalg.solve(np.transpose(self.M),self.Q)
           Sol     = np.linalg.solve(np.transpose(self.M),np.transpose(Rhs))
           self.Q  = np.transpose(Sol)
           Q11     = self.Q[0:self.nt-self.nm,0:self.nt-self.nm]
           Q12     = self.Q[0:self.nt-self.nm,self.nt-self.nm:self.nt]
           Q21     = self.Q[self.nt-self.nm:self.nt,0:self.nt-self.nm]
           Q22     = self.Q[self.nt-self.nm:self.nt,self.nt-self.nm:self.nt]
           self.Qr = Q11 - np.matmul(Q12,np.linalg.solve(Q22,Q21))
           
        #   
        # Final form of the reduced system is finally obtained
        # 
           
        self.istatus_reduced = 1        
               
    ## Method.
    #  @param self The object pointer. 
    #  @param state_vector        
    #  @param input_vector
               
    def Define_reduced_dynamics(self,state_vector,input_vector):
        """
        Define the dynamics of the reduced system.
        """
        
        assert self.connection_type == "Transformer", "The connection type must be of Transformer type."
        assert self.istatus_reduced == 1, "The reduced system should be defined before."
        assert state_vector.shape[0] == self.nt-self.nm, "Wrong size in the state vector."
        assert input_vector.shape[0] == self.nc, "Wrong size in the input vector."
    
        output_vector = np.zeros(shape=(self.nt-self.nm+self.nc))

        #  

        if self.istatus_Q == 1:
            w = np.matmul(self.Qr,state_vector)
        else:
            temp = self.Apply_gradient_Hamiltonian(state_vector) 
            w    = temp[0:self.nt-self.nm]
        
        # NB: output_vector collects xdot and the classical output_vector y
        # Reduced phs to be considered
        # We consider a constrained Phs of the form:
#                          Xdot = (J-R) GradH + B u + G lambda
#                          Y    = B^T   GradH + D u
#                          0    = G^T   GradH

        output_vector[0:self.nt-self.nm]                      = np.matmul(self.Sr,w) + np.matmul(self.Br,input_vector)
        output_vector[self.nt-self.nm:self.nt-self.nm+self.nc]= np.matmul(np.transpose(self.Br),w) + np.matmul(self.Dr,input_vector) 
            
        return output_vector
        
    ## Method.
    #  @param self The object pointer. 
    #  @param state_vector 
        
    def Apply_gradient_Hamiltonian(self,state_vector):
         """
         Apply the gradient of the Hamiltonian of the global system (eventually in the transformed basis depending on the connection type).
         """
         
         v     = np.zeros(shape=(self.ns))
         shift = 0
         
         # Due to the block diagonal structure apply the gradient independently to each subpart
         # We call the Evaluate_gradient_Hamiltonian function defined in the Subsystem class
         
         for index, Sub in enumerate(self.Subsystem_List): 
             v[0+shift:Sub.ns+shift] = Sub.Apply_gradient_Hamiltonian(state_vector[0+shift:Sub.ns+shift])
             shift += Sub.ns
         
         if self.connection_type == "Transformer":         
             return np.solve(np.transpose(self.M),v)
         elif self.connection_type == "Gyrator":
             return v
         elif self.connection_type == "None":
             return v
  
    ## Method.
    #  @param self The object pointer. 
    #  @param state_vector        
    #  @param input_vector
               
    def Define_dynamics(self,state_vector,input_vector):
        """
        Define the dynamics of the global nonreduced system.
        Xdot = (J-R) GradH + B u 
        Y    = B^T   GradH + D u
        """
        
        assert self.connection_type != "Transformer", "The connection type must be of Gyrator or None type."
        
        assert state_vector.shape[0] == self.ns, "Wrong size in the state vector."
        assert input_vector.shape[0] == self.ncc, "Wrong size in the input vector."
    
        # Define the output vector 
    
        output_vector = np.zeros(shape=(self.ns+self.ncc))

        # Apply the gradient of the Hamiltonian to the state vector

        if self.istatus_Q == 1:
            w = np.matmul(self.Q,state_vector)
            print(state_vector,w,self.Q,self.istatus_Q)
        else:
            w = self.Apply_gradient_Hamiltonian(state_vector) 
        
        # NB: output_vector collects both xdot and the classical output_vector y
        # We consider a Phs of the form:
        #           Xdot = (J-R) GradH + B u 
        #           Y    = B^T   GradH + D u

        output_vector[0:self.ns]                      = np.matmul(self.S[0:self.ns,0:self.ns],w) + np.matmul(self.B,input_vector)
        output_vector[self.ns:self.ns+self.ncc]       = np.matmul(np.transpose(self.B),w) + np.matmul(self.D,input_vector) 
            
        return output_vector
        
    def Define_dynamics_ode(self,time,state_vector):
        """
        Define the dynamics of the partial nonreduced system for the ODE integration.
        This version is required for the ODE method of the scipy.integrate module. 
        Xdot = (J-R) GradH + B u 
        """
        
        assert self.connection_type != "Transformer", "The connection type must be of Gyrator or None type."
        
        assert state_vector.shape[0] == self.ns, "Wrong size in the state vector."
    
        # Define the output vectors 
    
        output_vector = np.zeros(shape=(self.ns))
        
        # Input vector defined through a method
        
        #input_vector  = math.sin(math.pi*time/4.) * np.ones(shape=(self.ncc))
        
        input_vector = self.Input_function(time)

        # Apply the gradient of the Hamiltonian to the state vector

        if self.istatus_Q == 1:
            w = np.matmul(self.Q,state_vector)
        else:
            w = self.Apply_gradient_Hamiltonian(state_vector) 
        
        output_vector = np.matmul(self.S[0:self.ns,0:self.ns],w) + np.matmul(self.B,input_vector)
            
        return output_vector
        
    def Input_function(self,time):
        """
        Input function for the general ODE system. 
        """
        input_vector  = 0.1* math.sin(math.pi*time/4.) * np.ones(shape=(self.ncc))
        #input_vector  = 0.1 * np.ones(shape=(self.ncc))
        
        return input_vector
      
    def Define_dynamics_output(self,time,state_vector):
        """
        Define the dynamics of the output of the global system. 
        Y    = B^T   GradH + D u
        """
        
        assert self.connection_type != "Transformer", "The connection type must be of Gyrator or None type."
        
        assert state_vector.shape[0] == self.ns, "Wrong size in the state vector."
    
        # Define the output vector 
    
        output_vector = np.zeros(shape=(self.ncc))
        
        # Input vector defined through a method
        
        #input_vector  = math.sin(math.pi*time/4.) * np.ones(shape=(self.ncc))
        
        input_vector = self.Input_function(time)

        # Apply the gradient of the Hamiltonian to the state vector

        if self.istatus_Q == 1:
            w = np.matmul(self.Q,state_vector)
            #print state_vector,w,self.Q,self.istatus_Q
        else:
            w = self.Apply_gradient_Hamiltonian(state_vector) 
        
        output_vector = np.matmul(np.transpose(self.B),w) + np.matmul(self.D,input_vector) 
            
        return output_vector
         

if __name__ == '__main__':


    #from dynamics_manager import Dynamical
    from subsystem_manager import Subsystem
    from scipy.integrate import ode
    from linear_oscillator import Linear_Oscillator
    
    # Set the parameters for the first subsystem 
    
    ns, nr, nc, ni, nm  = 4, 0, 1, 2, 0
     
    state_vector  = np.random.rand(ns)
    input_vector  = np.random.rand(nr+nc)
    
    Sub_auto = Subsystem(ns,nr,nc,ni,mode="Automatic")    
    
    Subsytem_List = [Sub_auto]
   
   # Check with a single subsystem 
   
    Sys_single = System(Subsytem_List)
    
    Sys_single.Define_PHS()

    print(Sys_single.S)
    
    print(Sub_auto.S)
    
    print(Sys_single.Define_dynamics(state_vector,input_vector))

   # Check with two identical subsystems 

    Subsytem_List.append(Sub_auto)
    
    Sys_coupled = System(Subsytem_List)
    
    Sys_coupled.Define_PHS()
    
    print(Sys_coupled.S)
    
    state_vector  = np.random.rand(2*ns)
    input_vector  = np.random.rand(2*(nr+nc))    
    
    print(Sys_coupled.Define_dynamics(state_vector,input_vector))
    
   # Check coupling
    
    C = np.random.rand(2,2)
    
    Sys_interconnected = System(Subsytem_List,C,connection_type="Gyrator")

    Sys_interconnected.Define_PHS()
    
    print(Sys_interconnected.S)
    
    if  Sys_interconnected.connection_type == "Transformer":
        Sys_interconnected.Define_elimination_matrix()
        Sys_interconnected.Define_reduced_PHS()

   # Perform the time integration [solution and output parts of the dynamical system]
   # by standard methods from the scipy.integrate module

    y0 = np.random.rand(2*ns)
    t0 = 0 
    t1 = 1
    dt = 0.05
    
    time_integrator = ode(Sys_interconnected.Define_dynamics_ode).set_integrator('vode', method='bdf')
    time_integrator.set_initial_value(y0, t0)
    while time_integrator.successful() and time_integrator.t < t1:
          sol = time_integrator.integrate(time_integrator.t+dt)
          #print(time_integrator.t+dt, sol , Sys_interconnected.Define_dynamics_output(time_integrator.t+dt,sol))
   
    print(time_integrator.successful())
   
    #
    #  First example: two interconnected linear oscillators
    # 
    
    # Set the matrices of the first linear oscillator
  
    M = np.zeros(shape=(1,1))
    C = np.zeros(shape=(1,1)) 
    G = np.zeros(shape=(1,1))
    K = np.zeros(shape=(1,1))
    
    mass = 1.0
    omega = 0.5
    
    M[0,0] = mass
    K[0,0] = omega*omega
    Subsystem_List = []
    
    # Call the constructor for the Linear Oscillator
    
    Lo = Linear_Oscillator(M,C,G,K,0,2,2)
    Lo.Set_Subsystem()   
    
    # Add the linear oscillator to the list of subsystems
    Subsystem_List.append(Lo)
    
    # Call the constructor for the second Linear Oscillator
    
    M[0,0] = 2.0 * mass
    K[0,0] = omega*omega    
    Lo_2   = Linear_Oscillator(M,C,G,K,0,2,2)
    Lo_2.Set_Subsystem() 
    
    # Add the linear oscillator to the list of subsystems
    Subsystem_List.append(Lo_2)
    
    #Sys_coupled = System(Subsystem_List)
    
    # Create the connection matrix
    C = np.random.rand(2,2)
    
    # Create the coupled system
    Sys_coupled = System(Subsystem_List,C,connection_type="Gyrator")
    
    # Define the global system
    Sys_coupled.Define_PHS()
    
    print(Sys_coupled.S)   
    
    # Simulate the dynamical behaviour of the coupled system
    y0 = np.random.rand(Sys_coupled.ns)
    t0 = 0 
    t1 = 1
    dt = 0.05
    
    # Perform the numerical integration 
    time_integrator = ode(Sys_coupled.Define_dynamics_ode).set_integrator('vode', method='bdf')
    time_integrator.set_initial_value(y0, t0)
    
    # Print the global solution at each time step
    while time_integrator.successful() and time_integrator.t < t1:
          sol = time_integrator.integrate(time_integrator.t+dt)
          #print(time_integrator.t+dt, sol , Sys_coupled.Define_dynamics_output(time_integrator.t+dt,sol))
   
    # Final check
    print(time_integrator.successful())