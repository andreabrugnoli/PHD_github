#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Authors   : A. Brugnoli & X. Vasseur
Template  : Mindlin plate as port-Hamiltonian system        
Project   : INFIDHEM https://websites.isae-supaero.fr/infidhem/
Openforge : https://openforge.isae.fr/projects/scrimp
Contact   : andrea.brugnoli@isae.fr & xavier.vasseur@isae.fr
"""

from dolfin import *
from mshr import *

import numpy as np
import scipy.linalg as linalg
import scipy.integrate as integrate
from scipy.sparse import csc_matrix, csr_matrix, save_npz,\
 bmat, block_diag, hstack, vstack
from scipy.sparse.linalg import factorized, spsolve, eigs
from scikits import umfpack

from assimulo.problem import Explicit_Problem, Implicit_Problem
from assimulo.solvers.sundials import IDA, CVode 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as manimation

import sys
import time


        
class Mindlin:
    #%% CONSTRUCTOR
    def __init__(self):
        """
        Constructor for the Mindlin class
        """
        # Information related to the problem definition
        self.set_rectangular_domain           = 0
        self.set_physical_parameters          = 0
        self.set_damping                      = 0
        self.damp                             = []    
        self.dynamic_R                        = False                      
        self.set_boundary_control             = 0
        self.set_mixed_boundaries             = 0
        self.set_dirichlet_boundary_control   = 0
        self.set_normal_boundary_control      = 0
        self.set_initial_final_time           = 0
        self.problem_definition               = 0
        
        # Information related to the space-time discretization
        self.generate_mesh                    = 0
        self.set_finite_elements_spaces       = 0
        self.assembly                         = 0
        self.project_boundary_control         = 0
        self.project_initial_data             = 0

        
        # Information related to the post-processing
        self.interactive                      = False
        self.docker                           = False
        self.notebook                         = False
        
    #%%

    #%% Call method
    def __call__(self):
        print('Mindlin plate as port-Hamiltonian system: created by Andrea Brugnoli')
        print(40*'-', '\n')
        pass 

    #%% PROBLEM DEFINITION
    
    ## Method.
    #  @param self The object pointer.
    def Check_Problem_Definition(self):
        """
        Check if the problem definition has been performed correctly
        """             
        assert self.set_rectangular_domain == 1, \
            'The domain has not been defined properly'

        assert self.set_physical_parameters == 1, \
            'The physical parameters have not been set properly'
        
        assert self.set_boundary_control == 1 or \
        ( self.set_dirichlet_boundary_control == 1 and self.set_normal_boundary_control  == 1), \
            'Boundary control has not been set properly'

        assert self.set_initial_final_time == 1, \
            'The initial and final times have not been set' 
            
        self.problem_definition = 1
        
        return self.problem_definition
    
    
    ## Method.
    # @param self The object pointer.
    def Set_Rectangular_Domain(self, x0, xL, y0, yL):
        """
        Set the dimensions of the rectangular domain
        """
        self.x0 = x0
        self.xL = xL
        self.y0 = y0
        self.yL = yL
        
        self.set_rectangular_domain = 1

        pass  


    ## Method.
    #  @param self The object pointer.
    #  @param rho : The density
    #  @param h : The thickness
    #  @param E : The Young modulus
    #  @param nu : The Poisson ratio
    #  @param k : The shear correction factor
    def Set_Physical_Parameters(self, rho, h, E, nu, k, init_by_value=False, **kwargs):
        """
        Set the physical parameters as a FeniCS expression related to the PDE
        """
        if init_by_value:
            self.rho = rho
            self.h = h
            self.E = E
            self.nu =nu
            self.k = k
            
        else:
            self.rho = Expression(rho, degree=2,\
                                  x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                                  **kwargs)
            self.h = Expression(h, degree=2,\
                                  x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                                  **kwargs)
            
            self.E = Expression(E, degree=2,\
                                  x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                                  **kwargs)
            
            self.nu = Expression(nu, degree=2,\
                                  x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                                  **kwargs)
            
            self.k = Expression(k, degree=2,\
                                  x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                                  **kwargs)
           
        self.set_physical_parameters = 1
        
        pass
    
    
    ## Method.
    #  @param self The object pointer.
    def Set_Damping(self, damp=[], Rtime_func=None, eps=None,  **kwargs):
        if Rtime_func is not None :
            self.dynamic_R = True
            self.Rtime = Rtime_func
        else : self.dynamic_R = False
                
        self.damp = damp
      
        if 'internal' in self.damp:
            self.eps = Expression(eps, degree=2,\
              x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
              **kwargs)
        
        print('Damping:', ((len(damp)*'%1s,  ') % tuple(damp))[:-3])
        print('Damping: OK')
        print(40*'-', '\n')

        self.set_damping = 1
        
        pass

    
    ## Method.
    #  @param self The object pointer.    
    def Set_Mixed_Boundaries(self, Dir=[], Nor=[]):
        self.Dir = Dir
        self.Nor = Nor
        self.set_mixed_boundaries = 1
    
        print('Clamped condition:', (len(Dir)*'%s,  ' % tuple(Dir))[:-3])
        print('Free condition:', (len(Nor)*'%s,  ' % tuple(Nor))[:-3])
        print('Mixed Boundaries: OK')
        print(40*'-', '\n')
        
        pass
    
    ## Method.
    #  @param self The object pointer.
    def Set_Boundary_Control(self, Ub_tm0, Ub_sp0, Ub_tm1=lambda t : np.array([0,0,0]),\
                             Ub_sp1=('0','0','0'), **kwargs):
        """
        Set boundary control as callable time functions and regular FeniCS expression
        u_\partial(t,x) = Ub_tm0(t) * Ub_sp0(x) + Ub_tm1(t) + Ub_sp1(x)
        """
        self.Ub1_tm0 = lambda t : Ub_tm0(t)[0]
        self.Ub2_tm0 = lambda t : Ub_tm0(t)[1]
        self.Ub3_tm0 = lambda t : Ub_tm0(t)[2]

        self.Ub1_sp0_Expression = Expression(Ub_sp0[0], degree=2,\
                                              x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                                              **kwargs)
        
        self.Ub2_sp0_Expression = Expression(Ub_sp0[1], degree=2,\
                                              x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                                              **kwargs)
        
        self.Ub3_sp0_Expression = Expression(Ub_sp0[2], degree=2,\
                                              x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                                              **kwargs)
        
        self.Ub1_tm1 = lambda t : Ub_tm1(t)[0]
        self.Ub2_tm1 = lambda t : Ub_tm1(t)[1]
        self.Ub3_tm1 = lambda t : Ub_tm1(t)[2]

        self.Ub1_sp1_Expression = Expression(Ub_sp1[0], degree=2,\
                                              x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                                              **kwargs)
        
        self.Ub2_sp1_Expression = Expression(Ub_sp1[1], degree=2,\
                                              x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                                              **kwargs)
        
        self.Ub3_sp1_Expression = Expression(Ub_sp1[2], degree=2,\
                                              x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                                              **kwargs)
        self.set_boundary_control = 1
        
        print('Boundary control: OK')
        print(40*'-', '\n')
        
        pass 
    
    ## Method.
    #  @param self The object pointer.
    def Set_Mixed_BC_Dirichlet(self, Ub_tm0, Ub_sp0, Ub_tm1=lambda t : np.array([0,0,0]),\
                               Ub_sp1=('0','0','0'), Ub_tm0_dir=lambda t : np.array([0,0,0]),\
                               Ub_tm1_dir=lambda t : np.array([0,0,0]), **kwargs):
        """
        Set boundary control as callable time functions and regular FeniCS expression
        u_\partial(t,x) = Ub_tm0(t) * Ub_sp0(x) + Ub_tm1(t) + Ub_sp1(x)
        """
        self.Ub1_tm0_D = lambda t : Ub_tm0(t)[0]
        self.Ub2_tm0_D = lambda t : Ub_tm0(t)[1]
        self.Ub3_tm0_D = lambda t : Ub_tm0(t)[2]

        self.Ub1_sp0_D_Expression = Expression(Ub_sp0[0], degree=2,\
                                              x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                                              **kwargs)
        
        self.Ub2_sp0_D_Expression = Expression(Ub_sp0[1], degree=2,\
                                              x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                                              **kwargs)
        
        self.Ub3_sp0_D_Expression = Expression(Ub_sp0[2], degree=2,\
                                              x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                                              **kwargs)
        
        self.Ub1_tm1_D = lambda t : Ub_tm1(t)[0]
        self.Ub2_tm1_D = lambda t : Ub_tm1(t)[1]
        self.Ub3_tm1_D = lambda t : Ub_tm1(t)[2]

        self.Ub1_sp1_D_Expression = Expression(Ub_sp1[0], degree=2,\
                                              x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                                              **kwargs)
        
        self.Ub2_sp1_D_Expression = Expression(Ub_sp1[1], degree=2,\
                                              x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                                              **kwargs)
        
        self.Ub3_sp1_D_Expression = Expression(Ub_sp1[2], degree=2,\
                                              x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                                              **kwargs)
        
        self.Ub1_tm0_D_dir = lambda t : Ub_tm0_dir(t)[0]
        self.Ub2_tm0_D_dir = lambda t : Ub_tm0_dir(t)[1]
        self.Ub3_tm0_D_dir = lambda t : Ub_tm0_dir(t)[2]
        
        self.Ub1_tm1_D_dir = lambda t : Ub_tm1_dir(t)[0]
        self.Ub2_tm1_D_dir = lambda t : Ub_tm1_dir(t)[1]
        self.Ub3_tm1_D_dir = lambda t : Ub_tm1_dir(t)[2]     
        
        self.set_dirichlet_boundary_control = 1
        
        pass
    
    
    ## Method.
    #  @param self The object pointer.
    def Set_Mixed_BC_Normal(self, Ub_tm0, Ub_sp0, Ub_tm1=lambda t : np.array([0,0,0]),\
                            Ub_sp1=('0','0','0'), **kwargs):
        """
        Set boundary control as callable time functions and regular FeniCS expression
        u_\partial(t,x) = Ub_tm0(t) * Ub_sp0(x) + Ub_tm1(t) + Ub_sp1(x)
        """
        self.Ub1_tm0_N = lambda t : Ub_tm0(t)[0]
        self.Ub2_tm0_N = lambda t : Ub_tm0(t)[1]
        self.Ub3_tm0_N = lambda t : Ub_tm0(t)[2]

        self.Ub1_sp0_N_Expression = Expression(Ub_sp0[0], degree=2,\
                                              x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                                              **kwargs)
        
        self.Ub2_sp0_N_Expression = Expression(Ub_sp0[1], degree=2,\
                                              x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                                              **kwargs)
        
        self.Ub3_sp0_N_Expression = Expression(Ub_sp0[2], degree=2,\
                                              x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                                              **kwargs)
        
        self.Ub1_tm1_N = lambda t : Ub_tm1(t)[0]
        self.Ub2_tm1_N = lambda t : Ub_tm1(t)[1]
        self.Ub3_tm1_N = lambda t : Ub_tm1(t)[2]

        self.Ub1_sp1_N_Expression = Expression(Ub_sp1[0], degree=2,\
                                              x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                                              **kwargs)
        
        self.Ub2_sp1_N_Expression = Expression(Ub_sp1[1], degree=2,\
                                              x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                                              **kwargs)
        
        self.Ub3_sp1_N_Expression = Expression(Ub_sp1[2], degree=2,\
                                              x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                                              **kwargs)
        
        self.set_normal_boundary_control = 1

        pass 




    ## Method.
    #  @param self The object pointer. 
    def Set_Initial_Data(self, W_0=None, Th_0_1=None, Th_0_2=None,\
                         Apw_0=None, Apth1_0=None, Apth2_0=None,\
                         Aqth11_0=None, Aqth12_0=None, Aqth22_0=None,\
                         Aqw1_0=None, Aqw2_0=None,\
                         init_by_vector=False, W0=None, Th0=None,\
                         Apw0=None, Apth0=None, Aqth0=None, Aqw0=None, **kwargs):
        """
        Set initial data on the FE spaces 
        """
        # Expressions
        if not init_by_vector :
            self.init_by_vector = False
            
            self.W_0  = Expression(W_0, degree=2, x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL, **kwargs)
            self.Th_0  = Expression((Th_0_1, Th_0_2), degree=2, x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL, **kwargs)

            self.Apw_0 = Expression(Apw_0, degree=2, x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL, **kwargs) 
            self.Apth_0 = Expression((Apth1_0, Apth2_0), degree=2, x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL, **kwargs) 

            self.Aqth_0 = Expression((Aqth11_0, Aqth12_0, Aqth22_0),\
                                     degree=2, x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL, **kwargs) 
            
            self.Aqw_0 = Expression((Aqw1_0, Aqw2_0), degree=2, x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL, **kwargs) 

        else :
            # Vectors
            self.W0              = W0
            self.Th0             = Th0

            self.Apw0            = Apw0
            self.Apth0           = Apth0
            self.Aqth0           = Aqth0
            self.Aqw0            = Aqw0
            
            self.A0             = np.concatenate((self.Apw0,self.Apth0,\
                                                  self.Aqth0, self.Aqw0 ))
            self.init_by_vector = True

        self.set_initial_data = 1

        pass
   
    
    ## Method.
    #  @param self The object pointer.
    def Set_Initial_Final_Time(self, initial_time, final_time):
        """
        Set the initial, close and final times for defining the time domain
        """
        self.tinit  = initial_time 
        self.tfinal = final_time
        
        self.set_initial_final_time = 1
        
        pass
    
    #%% 
        
    #%% SPACE-TIME DISCRETIZATION
    
    ## Method.
    #  @param self The object pointer.
    def Check_Space_Time_Discretization(self):
        """
        Check if the space and time discretizations have been performed correctly
        """
        
        assert self.generate_mesh == 1, \
            'The finite element mesh must be generated first'
        
        assert self.set_finite_elements_spaces == 1, \
            'The FE approximation spaces must be selected first'
            
        assert self.assembly == 1, \
            'The PFEM formulation has to be applied' 
    
        assert self.project_boundary_control == 1, \
            'The BC must be interpolated on the FE spaces' 
        
        assert self.project_initial_data == 1, \
            'The initial data must be interpolated on the FE spaces' 

        assert self.set_time_setting == 1,\
            'The parameters for the time discretization must be set'
    
        self.space_time_discretization = 1
        
        return self.space_time_discretization
      
    ## Method.
    #  @param self The object pointer.   
    def Generate_Mesh(self, rfn, structured_mesh=False):
        """
        Perform the mesh generation through the Fenics meshing functionalities
        """
        self.rfn = rfn  
    
        rfny =  int (self.rfn * (self.yL - self.y0) / (self.xL - self.x0) ) 
        
        if structured_mesh: 
            self.Msh = RectangleMesh(Point(self.x0,self.y0), Point(self.xL,self.yL),\
                                     self.rfn, rfny, "crossed")
        else:
            self.Msh = generate_mesh(Rectangle(Point(self.x0,self.y0), Point(self.xL,self.yL)),\
                                     self.rfn)
        

        # Explicit coordinates
        self.xv = self.Msh.coordinates()[:,0]
        self.yv = self.Msh.coordinates()[:,1]
        self.Nv = self.Msh.num_vertices()
        
        self.norext   = FacetNormal(self.Msh)
        self.tanext = as_vector([-self.norext[1], self.norext[0]])

        print('Mesh:', 'hmax=', round(self.Msh.hmax(),3),'Nv=', self.Msh.num_vertices())
        print('Mesh: OK')
        print(40*'-', '\n')
        
        self.generate_mesh = 1
        
        pass  
    
    ## Method.
    # @param self The object pointer. 
    def Set_Gmsh_Mesh(self, xmlfile, rfn_num=0):
        """
        Set a mesh generated by Gmsh using "xml" format
        """
        # Create FENCIS mesh
        Msh = Mesh(xmlfile)

        num_vertices = Msh.num_vertices()
        num_cells = Msh.num_cells()
        vertices = Msh.coordinates()[:, :2]
        cells = Msh.cells()
        
        Msh = Mesh()
        editor = MeshEditor()
        editor.open(Msh, 'triangle', 2, 2) 
        editor.init_vertices(num_vertices)  
        editor.init_cells(num_cells)
        for i in range(num_vertices) :
            editor.add_vertex(i, vertices[i])    
        for i in range(num_cells) :
            editor.add_cell(i, cells[i])    
        editor.close()
        
        # Refinement
        Msh_list = [Msh]
        rfn, rfn_num = 0, rfn_num
        while rfn < rfn_num : 
            Msh = refine(Msh)
            Msh_list.append(Msh)
            rfn += 1
            
        # Mesh
        self.Msh = Msh
        self.norext   = FacetNormal(self.Msh)
        self.tanext = as_vector([-self.norext[1], self.norext[0]])

        # Explicit coordinates
        self.xv = Msh.coordinates()[:,0]
        self.yv = Msh.coordinates()[:,1]
        self.Nv = self.Msh.num_vertices()

        # Domain limits 
        self.x0, self.xL = self.xv.min(), self.xv.max() 
        self.y0, self.yL = self.yv.min(), self.yv.max()

        print('Mesh:', 'hmax=', round(self.Msh.hmax(),3),'Nv=', self.Msh.num_vertices() )
        print('Mesh: OK')
        print(40*'-', '\n')
        
        self.generate_mesh = 1
        
        pass
    
    ## Method.
    # @param self The object pointer.         
    def Assign_Mesh(self, Msh):
        """
        Assign already generated mesh
        """
        # Mesh
        self.Msh = Msh
        self.norext = FacetNormal(self.Msh)
        self.tanext = as_vector([-self.norext[1], self.norext[0]])

        
        # Explicit coordinates
        self.xv = Msh.coordinates()[:,0]
        self.yv = Msh.coordinates()[:,1]
        self.Nv = self.Msh.num_vertices()


        # Domain limits
        self.x0, self.xL = self.xv.min(), self.xv.max() 
        self.y0, self.yL = self.yv.min(), self.yv.max()
        
        print('Mesh:', 'hmax=', round(self.Msh.hmax(),3),'Nv=', self.Msh.num_vertices() )
        print('Mesh: OK')
        print(40*'-', '\n')
    
        self.generate_mesh = 1
        
        pass
    
    ## Method.
    #  @param self The object pointer. 
    #  @param family_b The element for boundary variables

    def Set_Finite_Elements_Spaces(self, r=1, family_b='CG', rb=1):        
        """
        Set the finite element approximation spaces related to the space discretization of the PDE
        """
        assert self.generate_mesh == 1, "The finite element mesh must be generated first"
        # Spaces
        P_pw = FiniteElement('CG', triangle, r)
        P_pth = VectorElement('CG', triangle, r)
        P_qth = VectorElement('DG', triangle, r-1, dim=3)
        P_qw = VectorElement('DG', triangle, r-1)
        
        element = MixedElement([P_pw, P_pth, P_qth, P_qw])
        V = FunctionSpace(self.Msh, element)
            
        self.V = V
        
        self.Vpw = self.V.sub(0)
        self.Vpth = self.V.sub(1)
        self.Vqth = self.V.sub(2)
        self.Vqw = self.V.sub(3)
  
        if rb == 0 :
            self.Vb = FunctionSpace(self.Msh, 'CR', 1)
        else :
            self.Vb = FunctionSpace(self.Msh, family_b, rb)
            
        # Orders
        self.r = r
        self.rb = rb
        
        # DOFs
        self.Npw = self.Vpw.dim()
        self.Npth = self.Vpth.dim()
        
        self.Nqth = self.Vqth.dim()
        self.Nqw = self.Vqw.dim()

        self.Np = self.Npw + self.Npth
        self.Nq = self.Nqth + self.Nqw
        self.Nsys = self.V.dim()
        
        self.dofs_Vpw = self.Vpw.dofmap().dofs()
        self.dofs_Vpth = self.Vpth.dofmap().dofs()
        
        self.dofs_Vp = np.concatenate((self.dofs_Vpw, self.dofs_Vpth))
        
        self.dofs_Vqth = self.Vqth.dofmap().dofs()
        self.dofs_Vqw = self.Vqw.dofmap().dofs()
        
        self.dofs_Vq = np.concatenate((self.dofs_Vqth, self.dofs_Vqw))
        
        # Explicit coordinates
        coord = self.V.tabulate_dof_coordinates()

        
        self.xpw = coord[self.dofs_Vpw,0]
        self.ypw = coord[self.dofs_Vpw,1]
        
        self.xpth = coord[self.dofs_Vpth,0]
        self.ypth = coord[self.dofs_Vpth,1]
        
        self.xqth = coord[self.dofs_Vqth,0]
        self.yqth = coord[self.dofs_Vqth,1] 
        
        self.xqw = coord[self.dofs_Vqw,0]
        self.yqw = coord[self.dofs_Vqw,1] 
        
        # Boundary DOFs for control and observation: necessary for B matrix        
        coord_b  = self.Vb.tabulate_dof_coordinates()
        xb       = coord_b[:,0]
        yb       = coord_b[:,1]
        # Boundary DOFs for control and observation: necessary for B matrix        
        coord_b = self.Vb.tabulate_dof_coordinates()
        def Get_Associated_DOFs(SubDomain, Space):
            dim_space = Space.num_sub_spaces()
            if dim_space == 0:
                BC = DirichletBC(Space, '0', SubDomain)
            else:
                BC = DirichletBC(Space, ('0',)*dim_space, SubDomain)
            bndr_dofs = []
            for key in BC.get_boundary_values().keys() :
                bndr_dofs.append(key)
            bndr_dofs = sorted(list(set(bndr_dofs)))
            return bndr_dofs
        b_ind = Get_Associated_DOFs('on_boundary', self.Vb)
        self.b_ind = b_ind
        

        # Exlpicit information about boundary DOFs 
        self.coord_b_full = coord_b
        coord_b      = coord_b[self.b_ind,:]
        self.Nb      = len(self.b_ind)
        self.xb      = xb[b_ind]
        self.yb      = yb[b_ind]
           
        # Corners indexes (boundary DOFs)
        self.Corner_indices = []
        for i in range(self.Nb) :
            if ( np.abs(self.xb[i] - self.x0) <= 1e-16 and np.abs(self.yb[i] - self.y0) <= 1e-16 ) \
            or ( np.abs(self.xb[i] - self.x0) <= 1e-16 and np.abs(self.yb[i] - self.yL) <= 1e-16 ) \
            or ( np.abs(self.xb[i] - self.xL) <= 1e-16 and np.abs(self.yb[i] - self.y0) <= 1e-16 ) \
            or ( np.abs(self.xb[i] - self.xL) <= 1e-16 and np.abs(self.yb[i] - self.yL) <= 1e-16 ) : 
                 self.Corner_indices.append(i)
         
#        print('Vpw=', family_pw+'_'+str(rp),\
#              ',\t Vpth=', family_pth +'_'+str(rp),\
#              ',\t Vqth=', family_qth +'_'+str(rq), \
#              ',\t Vqw=', family_qw +'_'+str(rq),\
#              ',\t Vb=', family_b+'_'+str(rb))
        print('Npw=', self.Npw,\
              ',\t Npth=', self.Npth,\
              ',\t Nqth=', self.Nqth,\
              ',\t Nqw=', self.Nqw,\
              ',\t Nb=', self.Nb)
        print('DOFsys=', self.Npw+self.Npth+self.Nqth+self.Nqw)
        print('FE spaces: OK')
        print(40*'-', '\n')
        
        self.set_finite_elements_spaces = 1
        
        pass
    
    ## Method.
    #  @param self The object pointer.   
    def Assembly(self):
        """
        Perform the matrix assembly related to the PFEM formulation
        """
        assert self.generate_mesh == 1, "The finite element mesh must be generated first"
        
        assert self.set_finite_elements_spaces == 1, \
                "The finite elements spaces must be selected first"
                
        fl_rot = 12. / (self.E * self.h ** 3)
        G = self.E / 2 / (1 + self.nu)
        F = G * self.h *self.k


        def bending_curv(momenta):
            d = self.Msh.geometry().dim()

            kappa = fl_rot * ((1+self.nu)*momenta - self.nu * Identity(d) * tr(momenta))
            return kappa
        
        def gradSym(u):
            return 0.5 * (nabla_grad(u) + nabla_grad(u).T)
            # return sym(nabla_grad(u))
        
        # Functions        
        v = TestFunction(self.V)
        v_pw, v_pth, v_qth, v_qw = split(v)
        
        e = TrialFunction(self.V)
        e_pw, e_pth, e_qth, e_qw = split(e)
        
        v_qth = as_tensor([[v_qth[0], v_qth[1]],
                            [v_qth[1], v_qth[2]]
                           ])
        
        e_qth = as_tensor([[e_qth[0], e_qth[1]],
                            [e_qth[1], e_qth[2]]
                           ])
        
        al_pw = self.rho * self.h * e_pw
        al_pth = (self.rho * self.h ** 3) / 12. * e_pth
        al_qth = bending_curv(e_qth)
        al_qw = 1. / F * e_qw
    
#        al_pw = e_pw
#        al_pth = e_pth
#        al_qth = e_qth
#        al_qw = e_qw
        
        ub = TrialFunction(self.Vb)
        
        u_qn, u_Mnn, u_Mns = TrialFunction(self.Vb), TrialFunction(self.Vb), TrialFunction(self.Vb)

        vb = TestFunction(self.Vb)
        
        v_wt, w_omn, v_oms = TestFunction(self.Vb), TestFunction(self.Vb), TestFunction(self.Vb)
        

        # Mass matrices
        m_form = v_pw * al_pw * dx \
            + dot(v_pth, al_pth) * dx \
            + inner(v_qth, al_qth) * dx \
            + dot(v_qw, al_qw) * dx 
        
        M_pet = PETScMatrix()
        
        assemble(m_form, M_pet)

        self.M = csr_matrix(M_pet.mat().getValuesCSR()[::-1])
        
#        self.Mpw = self.M[self.dofs_Vpw, :][:, self.dofs_Vpw]
#        self.Mpth = self.M[self.dofs_Vpth, :][:, self.dofs_Vpth]
#        self.Mqth = self.M[self.dofs_Vqth, :][:, self.dofs_Vqth]
#        self.Mqw = self.M[self.dofs_Vqw, :][:, self.dofs_Vqw]
        
        # Stiffness matrices
        
        j_grad = dot(v_qw, grad(e_pw)) * dx
        j_gradIP = -dot(grad(v_pw), e_qw) * dx
        
        j_gradSym = inner(v_qth, gradSym(e_pth)) * dx
        j_gradSymIP = -inner(gradSym(v_pth), e_qth) * dx
        
        j_Id = dot(v_pth, e_qw) * dx
        j_IdIP = -dot(v_qw, e_pth) * dx
        
        j_allgrad = j_grad + j_gradIP + j_gradSym + j_gradSymIP + j_Id + j_IdIP

        J_pet = PETScMatrix()
        assemble(j_allgrad, J_pet)
        
        self.J = csr_matrix(J_pet.mat().getValuesCSR()[::-1])
        # Boundary matrices
        Mb_pet = PETScMatrix()
        assemble(dot(vb, ub) * ds, Mb_pet)
        self.Mb = csr_matrix(Mb_pet.mat().getValuesCSR()[::-1])[self.b_ind,:][:,self.b_ind]
        
        B_pet1, B_pet2, B_pet3 = PETScMatrix(), PETScMatrix(), PETScMatrix()
        assemble(v_pw * u_qn * ds, B_pet1)
        assemble(dot(v_pth, self.norext)*u_Mnn*ds, B_pet2)
        assemble(dot(v_pth, self.tanext)*u_Mns*ds, B_pet3)
        
        self.B1 = csr_matrix(B_pet1.mat().getValuesCSR()[::-1])[:,self.b_ind] 
        self.B2 = csr_matrix(B_pet2.mat().getValuesCSR()[::-1])[:,self.b_ind] 
        self.B3 = csr_matrix(B_pet3.mat().getValuesCSR()[::-1])[:,self.b_ind] 
        
        self.Bext = hstack([self.B1, self.B2, self.B3])
        
        self.R = csr_matrix((self.Nsys,self.Nsys))
        
        if 'internal' in self.damp :
            Ri_pet = PETScMatrix()
            assemble(self.eps* v_pw*e_pw*dx, Ri_pet)
            self.Ri = csr_matrix(Ri_pet.mat().getValuesCSR()[::-1])
            self.R += self.Ri
        
        if self.dynamic_R :
            def Rtime(t):
                return self.R * self.Rtime(t)
            self.Rdyn = Rtime 
       
        self.assembly = 1
        
        pass

    ## Method.
    # @param self The object pointer
    def Assembly_Mixed_BC(self):
        # DOFs of each boundary
        """
                   Gamma_4
                 —— —— —— ——
                |           | 
        Gamma_1 |           | Gamma_3
                |           |
                 —— —— —— —— 
                   Gamma_2 
        """
        
        self.Assembly()

        
        self.Gamma_1, self.Gamma_2, self.Gamma_3, self.Gamma_4 = [], [], [], [] 
        for i in range(len(self.coord_b_full)) :
            if np.abs(self.coord_b_full[i,0] - self.x0) <= DOLFIN_EPS :
                self.Gamma_1.append(i)
            if np.abs(self.coord_b_full[i,1] - self.y0) <= DOLFIN_EPS :
                self.Gamma_2.append(i)
            if np.abs(self.coord_b_full[i,0] - self.xL) <= DOLFIN_EPS :
                self.Gamma_3.append(i)
            if np.abs(self.coord_b_full[i,1] - self.yL) <= DOLFIN_EPS :
                self.Gamma_4.append(i)
        
        # Indexes of each boundary
        self.D_index = []
        self.N_index = []
        
        if 'G1' in self.Dir :
            self.D_index += self.Gamma_1
        elif 'G1' in self.Nor:
            self.N_index += self.Gamma_1

        if 'G2' in self.Dir :
            self.D_index += self.Gamma_2
        elif 'G2' in self.Nor :
            self.N_index += self.Gamma_2
            
        if 'G3' in self.Dir :
            self.D_index += self.Gamma_3
        elif 'G3' in self.Nor :
            self.N_index += self.Gamma_3
            
        if 'G4' in self.Dir :
            self.D_index += self.Gamma_4
        elif 'G4' in self.Nor :
            self.N_index += self.Gamma_4

        # Remove common indexes
        for i in self.D_index :
           if i in self.N_index :
               self.N_index.remove(i)

                
        self.D_index = list(np.unique(np.array(self.D_index)))     
        self.N_index = list(np.unique(np.array(self.N_index)))      

        self.Nb_D, self.Nb_N = len(self.D_index), len(self.N_index)
        
        # New Boundary Matrices
        v = TestFunction(self.V)
        v_pw, v_pth, v_qth, v_qw = split(v)
        
        e = TrialFunction(self.V)
        e_pw, e_pth, e_qth, e_qw = split(e)
        
        v_qth = as_tensor([[v_qth[0], v_qth[1]],
                            [v_qth[1], v_qth[2]]
                           ])
        
        e_qth = as_tensor([[e_qth[0], e_qth[1]],
                            [e_qth[1], e_qth[2]]
                           ])
        
        vb = TestFunction(self.Vb)
        ub = TrialFunction(self.Vb)
        
        v_wt, v_omn, v_oms = TestFunction(self.Vb), TestFunction(self.Vb), TestFunction(self.Vb)
        u_qn, u_Mnn, u_Mns = TrialFunction(self.Vb), TrialFunction(self.Vb), TrialFunction(self.Vb)
        
        Mb1_N_pet = PETScMatrix()
        
        assemble( vb* ub * ds, Mb1_N_pet)
        Mb1_N = csr_matrix(Mb1_N_pet.mat().getValuesCSR()[::-1])[self.N_index, :][:, self.N_index]
        
        self.Mb_N = block_diag((Mb1_N, Mb1_N, Mb1_N))
        
        Mb1_D_pet = PETScMatrix()
        
        assemble( vb* ub * ds, Mb1_D_pet)
        Mb1_D = csr_matrix(Mb1_D_pet.mat().getValuesCSR()[::-1])[self.D_index, :][:, self.D_index]
        
        self.Mb_D = csr_matrix(block_diag((Mb1_D, Mb1_D, Mb1_D)))

        B1_N_pet, B2_N_pet, B3_N_pet = PETScMatrix(), PETScMatrix(), PETScMatrix()

        assemble(v_pw * u_qn * ds, B1_N_pet)
        self.B1_N = csr_matrix(B1_N_pet.mat().getValuesCSR()[::-1])[:, self.N_index]

        assemble(dot(v_pth, self.norext)*u_Mnn *ds, B2_N_pet)
        self.B2_N = csr_matrix(B2_N_pet.mat().getValuesCSR()[::-1])[:, self.N_index]
        
        assemble(dot(v_pth, self.tanext)*u_Mns *ds, B3_N_pet)
        self.B3_N = csr_matrix(B3_N_pet.mat().getValuesCSR()[::-1])[:, self.N_index]
        
        self.B_Next = hstack([self.B1_N, self.B2_N, self.B3_N])
        
        B1_D_pet, B2_D_pet, B3_D_pet = PETScMatrix(), PETScMatrix(), PETScMatrix()

        assemble(v_pw * u_qn * ds, B1_D_pet)
        self.B1_D = csr_matrix(B1_D_pet.mat().getValuesCSR()[::-1])[:, self.D_index]
        
        assemble(dot(v_pth, self.norext)*u_Mnn *ds, B2_D_pet)
        self.B2_D = csr_matrix(B2_D_pet.mat().getValuesCSR()[::-1])[:, self.D_index]
        
        assemble(dot(v_pth, self.tanext)*u_Mns *ds, B3_D_pet)
        self.B3_D = csr_matrix(B3_D_pet.mat().getValuesCSR()[::-1])[:, self.D_index]

        self.B_Dext = hstack([self.B1_D, self.B2_D, self.B3_D])

        # Lagrange multiplier initial data
        
        B1_normal_D_pet, B2_normal_D_pet, B3_normal_D_pet = PETScMatrix(),\
        PETScMatrix(), PETScMatrix()

        assemble(v_wt * dot(e_qw, self.norext) * ds, B1_normal_D_pet)
        self.B1_normal_D = csr_matrix(B1_normal_D_pet.mat().getValuesCSR()[::-1])[self.D_index, :]

        assemble(v_omn*inner(e_qth, outer(self.norext, self.norext))*ds, B2_normal_D_pet)                                      
        self.B2_normal_D = csr_matrix(B2_normal_D_pet.mat().getValuesCSR()[::-1])[self.D_index, :]

        assemble(v_oms*inner(e_qth, outer(self.norext, self.tanext))*ds, B3_normal_D_pet)                                      
        self.B3_normal_D = csr_matrix(B3_normal_D_pet.mat().getValuesCSR()[::-1])[self.D_index, :]
        
        self.B_normal_D = vstack([self.B1_normal_D, self.B2_normal_D, self.B3_normal_D])

        self.assembly = 1
        
        print('Nb_D=', self.Nb_D, ',\t Nb_N=', self.Nb_N)
        print('DOFsysDAE=', self.Nsys + self.Nb_D)
        print('DAE system: OK')
        print(40*'-', '\n')
        
        pass
    
    ## Method.
    #  @param self The object pointer. 
    def Simplectic_Splitting(self):
        
        dofs_Vp, dofs_Vq = self.dofs_Vp, self.dofs_Vq
        
        Mp = self.M[dofs_Vp, :][:, dofs_Vp]
        Mq = self.M[dofs_Vq, :][:, dofs_Vq]
        
        Dp = self.J[:,dofs_Vq][dofs_Vp, :]
        Dq = self.J[:,dofs_Vp][dofs_Vq, :]
        
        self.Rp = self.R[dofs_Vp, :][:, dofs_Vp]
        self.Rq = self.R[dofs_Vq, :][:, dofs_Vq]

        self.Bp = self.B[dofs_Vp, :]
        self.Bq = self.B[dofs_Vq, :]
        
        return Mp, Mq, Dp, Dq, Rp, Rq, Bp, Bq
    

        

    ## Method.
    #  @param self The object pointer.           
    def Set_Time_Setting(self, time_step, tf=None):
        """
        Specify the parameters related to the time integration
        string_solver specifies the method to be used.
        """
        if tf is not None: 
            self.tfinal = tf 
        self.dt     = time_step
        self.Nt     = int( np.floor(self.tfinal/self.dt) )
        self.tspan  = np.linspace(0,self.tfinal,self.Nt+1)
        
        self.set_time_setting = 1
        
        pass
    #%%

    
    #%% SPECTRUM
    ## Method.
    # @param self The object pointer.
    def Spectrum(self, k=10, xl=None, figsize=(8,6.5), method='sparse', **kwargs):
        if method == 'sparse' :
#            try :   
#                JR = bmat( [ [JR, csc_matrix(self.B_Dext)], [-csc_matrix(self.B_Dext).T, None] ] )
#        #            M_sparse = block_diag((M_sparse, csc_matrix((self.Nb_D,self.Nb_D))))
#                M_sparse = csc_matrix(block_diag((M_sparse, csc_matrix(self.Mb_D) )))
#            except :    pass
            eigen_values, self.eigenv_vectors = eigs(self.J -self.R, M=self.M, k=k)
        
        if method == 'dense' :
            J_dense = self.J.todense()
            R_dense = self.R.todense()
            M_dense = self.M.todense()
            eigen_values = linalg.eigvals(J_dense - R_dense, M_dense)
            k = self.Nsys

        self.eigen_values_Real, self.eigen_values_Imag = eigen_values.real, eigen_values.imag

        plt.figure(figsize=figsize)
        plt.plot(self.eigen_values_Real, self.eigen_values_Imag, '+', **kwargs)
        plt.xlabel('real part')
        plt.ylabel('imaginary part')
        plt.title('eigs($J-R$, $M$) with '+str(k)+' first eigenvalues')
        plt.xlim(xl)
        plt.savefig('spectrum.png')
        return self.eigen_values_Real, self.eigen_values_Imag
    #%%
    
    
    #%% PROJECTONS

    ## Method.
    #  @param self The object pointer.
    def Project_Boundary_Control(self):
        """
        Project boundary controlon the FE spaces 
        """        
        assert self.generate_mesh == 1, "The finite element mesh must be generated first"
    
        assert self.set_finite_elements_spaces == 1, \
                "The FE approximation spaces must be selected first"
       
        if self.set_mixed_boundaries == 0 : 
            self.Ub1_sp0 = interpolate(self.Ub1_sp0_Expression, self.Vb).vector()[self.b_ind]
            self.Ub2_sp0 = interpolate(self.Ub2_sp0_Expression, self.Vb).vector()[self.b_ind]
            self.Ub3_sp0 = interpolate(self.Ub3_sp0_Expression, self.Vb).vector()[self.b_ind]
            
            self.Ub_sp0 = lambda t : np.concatenate((self.Ub1_sp0 * self.Ub1_tm0(t),
                                             self.Ub2_sp0 * self.Ub2_tm0(t),
                                             self.Ub3_sp0 * self.Ub3_tm0(t),), axis=0)
           
            self.Ub1_sp1 = interpolate(self.Ub1_sp1_Expression, self.Vb).vector()[self.b_ind]
            self.Ub2_sp1 = interpolate(self.Ub2_sp1_Expression, self.Vb).vector()[self.b_ind]
            self.Ub3_sp1 = interpolate(self.Ub3_sp1_Expression, self.Vb).vector()[self.b_ind]
            
            self.Ub_sp1 = np.concatenate((self.Ub1_sp1, self.Ub2_sp1, self.Ub3_sp1), axis=0)

            self.Ub_tm1 = lambda t : np.concatenate((np.ones(self.Nb) * self.Ub1_tm1(t),
                                             np.ones(self.Nb) * self.Ub2_tm1(t),
                                             np.ones(self.Nb) * self.Ub3_tm1(t)), axis=0)
            
            self.Ub = lambda t : self.Ub_sp0(t) + self.Ub_sp1 + self.Ub_tm1(t)
        
        if self.set_mixed_boundaries == 1 and self.set_dirichlet_boundary_control==1 \
        and self.set_normal_boundary_control==1:
            self.Ub1_sp0_D = interpolate(self.Ub1_sp0_D_Expression, self.Vb).vector()[self.D_index]
            self.Ub2_sp0_D = interpolate(self.Ub2_sp0_D_Expression, self.Vb).vector()[self.D_index]
            self.Ub3_sp0_D = interpolate(self.Ub3_sp0_D_Expression, self.Vb).vector()[self.D_index]
            
            self.Ub_sp0_D = lambda t : np.concatenate((self.Ub1_sp0_D * self.Ub1_tm0_D(t),
                                             self.Ub2_sp0_D * self.Ub2_tm0_D(t),
                                             self.Ub3_sp0_D * self.Ub3_tm0_D(t)), axis=0)
            
            self.Ub1_sp1_D = interpolate(self.Ub1_sp1_D_Expression, self.Vb).vector()[self.D_index]
            self.Ub2_sp1_D = interpolate(self.Ub2_sp1_D_Expression, self.Vb).vector()[self.D_index]
            self.Ub3_sp1_D = interpolate(self.Ub3_sp1_D_Expression, self.Vb).vector()[self.D_index]
            
            self.Ub_sp1_D = np.concatenate((self.Ub1_sp1_D, self.Ub2_sp1_D,
                                             self.Ub3_sp1_D), axis=0)
            
            self.Ub_tm1_D = lambda t : np.concatenate((np.ones(self.Nb_D) * self.Ub1_tm1(t),
                                             np.ones(self.Nb_D) * self.Ub2_tm1(t),
                                             np.ones(self.Nb_D) * self.Ub3_tm1(t)), axis=0)
            
            self.Ub_D = lambda t : self.Ub_sp0_D(t) + self.Ub_sp1_D + self.Ub_tm1_D(t)
            
            self.Ub_sp0_D_dir = lambda t : np.concatenate((self.Ub1_sp0_D * self.Ub1_tm0_D_dir(t),
                                             self.Ub2_sp0_D * self.Ub2_tm0_D_dir(t),
                                             self.Ub3_sp0_D * self.Ub3_tm0_D_dir(t)), axis=0)
            
            self.Ub_tm1_D_dir = lambda t : np.concatenate((np.ones(self.Nb_D) * self.Ub1_tm1_D_dir(t),
                                             np.ones(self.Nb_D) * self.Ub2_tm1_D_dir(t),
                                             np.ones(self.Nb_D) * self.Ub3_tm1_D_dir(t)), axis=0)
            
            self.Ub_D_dir = lambda t : self.Ub_sp0_D_dir(t) + self.Ub_tm1_D_dir(t)

            self.Ub1_sp0_N = interpolate(self.Ub1_sp0_N_Expression, self.Vb).vector()[self.N_index]
            self.Ub2_sp0_N = interpolate(self.Ub2_sp0_N_Expression, self.Vb).vector()[self.N_index]
            self.Ub3_sp0_N = interpolate(self.Ub3_sp0_N_Expression, self.Vb).vector()[self.N_index]
            
            self.Ub_sp0_N = lambda t : np.concatenate((self.Ub1_sp0_N * self.Ub1_tm0_N(t),
                                             self.Ub2_sp0_N * self.Ub2_tm0_N(t),
                                             self.Ub3_sp0_N * self.Ub3_tm0_N(t)), axis=0)
            
            self.Ub1_sp1_N = interpolate(self.Ub1_sp1_N_Expression, self.Vb).vector()[self.N_index]
            self.Ub2_sp1_N = interpolate(self.Ub2_sp1_N_Expression, self.Vb).vector()[self.N_index]
            self.Ub3_sp1_N = interpolate(self.Ub3_sp1_N_Expression, self.Vb).vector()[self.N_index]
            
            self.Ub_sp1_N = np.concatenate((self.Ub1_sp1_N, self.Ub2_sp1_N, self.Ub3_sp1_N), axis=0)
            
            self.Ub_tm1_N = lambda t : np.concatenate((np.ones(self.Nb_N) * self.Ub1_tm1_N(t),
                                             np.ones(self.Nb_N) * self.Ub2_tm1_N(t),
                                             np.ones(self.Nb_N) * self.Ub3_tm1_N(t)), axis=0)
            
            self.Ub_N = lambda t : self.Ub_sp0_N(t) + self.Ub_sp1_N + self.Ub_tm1_N(t)            
        elif self.set_dirichlet_boundary_control ==0:
            print('No Dirichlet control set')
        elif self.set_normal_boundary_control == 0:
            print('No Neumann control set')

        self.project_boundary_control = 1
        
        pass

    ## Method.
    #  @param self The object pointer. 
    def Project_Initial_Data(self) :
        """
        Project initial data on the FE spaces 
        """
        assert self.generate_mesh == 1, "The finite element mesh must be generated first"
    
        assert self.set_finite_elements_spaces == 1, \
                "The FE approximation spaces must be selected first"
       
        if not self.init_by_vector :
            
            self.W0  = interpolate(self.W_0, self.Vpw.collapse()).vector()[:]
            
            Apw0 = interpolate(self.Apw_0, self.Vpw.collapse()).vector()[:]
            Apth0 = interpolate(self.Apth_0, self.Vpth.collapse()).vector()[:]
            Aqth0 = interpolate(self.Aqth_0, self.Vqth.collapse()).vector()[:]
            Aqw0 = interpolate(self.Aqw_0, self.Vqw.collapse()).vector()[:]
            
            self.A0  = np.zeros((self.Nsys, ))
            self.A0[self.dofs_Vpw] = Apw0
            self.A0[self.dofs_Vpth] = Apth0
            self.A0[self.dofs_Vqth] = Aqth0
            self.A0[self.dofs_Vqw] = Aqw0
            
        if self.project_boundary_control == 1 : print('Project BC: OK')
        print('Project init data: OK')
        print(40*'-', '\n')
        self.project_initial_data = 1
        
        pass
    #%%   
    
    
    #%% NUMERICAL APPROXIMATION SCHEMES  
    
    ## Method.
    #  @param self The object pointer.      
    def Time_Integration(self, string_mode, **kwargs):
        """
        Wrapper method for the time integration
        """
        
        if string_mode == 'DAE:Assimulo':
            A, Ham = self.Integration_DAE_Assimulo(**kwargs)
            done   = 1
            
        if string_mode == 'DAE:RK4Augmented':
            A, Ham = self.Integration_DAE_RK4_Augmented(**kwargs)
            done   = 1        

        if string_mode == 'DAE:SV2Augmented':
            A, Ham = self.Integration_DAE_SV2_Augmented(**kwargs)
            done   = 1        
        
        if string_mode == 'ODE:SV': 
            A, Ham = self.Integration_SV()
            done   = 1
            
        if string_mode == 'ODE:CN2': 
            A, Ham = self.Integration_CN2()
            done   = 1
        
        if string_mode == 'ODE:RK4': 
            A, Ham = self.Integration_RK4(**kwargs)
            done   = 1  
            
        if string_mode == 'ODE:Scipy': 
            A, Ham = self.Integration_Scipy(**kwargs)
            done   = 1     
                 
        if string_mode == 'ODE:Assimulo': 
            A, Ham = self.Integration_Assimulo(**kwargs)
            done   = 1  
           
        assert done == 1, "Unknown time discretization method in Time_integration"
        
        print(40*'-', '\n')
        
        return A, Ham
    
    ## Method.
    # @param self The object pointer.
    def Sparsity_solvers(self):
        self.M_sparse = csc_matrix(self.M)
        self.JR = csc_matrix(self.J - self.R)
        ' to be continued '
        pass
    
    
    ## Method.
    #  @param self The object pointer.       
    def Integration_SV(self):
        """
        $\theta$-scheme for the numerical integration of the ODE system
        """
        assert self.set_time_setting == 1, 'Time discretization must be specified first'
        
        print('Factorization and inversion, this may take a while ...')
        
        Np = self.Np
        Nq = self.Nq
        
        Mp, Mq, Dp, Dq, Rp, Rq, Bp, Bq = self.Simplectic_Splitting()
        Mpq = block_diag(Mp, Mq)

        facto_p   = umfpack.UmfpackLU(Mp)       
        facto_q   = umfpack.UmfpackLU(Mq)
        
        print('Factorization: OK')

        Sys_p, Sys_R_p, Sys_ctrl_p = facto_p.solve(Dp), facto_p.solve(Rp),\
                                        facto_p.solve(Bp)        
        Sys_q, Sys_R_q, Sys_ctrl_q = facto_q.solve(Dq), facto_q.solve(Rq),\
                                        facto_q.solve(Bq)
                                        
        print('Inversion: OK \n')
        
        # Solution and Hamiltonian versus time
        A   = np.zeros( (self.Nsys, self.Nt+1) )


        # Initialization
        A0p = self.A0[self.dofs_Vp]
        A0q = self.A0[self.dofs_Vq]
        
        A[:,0] = np.concatenate((A0p, A0q))
      
        # Time loop
        dt = self.dt
        Ub = self.Ub
        t = self.tspan
        for n in range(self.Nt):   
            
            Ap_n = A[:Np, n]
            Aq_n = A[Np:, n]

            
            Aqq = Aq_n + dt/2 * (Sys_q @ Ap_n - Sys_R_q @ Aq_n + Sys_ctrl_q @ (Ub(t[n+1]) + Ub(t[n]))/2)
            Apn = Ap_n + dt * (Sys_p @ Aqq - Sys_R_p @ Ap_n + Sys_ctrl_p @ (Ub(t[n+1]) + Ub(t[n]))/2)
            Aqn = Aqq + dt/2 * (Sys_q @ Apn - Sys_R_q @ Aqq + Sys_ctrl_q @ (Ub(t[n+1]) + Ub(t[n]))/2)
            
            A[:Np, n+1] = Apn

            A[Np:, n+1] = Aqn
                    
            # Progress bar
            perct = int(n/(self.Nt-1) * 100)  
            bar = ('Time-stepping SV : |' + '#' * int(perct/2) + ' ' + str(perct) + '%' + '|')
            sys.stdout.write('\r' + bar)
        print('\n integration completed \n')
        
        Ham = np.array([ 1/2 * A[:,n] @ Mpq @ A[:,n] for n in range(self.Nt+1) ])

        A_p = A[:Np, :]
        A_q = A[Np:, :]

        A_reordered = np.zeros_like(A)
        A_reordered[self.dofs_Vp, :] = A_p
        A_reordered[self.dofs_Vq, :] = A_q
        
        return A_reordered,  Ham
    
    ## Method.
    #  @param self The object pointer.       
    def Integration_CN2(self, theta=0.5):
        """
        $\theta$-scheme for the numerical integration of the ODE system
        """
        assert self.set_time_setting == 1, 'Time discretization must be specified first'
        self.M_sparse = csc_matrix(self.M)
        
        # Solution and Hamiltonian versus time
        A   = np.zeros( (self.Nsys, self.Nt+1) )

        # Initialization
        A[:,0] = self.A0

        # Inversion of matrix system
        if not self.dynamic_R :
            print('Factorization and inversion, this may take a while ...')
            facto = umfpack.UmfpackLU( csc_matrix( self.M - self.dt*theta * (self.J-self.R) ) )
            print('Factorization: OK')
            Sys = facto.solve(self.M + self.dt * (1-theta) * (self.J-self.R))          
            Sys_ctrl = facto.solve(self.dt* self.Bext)
            print('Inversion: OK \n')
        
        # Time loop
        for n in range(self.Nt):   
            if self.dynamic_R:
                print('Factorization and inversion, this may take a while ...')
                facto = umfpack.UmfpackLU( csc_matrix( self.M - self.dt*theta * (self.J-self.Rdyn(self.tspan[n])) ) )
                print('Factorization: OK')
                Sys = facto.solve(self.M + self.dt * (1-theta) * (self.J-self.Rdyn(self.tspan[n])))          
                Sys_ctrl = facto.solve(self.dt* self.Bext)
                print('Inversion: OK \n')
            A[:,n+1] = Sys @ A[:,n] + Sys_ctrl @ self.Ub(self.tspan[n+1])
            
            # Progress bar
            perct = int(n/(self.Nt-1) * 100)  
            bar = ('Time-stepping CN2 : |' + '#' * int(perct/2) + ' ' + str(perct) + '%' + '|')
            sys.stdout.write('\r' + bar)

        Ham = np.array([ 1/2 * A[:,n] @ self.M_sparse @ A[:,n] for n in range(self.Nt+1) ])

        print('\n integration completed \n')
        
        return A, Ham
    
    ## Method.
    #  @param self The object pointer.       
    def Integration_RK4(self):
        """
        4th order Runge-Kutta scheme for the numerical integration of the ODE system
        """
        assert self.set_time_setting == 1, 'Time discretization must be specified first'
        
        tspan = self.tspan
        dt = self.dt
        
        print('Factorization and inversion, this may take a while ...')
        self.M_sparse = csc_matrix(self.M)
        self.JR_sparse = csc_matrix(self.J-self.R)
        facto = umfpack.UmfpackLU(self.M_sparse)
                
        print('Factorization: OK')
        if not self.dynamic_R :
            Sys = facto.solve(self.JR_sparse)
        Sys_ctrl = facto.solve(self.Bext) 
        print('Inversion: OK \n')
        
        # Sys_nondis = facto.solve(self.J)
        def dif_func(t,y):
            if self.dynamic_R :
                Sys_loc = facto.solve(self.J-self.Rdyn(t))
                return Sys_loc @ y + Sys_ctrl @ self.Ub(t)
            else :
                return Sys @ y + Sys_ctrl @ self.Ub(t)
        
        A = np.zeros((self.Nsys, self.Nt+1))
        Ham = np.zeros(self.Nt+1)
        
        A[:,0] = self.A0
        Ham[0] = 1/2 * A[:,0] @ self.M_sparse @  A[:,0] 
        
        for k in range(self.Nt):
            k1 = dif_func(tspan[k], A[:,k])
            k2 = dif_func(tspan[k] + dt/2, A[:,k] + dt/2 * k1)
            k3 = dif_func(tspan[k] + dt/2, A[:,k] + dt/2 * k2)
            k4 = dif_func(tspan[k] + dt, A[:,k] + dt * k3)
            
            A[:,k+1] = A[:,k] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            Ham[k+1] = 1/2 * A[:,k+1] @ self.M_sparse @ A[:,k+1]
        
            # Progress bar
            perct = int(k/(self.Nt-1) * 100)  
            bar = ('Time-stepping RK4 : |' + '#' * int(perct/2) + ' ' + str(perct) + '%' + '|')
            sys.stdout.write('\r' + bar)
        print('\n integration completed \n')
        
        return A, Ham

    ## Method.
    #  @param self The object pointer.    
    def Integration_Scipy(self, **kwargs):
        """
        Perform time integration for ODEs with the scipy.integrate.IVP package
        """
        assert self.set_time_setting == 1, 'Time discretization must be specified first'

        self.M_sparse = csc_matrix(self.M)
        facto = umfpack.UmfpackLU(self.M_sparse)     
        
        # Definition of the rhs function required in scipy integration
        Sys = facto.solve(csc_matrix(self.J-self.R))
        Sys_ctrl = facto.solve(csc_matrix(self.Bext))
        def rhs(t,y):
            """
            Definition of the rhs function required in the ODE part of IVP
            """ 
            if not self.dynamic_R:  
                return Sys @ y + Sys_ctrl @ self.Ub(t) 
            else:
                return (self.J-self.Rdyn(t)) @ y + Sys_ctrl @ self.Ub(t)            
            
        print('ODE Integration using scipy.integrate built-in functions:')

        ivp_ode    = integrate.solve_ivp(rhs, (self.tinit,self.tfinal), self.A0,\
                                         **kwargs, t_eval=self.tspan, atol=1.e-6)   
        
        A          = ivp_ode.y
        self.Nt    = len(self.tspan) - 1
        
        print("Scipy: Number of evaluations of the right-hand side ",ivp_ode.nfev)
        print("Scipy: Number of evaluations of the Jacobian ",ivp_ode.njev)
        print("Scipy: Number of LU decompositions ",ivp_ode.nlu)
                        
        # Hamiltonian 
        Ham        = np.zeros(self.Nt+1)
        
        for k in range(self.Nt+1):
            Ham[k] = 1/2 * A[:,k] @ self.M_sparse.dot(A[:,k])

        self.Ham = Ham
    
        return A, Ham

        ## Method.
    #  @param self The object pointer.    
    def Integration_Assimulo(self, **kwargs):
        """
        Perform time integration for ODEs with the assimulo package
        """
        assert self.set_time_setting == 1, 'Time discretization must be specified first'

        self.M_sparse = csc_matrix(self.M)
        facto = factorized(self.M_sparse)    
        my_jac  = csr_matrix(facto((self.J - self.R)))
                
        # Definition of the rhs function required in scipy assimulo
        Sys = facto(self.J-self.R)
        Sys_ctrl = facto(self.Bext)
        def rhs(t,y):
            """
            Definition of the rhs function required in the ODE part of IVP
            """ 
            if not self.dynamic_R:  
                return Sys @ y + Sys_ctrl @ self.Ub(t) 
            else:
                return (self.J-self.Rdyn(t)) @ y + Sys_ctrl @ self.Ub(t)
            
        def jacobian(t,y):
            """
            Jacobian matrix related to the ODE
            """
            if not self.dynamic_R:
                return facto(self.J - self.R)        
            else:
                return facto(self.J - self.Rdyn(t))    
        
        def jacv(t,y,fy,v):
            """
            Jacobian matrix-vector product related to the ODE formulation
            """
            if not self.dynamic_R:
                z = (self.J - self.R) @ v        
            else:
                z = (self.J - self.Rdyn(t)) @ v    
            return facto(z)
           
        print('ODE Integration using assimulo built-in functions:')

#
# https://jmodelica.org/assimulo/_modules/assimulo/examples/cvode_with_preconditioning.html#run_example
#
        
        model                     = Explicit_Problem(rhs,self.A0,self.tinit)
        model.jac                 = jacobian
        model.jacv                = jacv
        sim                       = CVode(model,**kwargs)
        sim.atol                  = 1e-3 
        sim.rtol                  = 1e-3 
        sim.linear_solver         = 'SPGMR' 
        sim.maxord                = 3
        #sim.usejac                = True
        #sim                       = RungeKutta34(model,**kwargs)
        time_span, ODE_solution   = sim.simulate(self.tfinal)
        
        A = ODE_solution.transpose()
        
        # Hamiltonian
        self.Nt    = A.shape[1]-1
        self.tspan = np.array(time_span)
        
        Ham = np.zeros(self.Nt+1)
        
        for k in range(self.Nt+1):
            Ham[k] = 1/2 * A[:,k].T @ self.M_sparse.dot(A[:,k])

        self.Ham = Ham
    
        return A, Ham
    
    ## Method.
    #  @param self The object pointer.    
    def Integration_DAE_Assimulo(self, **kwargs):
        Nsys, Nb_D = self.Nsys, self.Nb_D
        Ub_N, Ub_D, Ub_D_dir = self.Ub_N, self.Ub_D, self.Ub_D_dir
                
        self.M_sparse = csc_matrix(self.M)
        facto = umfpack.UmfpackLU(self.M_sparse)
        Sys = facto.solve(csc_matrix(self.J - self.R))
        Sys_Ctrl_N = facto.solve(csc_matrix(self.B_Next))
        Sys_Ctrl_D = facto.solve(csc_matrix(self.B_Dext))
        try : Sys_Ctrl_DT = umfpack.spsolve(csc_matrix(self.Mb_D), csc_matrix(self.B_Dext).T)
        except : Sys_Ctrl_DT = np.empty((self.Nb_D,self.Nsys))

        def PHDAE(t,y,yd):
#            res_0 = np.zeros(Nsys)
#            res_1 = np.zeros(Nb_D)
            
            res_0 = yd[:Nsys] - Sys @ y[:Nsys] - Sys_Ctrl_N @ Ub_N(t) - Sys_Ctrl_D @ y[Nsys:] 
            res_1 = Ub_D_dir(t) - Sys_Ctrl_DT @ yd[:Nsys] 
            
             
            return np.concatenate((res_0, res_1))
        
        # The initial conditons
        
        y0 =  np.block([self.A0, spsolve(self.Mb_D, self.B_normal_D @ self.A0 ) ])# Initial conditions
        yd0 = np.zeros(self.Nsys+Nb_D*3) # Initial conditions
        
            
        def handle_result(solver, t ,y, yd):
            global order
            order.append(solver.get_last_order())
            
            solver.t_sol.extend([t])
            solver.y_sol.extend([y])
            solver.yd_sol.extend([yd])   

        # Create an Assimulo implicit problem
        imp_mod = Implicit_Problem(PHDAE, y0, yd0, name='PHDAE')
           
        # Set the algebraic components
        imp_mod.algvar = list( np.block([np.ones(Nsys), np.zeros(3*Nb_D)]) ) 
        
        # Create an Assimulo implicit solver (IDA)
        imp_sim = IDA(imp_mod) #Create a IDA solver
         
        # Sets the paramters
        imp_sim.atol = 1e-6 #Default 1e-6
        imp_sim.rtol = 1e-6 #Default 1e-6
        imp_sim.suppress_alg = True #Suppres the algebraic variables on the error test
        imp_sim.report_continuously = True
             
        # Let Sundials find consistent initial conditions by use of 'IDA_YA_YDP_INIT'
        imp_sim.make_consistent('IDA_YA_YDP_INIT')
         
        # Simulate
        t, y, yd = imp_sim.simulate(self.tfinal, self.tinit, self.tspan ) 
        A = y[:,:self.Nsys].T
        
        # Hamiltonian
        Ham = np.array([1/2 * A[:,i] @ self.M @ A[:,i] for i in range(self.Nt+1)])
        
        return A, Ham

    ## Method.
    #  @param self The object pointer.  
    def Integration_DAE_RK4_Augmented(self):
        self.M_sparse      = csc_matrix(self.M)
        self.JR            = csc_matrix(self.J - self.R)
        self.B_Dext_sparse = csc_matrix(self.B_Dext)
        self.B_Next_sparse = csc_matrix(self.B_Next)
        
        M_sparseLU = umfpack.UmfpackLU(self.M_sparse)
        
        Sys        = M_sparseLU.solve(self.JR)
        Sys_Ctrl_N = M_sparseLU.solve(self.B_Next_sparse)
        
        BDMinvBDT = self.B_Dext_sparse.T @ M_sparseLU.solve(self.B_Dext_sparse) 
        
        prefix         = M_sparseLU.solve(self.B_Dext_sparse)
        Sys_Ctrl_D     = prefix @ spsolve(BDMinvBDT, self.Mb_D)
        Sys_Aug        = - prefix @ spsolve(BDMinvBDT, self.B_Dext.T) @ Sys
        Sys_Ctrl_N_Aug = - prefix @ spsolve(BDMinvBDT, self.B_Dext.T) @ Sys_Ctrl_N
        
        Sys_AUG        = Sys + Sys_Aug
        Sys_Ctrl_N_AUG = Sys_Ctrl_N + Sys_Ctrl_N_Aug
        
        def dif_aug_func(t,y):
            return Sys_AUG @ y + Sys_Ctrl_N_AUG @ self.Ub_N(t) + Sys_Ctrl_D @ self.Ub_D_dir(t)
        
        A = np.zeros((self.Nsys, self.Nt+1))
        
        A[:,0] = self.A0
        
        t  = self.tspan
        dt = self.dt
        
        for k in range(self.Nt):
            k1 = dif_aug_func(t[k]       , A[:,k])
            k2 = dif_aug_func(t[k] + dt/2, A[:,k] + dt/2 * k1)
            k3 = dif_aug_func(t[k] + dt/2, A[:,k] + dt/2 * k2)
            k4 = dif_aug_func(t[k] + dt  , A[:,k] + dt * k3)
            
            A[:,k+1] = A[:,k] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        
            # Progress bar
            perct = int(k/(self.Nt-1) * 100)  
            bar = ('Time-stepping RK4Augmented : |' + '#' * int(perct/2) + ' ' + str(perct) + '%' + '|')
            sys.stdout.write('\r' + bar)

        Ham = np.array([1/2 * A[:,k] @ self.M_sparse @ A[:,k] for k in range(self.Nt+1)])

        print('\n integration completed \n')
        
        return A, Ham

    ## Method.
    #  @param self The object pointer.  
    def Integration_DAE_SV2_Augmented(self):
        self.M_sparse      = csc_matrix(self.M)
        self.JR            = csc_matrix(self.J - self.R)
        self.B_Dext_sparse = csc_matrix(self.B_Dext)
        self.B_Next_sparse = csc_matrix(self.B_Next)
        
        M_sparseLU = umfpack.UmfpackLU(self.M_sparse)
        
        Sys        = M_sparseLU.solve(self.JR)
        Sys_Ctrl_N = M_sparseLU.solve(self.B_Next_sparse)
        
        BDMinvBDT = self.B_Dext_sparse.T @ M_sparseLU.solve(self.B_Dext_sparse) 
        
        prefix         = M_sparseLU.solve(self.B_Dext_sparse)
        Sys_Ctrl_D     = prefix @ spsolve(BDMinvBDT, self.Mb_D)
        Sys_Aug        = - prefix @ spsolve(BDMinvBDT, self.B_Dext.T) @ Sys
        Sys_Ctrl_N_Aug = - prefix @ spsolve(BDMinvBDT, self.B_Dext.T) @ Sys_Ctrl_N
        
        Sys_AUG        = Sys + Sys_Aug
        Sys_Ctrl_N_AUG = Sys_Ctrl_N + Sys_Ctrl_N_Aug
        
        def dif_aug_func(t,y):
            return Sys_AUG @ y + Sys_Ctrl_N_AUG @ self.Ub_N(t) + Sys_Ctrl_D @ self.Ub_D_dir(t)
        
        Nq = self.Nq
        Np = self.Np
        
        dofs_Vp = self.dofs_Vp
        dofs_Vq = self.dofs_Vq
        
        Sys_AUG_qq          = Sys_AUG[dofs_Vq, :][:, dofs_Vq]
        Sys_AUG_qp          = Sys_AUG[dofs_Vq, :][:, dofs_Vp]
        Sys_AUG_pq          = Sys_AUG[dofs_Vp, :][:, dofs_Vq]
        Sys_AUG_pp          = Sys_AUG[dofs_Vp, :][:, dofs_Vp]
        
        Sys_Ctrl_N_AUG_q    = Sys_Ctrl_N_AUG[dofs_Vq]
        Sys_Ctrl_N_AUG_p    = Sys_Ctrl_N_AUG[dofs_Vp]        
        
        Sys_Ctrl_D_q        = Sys_Ctrl_D[dofs_Vq]
        Sys_Ctrl_D_p        = Sys_Ctrl_D[dofs_Vp]
        
        A_pq = np.zeros((self.Nsys, self.Nt+1))
        
        A0p = self.A0[self.dofs_Vp]
        A0q = self.A0[self.dofs_Vq]
        
        A_pq[:,0] = np.concatenate((A0p, A0q))
        
        
        t  = self.tspan
        dt = self.dt
        
        for k in range(self.Nt):
            
            Ap_k = A_pq[:Np, k]
            Aq_k = A_pq[Np:, k]
            
            App         = Ap_k + dt/2 * (Sys_AUG_pq @ Aq_k + Sys_AUG_pp @ Ap_k \
                                         +Sys_Ctrl_N_AUG_p @ self.Ub_N(t[k]) \
                                         + Sys_Ctrl_D_p @ self.Ub_D_dir(t[k]))   
            Aqn = Aq_k + dt * (Sys_AUG_qq @ Ap_k + Sys_AUG_qp @ App \
                                         + Sys_Ctrl_N_AUG_q @ self.Ub_N(t[k]) \
                                         + Sys_Ctrl_D_q @ self.Ub_D_dir(t[k])) 
            Apn = App + dt/2 * (Sys_AUG_pq @ Aqn + Sys_AUG_pp @ App + Sys_Ctrl_N_AUG_p @ self.Ub_N(t[k]) + Sys_Ctrl_D_p @ self.Ub_D_dir(t[k]))  
        
            A_pq[:Np, n+1] = Apn
            A_pq[Np:, n+1] = Aqn
            # Progress bar
            perct = int(k/(self.Nt-1) * 100)  
            bar = ('Time-stepping SV2Augmented : |' + '#' * int(perct/2) + ' ' + str(perct) + '%' + '|')
            sys.stdout.write('\r' + bar)
            
        A_p = A_pq[:Np, :]
        A_q = A_pq[Np:, :]

        A = np.zeros_like(A_pq)
        A[self.dofs_Vp, :] = A_p
        A[self.dofs_Vq, :] = A_q
        
        Ham = np.array([1/2 * A[:,k] @ self.M_sparse @ A[:,k] for k in range(self.Nt+1)])

        print('\n intergation completed \n')
        
        return A, Ham
    ## Method.
    #  @param self The object pointer.  
    def Integration_RK4_DAE_extended(self):
        """
        4th order Runge-Kutta scheme for the numerical integration of the ODE system
        """
        assert self.set_time_setting == 1, 'Time discretization must be specified first'
        
        tspan = self.tspan
        dt = self.dt
        
        JR = csc_matrix(self.J - self.R)
        self.M_sparse = csc_matrix(self.M)
        
        print('Linear system resolution, this may take a while ...')
        Mfacto = umfpack.UmfpackLU(self.M_sparse)
        Sys = Mfacto.solve_sparse(JR)
        B_D_sparse = csc_matrix(self.B_Dext)
        B_N_sparse = csc_matrix(self.B_Next)
        BMBfacto = umfpack.UmfpackLU(B_D_sparse.T @ Mfacto.solve_sparse(B_D_sparse) )
        
        Sys = Mfacto.solve_sparse(JR)
        
        Sys_ext = JR - B_D_sparse @ BMBfacto.solve_sparse(B_D_sparse.T) @ JR
        Sys_ext = Mfacto.solve_sparse(Sys_ext)
        
        Sys_N = B_N_sparse - B_D_sparse @ \
                        BMBfacto.solve_sparse(B_D_sparse.T @ Mfacto.solve_sparse(B_N_sparse))
        Sys_N = Mfacto.solve_sparse(Sys_N)
        
        Sys_D = B_D_sparse @ BMBfacto.solve_sparse(csc_matrix(self.Mb_D))
        Sys_D = Mfacto.solve_sparse(Sys_D)
        print('Linear system resolution: OK \n')
        
        def dif_func(t,y):
            return Sys_ext @ y + Sys_N @ self.Ub_N(t) + Sys_D @ self.Ub_D_der(t)
        
        A = np.zeros((self.Nsys, self.Nt+1))
        Ham = np.zeros(self.Nt+1)
        
        A[:,0] = self.A0
        Ham[0] = 1/2 * A[:,0] @ self.M @  A[:,0] 
        
        for k in range(self.Nt):
            k1 = dif_func(tspan[k], A[:,k])
            k2 = dif_func(tspan[k] + dt/2, A[:,k] + dt/2 * k1)
            k3 = dif_func(tspan[k] + dt/2, A[:,k] + dt/2 * k2)
            k4 = dif_func(tspan[k] + dt, A[:,k] + dt * k3)
            
            A[:,k+1] = A[:,k] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            Ham[k+1] = 1/2 * A[:,k+1] @ self.M @ A[:,k+1]
        
            # Progress bar
            perct = int(k/(self.Nt-1) * 100)  
            bar = ('Time-stepping RK4 : |' + '#' * int(perct/2) + ' ' + str(perct) + '%' + '|')
            sys.stdout.write('\r' + bar)
        print('\n intergation completed \n')
        
        return A, Ham

    
    ## Method.
    #  @param self The object pointer.   
    def Get_Deflection(self, A):
        # Get p variables
        Ap = A[self.dofs_Vpw,:]
                
        w      = np.zeros((self.Npw,self.Nt+1))
        w[:,0] = self.W0[:]
            
        for n in range(self.Nt):
            w[:,n+1] = w[:,n] + self.dt * 0.5 * (Ap[:,n+1] + Ap[:,n] ) 
            perct       = int(n/(self.Nt-1) * 100)  
            bar         = ('Get deflection \t' + '|' + '#' * int(perct/2) + ' ' + str(perct) + '%' + '|')
            sys.stdout.write('\r' + bar)
        print('\n integration completed')
        print(40*'-', '\n')        
        return w
    #%%
    
    #%% VISUALIZATION
    
    ## Method.
    #  @param self The object pointer.
    def Plot_Mesh(self, figure=True, figsize=(8,6.5), **kwargs):
        """
        Plot the two-dimensional mesh with the Fenics plot method
        """
        
        assert self.generate_mesh == 1, "The finite element mesh must be generated first"
       
        if figure : plt.figure(figsize=figsize)
        plot(self.Msh, **kwargs)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
        plt.title("Mesh with " + "Nv=" + str( self.Msh.num_vertices()) + ", hmax="+str(round(self.Msh.hmax(),3)) )
        plt.savefig("Mesh.png")

    def Plot_Mesh_with_DOFs(self, figure=True, figsize=(8,6.5), **kwargs):
        """
        Plot the two-dimensional mesh with the Fenics plot method including DOFs
        """
        
        assert self.generate_mesh == 1, "The finite element mesh must be generated first"

        if figure : plt.figure(figsize=figsize)
        plot(self.Msh, **kwargs)
        plt.plot(self.xpw, self.ypw, 'o', label='Dof of $e_w$ variables')
        plt.plot(self.xpth, self.ypth, '*', label='Dof of $e_{\theta}$ variables')
        plt.plot(self.xqth, self.yqth, '-', label='Dof of $E_{\kappa}$ variables')
        plt.plot(self.xqw, self.yqw, '^', label='Dof of $e_{\gamma}$ variables')
        plt.plot(self.xb, self.yb, 'ko', label='Dof of $\partial$ variables')
        plt.title('Mesh with associated DOFs, $Nq=$'+ str(self.Nq)+ ', $Np=$'+ str(self.Np)+ ', $N_\partial=$'+ str(self.Nb) + ', $Nv=$' + str( self.Msh.num_vertices()) )
        plt.legend()
        if not(self.notebook):
            plt.show()
        plt.savefig("Mesh_with_DOFs.png")
        
    def Plot_Hamiltonian(self, tspan, Ham, figsize=(8,6.5), **kwargs):
        """
        Plot the Hamiltonian function versus time 
        """
        plt.figure(figsize=figsize)
        plt.plot(tspan, Ham, **kwargs)
        plt.xlabel('Time $(s)$')                                                                   
        plt.ylabel('Hamiltonian')
        plt.title('Hamiltonian')
        plt.grid(True)
        if not(self.notebook):
            plt.show()
        plt.savefig("Hamiltonian.png")    
        
    def Plot_Hamiltonian_RE(self, tspan, Ham, figsize=(8,6.5), **kwargs):
        """
        Plot the Hamiltonian function versus time 
        """
        print('Only for closed system')
        plt.figure(figsize=figsize)
        plt.semilogy(tspan, 100 * np.abs(Ham-Ham[0])/Ham[0], **kwargs)
        plt.xlabel('Time $(s)$')
        plt.ylabel('Error (%)')
        plt.title('Hamiltonian Relative Error')
        plt.grid(True)
        if not(self.notebook):
            plt.show()
        plt.savefig("Hamiltonian_RE.png")
        
    def Trisurf(self, SpaceTimeVector_Lag, instance=0, title='', save=False, figsize=(8,6.5), **kwargs):
        instance = int(np.where(self.tspan == instance)[0])
        fig = plt.figure(figsize=figsize)
        ax = fig.gca(projection='3d')
        #fig.colorbar(wfram, shrink=0.5, aspect=5)
        ax.set_xlabel('x coordinate')
        ax.set_ylabel('y coordinate')
        plt.title(title)
        ax.set_xlim(self.x0, self.xL)
        ax.set_ylim(self.y0, self.yL)
        ax.set_zlim(np.min(SpaceTimeVector_Lag), np.max(SpaceTimeVector_Lag))
        ax.plot_trisurf(self.xp, self.yp, SpaceTimeVector_Lag[:,instance], linewidth=0.2, \
                                         antialiased=True, cmap=plt.cm.CMRmap)#, **kwargs)
        if save: fig.save(title+'png')
        
    
    # Set writer for video saving
    def Set_Video_Writer(self, fps=128, dpi=256):
        """
        Set video writer options
        """
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata     = dict(title='Movie Test', artist='Matplotlib',
                            comment='Movie support!')
        self.writer  = FFMpegWriter(fps=fps, metadata=metadata)
        self.dpi     = dpi
        
    def Moving_Plot(self, y, x,  step=1, title='', save=False, figsize=(8,6.5), **kwargs):
        """
        Create a 2D animation with the plot command
        """
        fig = plt.figure(figsize=figsize)
        plt.xlim(x[0], x[-1])
        plt.ylim(np.min(y)*0.875, np.max(y)*1.125)
        plt.grid(True)
        
        # Save 
        if save :
            with self.writer.saving(fig, title+'.mp4', self.dpi):
                for i in range(0,self.Nt,step):
                    plt.plot(x[i:i+1], y[i:i+1], '.', color='#1f77b4')
                    plt.plot(x[0:i+1], y[0:i+1], '-', color='#1f77b4', **kwargs)
                    plt.title(title+ ' t='+np.array2string(round(x[i],2)) + '/' + np.array2string(round(x[-1],2)))
                    self.writer.grab_frame()
                    plt.pause(0.01)
        # Do not save
        else :
            for i in range(0,self.Nt,step):
                plt.plot(x[i:i+1], y[i:i+1], '.', color='#1f77b4')
                plt.plot(x[0:i+1], y[0:i+1], '-', color='#1f77b4', **kwargs)
                plt.title(title + ' t='+np.array2string(x[i]) + '/' + np.array2string(x[-1]))
                plt.pause(0.01)
    
    # Moving trisurf
    def Moving_Trisurf(self, SpaceTimeVector_Lag, step=1, title='', save=False, figsize=(8,6.5), cmap=plt.cm.CMRmap, **kwargs):
        """
        Create a 3D animation with the plot_trisurf Matplotlib command
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.gca(projection='3d')
        #fig.colorbar(wfram, shrink=0.5, aspect=5)
        ax.set_xlabel('x coordinate')
        ax.set_ylabel('y coordinate')
        plt.title(title)
        ax.set_xlim(self.x0, self.xL)
        ax.set_ylim(self.y0, self.yL)
        ax.set_zlim(np.min(SpaceTimeVector_Lag), np.max(SpaceTimeVector_Lag))
        
        wframe = None
        
        # Save
        if save :
            with self.writer.saving(fig, title+'.mp4', self.dpi):
                for i in range(0, self.Nt+1, step):
                    if wframe:
                        ax.collections.remove(wframe)
                    wframe = ax.plot_trisurf(self.xp, self.yp, SpaceTimeVector_Lag[:,i], linewidth=0.2, \
                                             antialiased=True, cmap=cmap, **kwargs)
                    ax.set_zlabel('time=' + np.array2string(self.tspan[i]) + '/' + np.array2string(self.tspan[-1]) ) 
                    self.writer.grab_frame()
                    plt.pause(.001)
        # Do not save
        else:
            for i in range(0, self.Nt+1, step):
                if wframe:
                    ax.collections.remove(wframe)
                wframe = ax.plot_trisurf(self.xpw, self.ypw, SpaceTimeVector_Lag[:,i], linewidth=0.2, \
                                         antialiased=True, cmap=cmap, **kwargs)
                ax.set_zlabel('time=' + np.array2string(self.tspan[i]) + '/' + np.array2string(self.tspan[-1]) ) 
                plt.pause(.001)
    
    # Moving contour
    def Moving_Contour(self, time_space_Vector, step=1, with_mesh=False, title='', save=False, figsize=(8,6.5), **kwargs):
        """
        Create a 2D animation with arrows on vector quantities
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        ax.set_xlabel('x coordinate')
        ax.set_ylabel('y coordinate')
        wframe = None
        tstart = time.time()
        temp_vec = Function(self.Vp)
        
        # Save
        if save :
            with self.writer.saving(fig, title+'.mp4', self.dpi):
                if with_mesh : plot(self.Msh, linewidth=.25)
                for i in range(0, self.Nt+1, step):
                    if wframe :
                        ax.collections.remove(wframe)
                    
                    temp_vec.vector()[range(self.Np)] = time_space_Vector[range(self.Np),i] 
                    wframe = plot(temp_vec, title=title+', t='+str(np.round(self.tspan[i],2)), **kwargs)
                    self.writer.grab_frame()
                    plt.pause(.001)
        # Do not Save
        else :
            if with_mesh : plot(self.Msh, linewidth=.25)
            for i in range(0, self.Nt+1, step):
                if wframe :
                    ax.collections.remove(wframe)
                temp_vec.vector()[range(self.Np)] = time_space_Vector[range(self.Np),i] 
                    
                wframe = plot(temp_vec, title=title+', t='+str(np.round(self.tspan[i],2), **kwargs)+'/'+str(self.tfinal))
                plt.pause(.001)
    
    # Moving quiver
    def Moving_Quiver(self, time_space_Vector, step=1, with_mesh=False, title='', save=False, figsize=(8,6.5), **kwargs):
        """
        Create a 2D animation with arrows on vector quantities
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        ax.set_xlabel('x coordinate')
        ax.set_ylabel('y coordinate')
        wframe = None
        tstart = time.time()
        temp_vec = Function(self.Vq)
        plot(self.Msh, linewidth=0.5)
        
        # Save
        if save :
            with self.writer.saving(fig, title+'.mp4', self.dpi):
                if with_mesh : plot(self.Msh, linewidth=.25)
                for i in range(0, self.Nt+1, step):
                    if wframe :
                        ax.collections.remove(wframe)
                    
                    temp_vec.vector()[range(self.Nq)] = time_space_Vector[range(self.Nq),i] 
                    wframe = plot(temp_vec, title=title+', t='+str(np.round(self.tspan[i],2))+'/'+str(self.tfinal), **kwargs)
                    self.writer.grab_frame()
                    plt.pause(.001)
        # Do not Save
        else :
            if with_mesh : plot(self.Msh, linewidth=.25)
            for i in range(0, self.Nt+1, step):
                if wframe :
                    ax.collections.remove(wframe)
                temp_vec.vector()[range(self.Nq)] = time_space_Vector[range(self.Nq),i] 
                wframe = plot(temp_vec, title=title+', t='+str(np.round(self.tspan[i],2))+'/'+str(self.tfinal), **kwargs)
                plt.pause(.001)
            
