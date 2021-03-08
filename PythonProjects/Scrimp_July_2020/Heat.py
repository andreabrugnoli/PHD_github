#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Authors   : A. Serhani & X. Vasseur
Template  : Heat diffusion as a port-Hamiltonian system        
Project   : INFIDHEM https://websites.isae-supaero.fr/infidhem/
Openforge : https://openforge.isae.fr/projects/scrimp
Contact   : anass.serhani@isae.fr & xavier.vasseur@isae.fr
"""

from dolfin import *
from mshr import *

import numpy as np
from scipy import linalg
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse import issparse
from scikits import umfpack

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as manimation

from assimulo.problem import Implicit_Problem
from assimulo.solvers.sundials import IDA

import sys
import copy


class Heat_2D:
    #%% CLASS INSTANTIATION
    def __init__(self):
        """
        Constructor for the Heat_2D class
        """        
        # Information related to the problem definition
        self.problem_definition               = 0                  
        self.set_boundary_control             = 0
        self.set_initial_final_time           = 0
        self.set_mixed_boundaries             = 0
        self.set_dirichlet_boundary_control   = 0
        self.set_normal_boundary_control      = 0
        self.set_physical_parameters          = 0
        self.set_rectangular_domain           = 0
 
        
        # Information related to the space-time discretization
        self.assembly                         = 0
        self.generate_mesh                    = 0
        self.project_boundary_control         = 0
        self.project_initial_data             = 0
        self.set_finite_elements_spaces       = 0
        
        # Information related to the post-processing
        self.docker                           = False
        self.interactive                      = False
        self.notebook                         = False
        
        # Information related to printing
        self.verbose                          = True
        
        # Information related to the memory 
        self.memory_constrained               = False
        
    ## Method.
    #  @param self The object pointer.        
    def __call__(self):
        if self.verbose: print('Heat class of Scrimp created by A. Serhani')
        if self.verbose: print(40*'-', '\n')

    
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
    #  @param self The object pointer.
    def Set_Boundary_Control(self, Ub_tm0, Ub_sp0, Ub_tm1=lambda t : 0, Ub_sp1='0', **kwargs):
        """
        Set boundary control as callable time functions and regular FEniCS expression
        u_\partial(t,x) = Ub_tm0(t) * Ub_sp0(x) + Ub_tm1(t) + Ub_sp1(x)
        """
        assert self.set_rectangular_domain == 1, 'The domain must be defined before.'
        
        self.Ub_tm0 = Ub_tm0

        self.Ub_sp0_Expression = Expression(Ub_sp0, degree=2,\
                                              x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                                              **kwargs)
        
        self.Ub_tm1 = Ub_tm1

        self.Ub_sp1_Expression = Expression(Ub_sp1, degree=2,\
                                              x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                                              **kwargs)
        self.set_boundary_control = 1
        
        
        if self.verbose: print(40*'-')
        if self.verbose: print('Boundary control: OK')
        if self.verbose: print(40*'-')
        
        return self.set_boundary_control
    
    
    ## Method.
    #  @param self The object pointer.
    def Set_Initial_Final_Time(self, initial_time, final_time):
        """
        Set the initial, close and final times for defining the time domain
        """
        self.tinit  = initial_time 
        self.tfinal = final_time
        
        self.set_initial_final_time = 1
        
        if self.verbose: print(40*'-')
        if self.verbose: print('Initial/Final time: OK')
        if self.verbose: print(40*'-')
        
        return self.set_initial_final_time

    ## Method.
    #  @param self The object pointer.
    def Set_Physical_Parameters(self, rho, Lambda11, Lambda12, Lambda22, CV, **kwargs):
        """
        Set the physical parameters as a FeniCS expression related to the PDE
        """
        assert self.set_rectangular_domain == 1, 'The domain must be defined before.'
        
        self.rho        = Expression(rho, degree=2,\
                              x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                              **kwargs)
        self.Lambda     = Expression( ( (Lambda11, Lambda12), (Lambda12, Lambda22) ), degree=2,\
                              x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                              **kwargs)
        self.CV         = Expression(CV, degree=2,\
                              x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                              **kwargs)

        self.set_physical_parameters = 1
    
        if self.verbose: print(40*'-')
        if self.verbose: print('Physical parameters: OK')
        if self.verbose: print(40*'-')
        
        return self.set_physical_parameters
    
    
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

        if self.verbose: print(40*'-')
        if self.verbose: print('Rectangular domain: OK')
        if self.verbose: print(40*'-')

        return self.set_rectangular_domain
    
    
    
    #%% SPACE-TIME DISCRETIZATION
  
    ## Method.
    # @param self The object pointer.
    def Assembly(self):
        """
        Perform the matrix assembly related to the PFEM formulation
        """
        assert self.generate_mesh == 1, "The finite element mesh must be generated first"
        
        assert self.set_finite_elements_spaces == 1, \
                "The finite element spaces must be selected first"
        
        # Mass/Weighted-Mass matrices
        self.M      = assemble( self.alpha * self.va * dx).array()
        self.Mrho   = assemble( self.rho * self.alpha * self.va * dx).array()
        self.MrhoCV = assemble( self.rho * self.CV * self.alpha * self.va * dx).array()
        self.MCV    = assemble( self.CV * self.alpha * self.va * dx).array()
        self.MM     = assemble( dot(self.Alpha, self.vA) * dx).array()
        self.LAMBDA = assemble( dot(self.Lambda * self.Alpha, self.vA) * dx).array()
        self.Mb     = assemble( self.ab * self.vb * ds).array()[self.b_ind,:][:,self.b_ind]           

        # Structure/Boundary matrices
        self.G      = assemble( dot(self.Alpha, grad(self.va)) * dx).array()
        self.BG     = assemble( self.ab * self.va * ds).array()[:,self.b_ind] 
        
        self.D      = assemble( - div(self.Alpha) * self.va * dx).array()
        self.BD     = assemble( - self.ab * dot(self.vA, self.norext) * ds).array()[:,self.b_ind] 

        # Deduce sparse formats here
        
        self.M      = csr_matrix(self.M)
        self.Mrho   = csr_matrix(self.Mrho)
        self.MrhoCV = csr_matrix(self.MrhoCV)
        self.MCV    = csr_matrix(self.MCV)
        self.MM     = csr_matrix(self.MM)
        self.LAMBDA = csr_matrix(self.LAMBDA)
        self.Mb     = csr_matrix(self.Mb)
               
        self.BD     = csr_matrix(self.BD)
        self.BG     = csr_matrix(self.BG)
        self.D      = csr_matrix(self.D)
        self.G      = csr_matrix(self.G)
        
        self.assembly = 1
    
        if self.verbose: print(40*'-')
        if self.verbose: print('Assembly: OK')
        if self.verbose: print(40*'-')
        
        return self.assembly

    ## Method.
    # @param self The object pointer.         
    def Assign_Mesh(self, Msh):
        """
        Assign an already generated mesh
        """
        # Mesh
        self.Msh = Msh
        self.norext   = FacetNormal(self.Msh)
        
        # Explicit coordinates
        self.xv = Msh.coordinates()[:,0]
        self.yv = Msh.coordinates()[:,1]
        self.Nv = self.Msh.num_vertices()

        # Domain limits
        self.x0, self.xL = self.xv.min(), self.xv.max() 
        self.y0, self.yL = self.yv.min(), self.yv.max()
        
        if self.verbose: print(40*'-')
        if self.verbose: print('Mesh:', 'hmax=', round(self.Msh.hmax(),3),'Nv=', self.Msh.num_vertices() )
        if self.verbose: print('Mesh: OK')
        if self.verbose: print(40*'-')
    
        self.generate_mesh = 1
        
        return self.generate_mesh
  
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
            'The boundary control must be interpolated on the FE spaces' 
        
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
        Perform the mesh generation through the FEniCS meshing functionalities
        """
        self.rfn = rfn  
    
        rfny     =  int (self.rfn * (self.yL - self.y0) / (self.xL - self.x0) ) 
        
        if structured_mesh: 
            self.Msh = RectangleMesh(Point(self.x0,self.y0), Point(self.xL,self.yL),\
                                     self.rfn, rfny, 'crossed')
        else:
            self.Msh = generate_mesh(Rectangle(Point(self.x0,self.y0), Point(self.xL,self.yL)),\
                                     self.rfn)
        
        # Explicit coordinates
        self.xv = self.Msh.coordinates()[:,0]
        self.yv = self.Msh.coordinates()[:,1]
        self.Nv = self.Msh.num_vertices()
        
        self.norext   = FacetNormal(self.Msh)
        
        if self.verbose: print(40*'-')
        if self.verbose: print('Mesh:', 'hmax=', round(self.Msh.hmax(),3),'Nv=', self.Msh.num_vertices())
        if self.verbose: print('Mesh: OK')
        if self.verbose: print(40*'-')
        
        self.generate_mesh = 1
        
        return self.generate_mesh 
     
    ## Method.
    #  @param self The object pointer.
    def Project_Boundary_Control(self):
        """
        Project boundary control on the FE spaces 
        """        
        assert self.generate_mesh == 1, "The finite element mesh must be generated first"
    
        assert self.set_finite_elements_spaces == 1, \
                "The FE approximation spaces must be selected first"
       
        self.Ub_sp0 = interpolate(self.Ub_sp0_Expression, self.Vb).vector()[self.b_ind]
        self.Ub_sp1 = interpolate(self.Ub_sp1_Expression, self.Vb).vector()[self.b_ind]
        self.Ub     = lambda t : self.Ub_sp0 * self.Ub_tm0(t) + self.Ub_sp1 + self.Ub_tm1(t) * np.ones(self.Nb)
        
        self.project_boundary_control = 1
        
        if self.verbose: print(40*'-')    
        if self.verbose: print('Project boundary control: OK')
        if self.verbose: print(40*'-')
        
        return self.project_boundary_control   
 
    ## Method.
    #  @param self The object pointer.   
    def Set_Finite_Element_Spaces(self, family_scalar, family_Vector, family_boundary, rs, rV, rb):        
        """
        Set the finite element approximation spaces related to the space discretization of the PDE
        """
        assert self.generate_mesh == 1, "The finite element mesh must be generated first"
        
        # Spaces
        if family_Vector == 'P':
            if rV == 0 :  
                self.VV = VectorFunctionSpace(self.Msh, 'DG', 0, dim=2) 
            else :  
                self.VV = VectorFunctionSpace(self.Msh, 'P', rV, dim=2) 
        elif family_Vector == 'RT' :
            self.VV = FunctionSpace(self.Msh, 'RT', rV+1)
        else :
            self.VV = FunctionSpace(self.Msh, family_Vector, rV)
  
        if rs == 0 :
            self.Vs = FunctionSpace(self.Msh, 'DG', 0)
        else :
            self.Vs = FunctionSpace(self.Msh, family_scalar, rs)

        if rb == 0 :
            self.Vb = FunctionSpace(self.Msh, 'CR', 1)
        else :
            self.Vb = FunctionSpace(self.Msh, family_boundary, rb)
            
        # Orders
        self.rV = rV
        self.rs = rs
        self.rb = rb
        
        # DOFs
        self.NV = self.VV.dim()
        self.Ns = self.Vs.dim()
        self.Nsys = self.NV + self.Ns
        coord_V = self.VV.tabulate_dof_coordinates()
        coord_s = self.Vs.tabulate_dof_coordinates()
        
        # Explicit coordinates
        self.xs = coord_s[:,0]
        self.ys = coord_s[:,1]
        
        self.xV = coord_V[:,0]
        self.yV = coord_V[:,1] 
        
        # Boundary DOFs for control and observation: necessary for B matrix        
        coord_b  = self.Vb.tabulate_dof_coordinates()
        xb       = coord_b[:,0]
        yb       = coord_b[:,1]
        # Boundary DOFs for control and observation: necessary for B matrix        
        coord_b = self.Vb.tabulate_dof_coordinates()
        def Get_Associated_DOFs(SubDomain, Space):
            BC = DirichletBC(Space, '0', SubDomain)
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
            if  ( np.abs(self.xb[i] - self.x0) <= 1e-16 and np.abs(self.yb[i] - self.y0) <= 1e-16 ) \
            or ( np.abs(self.xb[i] - self.x0) <= 1e-16 and np.abs(self.yb[i] - self.yL) <= 1e-16 ) \
            or ( np.abs(self.xb[i] - self.xL) <= 1e-16 and np.abs(self.yb[i] - self.y0) <= 1e-16 ) \
            or ( np.abs(self.xb[i] - self.xL) <= 1e-16 and np.abs(self.yb[i] - self.yL) <= 1e-16 ) : 
                 self.Corner_indices.append(i)
         
        if self.verbose: print(40*'-')
        if self.verbose: print('VV=', family_Vector+'_'+str(rV), ',\t Vs=', family_scalar+'_'+str(rs), ',\t Vb=', family_boundary+'_'+str(rb))
        if self.verbose: print('NV=', self.NV, ',\t Ns=', self.Ns, ',\t Nb=', self.Nb)
        if self.verbose: print('DOFsys=', self.NV+self.Ns)
        if self.verbose: print('FE spaces: OK')
        if self.verbose: print(40*'-')
        
        # Declare functions for variational formulation
        self.alpha, self.Alpha  = TrialFunction(self.Vs), TrialFunction(self.VV)
        self.va, self.vA        = TestFunction(self.Vs), TestFunction(self.VV)
        self.ab, self.vb        = TrialFunction(self.Vb), TestFunction(self.Vb)
        
        self.set_finite_elements_spaces = 1
        
        return self.set_finite_elements_spaces
    
    ## Method.
    #  @param self The object pointer.           
    def Set_Formulation(self, formulation):
        """
        Specify the PFEM formulation
        """
            
        self.formulation     = formulation
        self.set_formulation = 1
        
        if self.verbose: print(40*'-')
        if self.verbose: print('Formulation '+formulation+'-'+formulation+': OK')
        if self.verbose: print(40*'-')
        
        return self.set_formulation
    
    ## Method.
    # @param self The object pointer. 
    def Set_Gmsh_Mesh(self, xmlfile, rfn_num=0):
        """
        Set a mesh generated by Gmsh using the "xml" format
        """
        # Create FEniCS mesh
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
        
        # Explicit coordinates
        self.xv = Msh.coordinates()[:,0]
        self.yv = Msh.coordinates()[:,1]
        self.Nv = self.Msh.num_vertices()

        # Domain limits 
        self.x0, self.xL = self.xv.min(), self.xv.max() 
        self.y0, self.yL = self.yv.min(), self.yv.max()

        if self.verbose: print(40*'-')
        if self.verbose: print('Mesh:', 'hmax=', round(self.Msh.hmax(),3),'Nv=', self.Msh.num_vertices() )
        if self.verbose: print('Mesh: OK')
        if self.verbose: print(40*'-')
        
        self.generate_mesh = 1
        
        return self.generate_mesh
        
    ## Method.
    #  @param self The object pointer.           
    def Set_Time_Setting(self, time_step, tf=None):
        """
        Specify the parameters related to the time integration
        """
        if tf is not None: 
            self.tfinal = tf 
            
        self.dt     = time_step
        self.Nt     = int( np.floor(self.tfinal/self.dt) )
        self.tspan  = np.linspace(0,self.tfinal,self.Nt+1)
        
        self.set_time_setting = 1
        
        if self.verbose: print(40*'-')
        if self.verbose: print('dt =', self.dt)
        if self.verbose: print('Time setting: OK')
        if self.verbose: print(40*'-')
        
        return self.set_time_setting

    #%% POST-PROCESSING
    
    ## Method.
    #  @param self The object pointer. 
    
    def Contour_Quiver(self, var_scalar, var_Vect, t, with_mesh=True, title='', margin = .125, save=False, figsize=(8,6.5), cmap=plt.cm.CMRmap, levels=265):    
        """
        Create a Contour/Quiver plot at a specific time t
        """
        t   = int(np.where(self.tspan == t)[0])
        fig = plt.figure(figsize=figsize)
        
        plt.xlabel('x coordinate')
        plt.ylabel('y coordinate')
        Dxy = margin * min(self.xL - self.x0, self.yL - self.y0)
        plt.xlim([self.x0 - Dxy, self.xL + Dxy])
        plt.ylim([self.y0 - Dxy, self.yL + Dxy])
        plt.title(title)
    
        f_scalar    = Function(self.Vs)
        f_Vect      = Function(self.VV)
    
        Vmin=(var_scalar).min() 
        Vmax=(var_scalar).max()
    
        f_scalar.vector()[range(self.Ns)]   = var_scalar[range(self.Ns), t]
        f_Vect.vector()[range(self.NV)]     = var_Vect[range(self.NV), t]
        
        levels  = np.linspace(Vmin,Vmax, levels)
        p       = plot(f_scalar, vmin=Vmin, vmax=Vmax, levels=levels, cmap=cmap)
        plt.colorbar(p)
        
        plot(f_Vect, width=.0025, cmap=cmap, clim = (var_Vect.min()/linalg.norm(var_Vect), var_Vect.max()/linalg.norm(var_Vect)))
       
        if with_mesh : plot(self.Msh, linewidth=.5)
    
        if save: fig.save(title+'.png')  
        
    ## Method.
    #  @param self The object pointer.  
          
    def Plot_Hamiltonian(self, tspan, Ham, figsize=(8,6.5), title='Hamiltonian', **kwargs):
        """
        Plot the Hamiltonian function versus time 
        """
        plt.figure(figsize=figsize)
        plt.plot(tspan, Ham, **kwargs)
        plt.xlabel('Time $(s)$')                                                                   
        plt.ylabel('Hamiltonian $(J)$')
        plt.title(title)
        plt.grid(True)
        if not(self.notebook):
            plt.show()
        plt.savefig("Hamiltonian.png")    
        
        
    ## Method.
    #  @param self The object pointer.    
      
    def Plot_Hamiltonian_Relative_Error(self, tspan, Ham, figsize=(8,6.5), **kwargs):
        """
        Plot the relative error related to the Hamiltonian versus time 
        """
        
        plt.figure(figsize=figsize)
        plt.semilogy(tspan, 100 * np.abs(Ham-Ham[0])/Ham[0], **kwargs)
        plt.xlabel('Time $(s)$')
        plt.ylabel('Error (%)')
        plt.title('Hamiltonian Relative Error')
        plt.grid(True)
        
        if not(self.notebook):
            plt.show()
        plt.savefig("Hamiltonian_Relative_Error.png") 
        
    ## Method.
    #  @param self The object pointer.
    
    def Plot_Mesh(self, figure=True, figsize=(8,6.5), **kwargs):
        """
        Plot the two-dimensional mesh with the FEniCS plot method
        """
        
        assert self.generate_mesh == 1, "The finite element mesh must be generated first"
       
        if figure : plt.figure(figsize=figsize)
        plot(self.Msh, **kwargs)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
        plt.title("Mesh with " + "Nv=" + str( self.Msh.num_vertices()) + ", hmax="+str(round(self.Msh.hmax(),3)) )
        plt.savefig("Mesh.png")
        
        
    ## Method.
    #  @param self The object pointer.

    def Plot_Mesh_with_DOFs(self, figure=True, figsize=(8,6.5), **kwargs):
        """
        Plot the two-dimensional mesh with the FEniCS plot method including DOFs
        """
        assert self.generate_mesh == 1, "The finite element mesh must be generated first"

        if figure : plt.figure(figsize=figsize)
        plot(self.Msh, **kwargs)
        plt.plot(self.xV, self.yV, 'o', label='Dof of $V$ variables')
        plt.plot(self.xs, self.ys, '^', label='Dof of $s$ variables')
        plt.plot(self.xb, self.yb, 'ko', label='Dof of $\partial$ variables')
        plt.title('Mesh with associated DOFs, $Nq=$'+ str(self.NV)+ ', $Np=$'+ str(self.Ns)+ ', $N_\partial=$'+ str(self.Nb) + ', $Nv=$' + str( self.Msh.num_vertices()) )
        plt.legend()
        if not(self.notebook):
            plt.show()
        plt.savefig("Mesh_with_DOFs.png")

    #%% UTILITY
    
    ## Method
    #  @param self The object pointer.
    def my_mult(self, A, B):
        """
        Matrix multiplication.

        Multiplies A and B together via the "dot" method.

        Parameters
        ----------
        A : array_like
            first matrix in the product A*B being calculated
        B : array_like
            second matrix in the product A*B being calculated

        Returns
        -------
        array_like
            product of the inputs A and B
        """
        
        if issparse(B) and not issparse(A):
            # dense.dot(sparse) is not available in scipy.
            return B.T.dot(A.T).T
        else:
            return A.dot(B)    
    
    
class Energy(Heat_2D): 
    #%% PROBLEM DEFINITION

    ## Method.
    #  @param self The object pointer. 
      
    def Create_Initial_Data(self, level, amplitude, **kwargs):
        """
        Create initial data as Numpy arrays
        """
        assert self.set_rectangular_domain == 1, 'The domain must be defined before.'
        assert self.set_finite_elements_spaces == 1, 'The FE spaces must be specified first'
        
        # Create a Gaussian profile 
        
        sX, sY, X0, Y0  = self.xL/4, self.yL/4, self.xL/2, self.yL/2 
        
        Gaussian_init   = Expression(' ampl * exp(- pow( (x[0]-X0)/sX, 2) - pow( (x[1]-Y0)/sY, 2) ) + levinit', 
                                     degree=2, levinit=level, ampl=amplitude, sX=sX, sY=sY, X0=X0, Y0=Y0)
        
        Ginit           = interpolate(Gaussian_init, self.Vs)  
        
        a_init          = level*np.ones(self.Ns)
        
        e_init          = Ginit.vector()[:]

        self.create_initial_data = 1

        return a_init, e_init    
    
    ## Method.
    #  @param self The object pointer. 
      
    def Set_Initial_Data(self, au_0=None, eu_0=None, init_by_vector=False, au0=None, eu0=None, **kwargs):
        """
        Set initial data on the FE spaces 
        """
        
        assert self.set_rectangular_domain == 1, 'The domain must be defined before.'

        # Expressions
        if not init_by_vector :
            self.au_0  = Expression(au_0, degree=2, x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL, **kwargs)
            self.eu_0  = Expression(eu_0, degree=2, x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL, **kwargs)
            self.init_by_vector = False
        else :
            # Vectors
            self.au0            = au0            
            self.eu0            = eu0
            self.init_by_vector = True

        self.set_initial_data = 1
        
        if self.verbose: print(40*'-')
        if self.verbose: print('Initial data: OK')
        if self.verbose: print(40*'-')

        return self.set_initial_data   
    
    #%% SPACE-TIME DISCRETIZATION
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
            self.au0 = interpolate(self.au_0, self.Vs).vector()[:]
            self.eu0 = interpolate(self.eu_0, self.Vs).vector()[:]
        
        if self.project_boundary_control == 1 : print('Project BC: OK')
        
        if self.verbose: print(40*'-')
        if self.verbose: print('Project initial data: OK')
        if self.verbose: print(40*'-')
        
        self.project_initial_data = 1
        
        return self.project_initial_data 
    
    #%% TIME INTEGRATION
    ## Method.
    #  @param self The object pointer.    
    def Integration_DAE(self):
        """
        Solution of the DAE system related to the energy PFEM formulation  with a forward in time stepping method
        """
        # Dimensions
        Nt  = self.Nt
        Ns  = self.Ns
        NV  = self.NV
        Vs  = self.Vs
        VV  = self.VV
        
        # Continuous FEniCS functions
        au_diff_    = Function(Vs)
        fU_         = Function(VV)
        fsig_       = Function(Vs)
        eu_         = Function(Vs)      
        eU_         = Function(VV)
        
        # Prepare vectors and matrices
        
        Mrho_solver = umfpack.UmfpackLU(csc_matrix(self.Mrho))
        
        if not self.memory_constrained:
            MrhoD   = Mrho_solver.solve(self.D)
            MrhoM   = Mrho_solver.solve(self.M)
            
        MM_solver   = umfpack.UmfpackLU(csc_matrix(self.MM)) 
        
        if not self.memory_constrained:
            MMDT    = MM_solver.solve(self.D.T)                          
            MMBD    = MM_solver.solve(self.BD)

        
        # (I) - initialization
        au          = np.zeros((Ns, Nt+1))
        au[:,0]     = self.au0
        
        # (II) - initialization
        eu          = np.zeros((Ns, Nt+1))
        eu[:,0]     = self.eu0

        # (III) - initialization
        fU          = np.zeros((NV, Nt+1))
        if not self.memory_constrained:
            fU[:,0] = - MMDT @ eu[:,0] + MMBD @ self.Ub(self.tspan[0])
        else:
            fU[:,0] = MM_solver.solve(-self.my_mult(self.D.T,eu[:,0])+self.my_mult(self.BD,self.Ub(self.tspan[0])))
        
        # (IV) - initialization
        fsig        = np.zeros((Ns, Nt+1))
        fsig[:,0]   = eu[:,0]
        
        # (V) - initialization
        eU                      = np.zeros((NV, Nt+1))
        eu_.vector()[range(Ns)] = eu[range(Ns),0]
        Meu                     = csc_matrix(assemble(eu_ * dot(self.Alpha, self.vA) * dx).array())
        Meu_solver              = umfpack.UmfpackLU(Meu)
        fU_.vector()[range(NV)] = fU[range(NV),0]
        LfU                     = assemble(dot(self.Lambda * fU_, self.vA ) * dx)[:]
        eU[:,0]                 = Meu_solver.solve(LfU)

        # (VI) - initialization
        esig                        = np.zeros((Ns, Nt+1))
        fsig_.vector()[range(Ns)]   = fsig[range(Ns),0]
        Mfsig                       = csc_matrix(assemble(fsig_ * self.alpha * self.va * dx).array())
        Mfsig_solver                = umfpack.UmfpackLU(Mfsig)
        eU_.vector()[range(NV)]     = eU[range(NV),0]
        LeU                         = assemble(dot(eU_, fU_) * self.va * dx)[:]
        esig[:,0]                   = -Mfsig_solver.solve(LeU)
        
        for k in range(Nt):
            # (I) - update (time-stepping)
            if not self.memory_constrained:
                au[:,k+1]   = au[:,k] + self.dt * ( MrhoD @ eU[:,k] - MrhoM @ esig[:, k] )
            else:
                au[:,k+1]   = au[:,k] + self.dt * Mrho_solver.solve(self.my_mult(self.D,eU[:,k])-self.my_mult(self.M,esig[:, k]))
        
            # (II) - update (Dulong-Petit law)
            au_diff_.vector()[range(Ns)]    = (au[range(Ns), k+1] - au[range(Ns), k])
            M_au                            = csc_matrix(self.MCV - self.dt * assemble( self.alpha * au_diff_/self.dt * self.va * dx ).array())
            M_au_solver                     = umfpack.UmfpackLU(M_au)
            eu[:,k+1]                       = M_au_solver.solve(self.my_mult(self.MCV,eu[:,k]))
            
            # (III) - update (closure 2)
            if not self.memory_constrained:
                fU[:,k+1]   = - MMDT @ eu[:,k+1] + MMBD @ self.Ub(self.tspan[k+1])
            else:
                fU[:,k+1]   = MM_solver.solve(-self.my_mult(self.D.T,eu[:,k+1])+self.my_mult(self.BD,self.Ub(self.tspan[k+1])))
            
            # (IV) - update (closure 3)
            fsig[:,k+1] = eu[:,k+1]
            
            # (V) - update (Fourier law)
            eu_.vector()[range(Ns)] = eu[range(Ns),k+1]
            Meu                     = csc_matrix(assemble(eu_ * dot(self.Alpha, self.vA) * dx).array())
            Meu_solver              = umfpack.UmfpackLU(Meu)
            fU_.vector()[range(NV)] = fU[range(NV),k+1]
            LfU                     = assemble(dot(self.Lambda * fU_, self.vA ) * dx)[:]
            eU[:,k+1]               = Meu_solver.solve(LfU)
            
            # (VI) - update (closure sigma)
            fsig_.vector()[range(Ns)]   = fsig[range(Ns),k+1]
            Mfsig                       = csc_matrix(assemble(fsig_ * self.alpha * self.va * dx).array())
            Mfsig_solver                = umfpack.UmfpackLU(Mfsig)
            eU_.vector()[range(NV)]     = eU[range(NV),k+1]
            LeU                         = assemble(dot(eU_, fU_) * self.va * dx)[:]
            esig[:,k+1]                 = -Mfsig_solver.solve(LeU)
                     
            perct = int(k/(Nt-1) * 100)  
            bar = ('Time integration: |' + '#' * int(perct/2) + ' ' + str(perct) + '%' + '|')
            sys.stdout.write('\r' + bar)
            
    

        
        Ham = np.zeros(Nt+1)
        
        for k in range(Nt+1):
            T_                      = Function(Vs)
            T_.vector()[range(Ns)]  = eu[range(Ns),k]        
            Ham[k] = assemble(self.rho * self.CV * T_ * dx(self.Msh))        
        
#        T_                      = Function(Vs)
#        T_.vector()[range(Ns)]  = eu[range(Ns),0]        
#        Ham[0] = assemble(self.rho * self.CV * T_ * dx(self.Msh))
#        for k in range(Nt):
#            #Ham[k+1] = Ham[k] + self.dt * self.Ub(self.tspan[k+1]) @ self.BD.T @ eU[:,k+1] 
#            if self.formulation == 'div':
#                rest = self.my_mult(self.Ub(self.tspan[k+1]).T,self.my_mult(self.BD.T,eU[:,k+1]))
#
#            if self.formulation == 'grad':
#                rest = -self.my_mult(self.Ub(self.tspan[k+1]).T,self.my_mult(self.BG.T,eu[:,k+1]))
#
#            Ham[k+1] = Ham[k] + self.dt * rest
            
        


        return au, fU, fsig, eu, eU, esig, Ham

  

class Entropy(Heat_2D): 
    
    #%% PROBLEM DEFINITION
    ## Method.
    #  @param self The object pointer. 
    def Set_Initial_Data(self, As_0=None, init_by_vector=False, As0=None, **kwargs):
        """
        Set initial data on the FE spaces 
        """
        assert self.set_rectangular_domain == 1, 'The domain must be defined before.'

        # Expressions
        if not init_by_vector :
            self.As_0  = Expression(As_0, degree=2, x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL, **kwargs)
            self.init_by_vector = False
        else :
            # Vectors
            self.As0            = As0
            self.init_by_vector = True

        self.set_initial_data = 1

        if self.verbose: print(40*'-')
        if self.verbose: print('Initial data: OK')
        if self.verbose: print(40*'-')

        return self.set_initial_data  
    
    #%% SPACE-TIME DISCRETIZATION
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
            self.As0 = interpolate(self.As_0, self.Vs).vector()[:]
        
        if self.project_boundary_control == 1 : print('Project BC: OK')
        
        if self.verbose: print(40*'-')
        if self.verbose: print('Project init data: OK')
        if self.verbose: print(40*'-')
        
        self.project_initial_data = 1
        
        return self.project_initial_data   
    
    
    #%% TIME INTEGRATION    
    ## Method.
    #  @param self The object pointer.  
    def Integration_DAE(self):
        """
        Solution of the DAE system related to the entropy PFEM formulation with a forward in time stepping method 
        """
        # Dimensions
        Nt  = self.Nt
        Ns  = self.Ns
        NV  = self.NV
        Vs  = self.Vs
        
        # Continuous FEniCS functions
        As_ = Function(self.Vs)
        es_ = Function(self.Vs)    
        
        # Prepare vectors and matrices
        Mrho_solver = umfpack.UmfpackLU(csc_matrix(self.Mrho))
        
        if not self.memory_constrained:
            MrhoG   = Mrho_solver.solve(self.G)
            MrhoBG  = Mrho_solver.solve(self.BG)
        
        L_CV        = assemble(self.CV * self.va * dx)[:]   
        MM_solver   = umfpack.UmfpackLU(csc_matrix(self.MM)) 
        
        if not self.memory_constrained:
            MMGT    = MM_solver.solve(self.G.T)                          
        
        # (I) - initialization
        As      = np.zeros((Ns, Nt+1))
        As[:,0] = self.As0
        
        # (II) - initialization
        es                          = np.zeros((Ns, Nt+1))
        As_.vector()[range(Ns)]     = As[range(Ns), 0]
        M_as                        = csc_matrix(assemble(As_ * self.alpha * self.va * dx).array())
        M_as_solver                 = umfpack.UmfpackLU(M_as)                   
        es[:,0]                     = M_as_solver.solve(L_CV)
        
        # (III) - initialization
        fS      = np.zeros((NV, Nt+1))
        if not self.memory_constrained:
            fS[:,0] = -MMGT @ es[:,0]
        else:
            fS[:,0] = - MM_solver.solve(self.my_mult(self.G.T,es[:,0]))

        # (IV) - initialization
        eS                          = np.zeros((NV, Nt+1))
        es_.vector()[range(Ns)]     = es[range(Ns), 0]
        L_es                        = assemble( 1/(es_ * es_) * dot(self.Lambda * grad(es_), self.vA) * dx )[:]
        eS[:,0]                     = MM_solver.solve(L_es)   
   
        for k in range(Nt):
            # (I) - update (time-stepping)
            if not self.memory_constrained:
                As[:,k+1] = As[:,k] + self.dt * (MrhoG @ eS[:,k] + MrhoBG @ self.Ub(self.tspan[k]) ) 
            else:
                As[:,k+1] = As[:,k] + self.dt * Mrho_solver.solve(self.my_mult(self.G,eS[:,k])+self.my_mult(self.BG,self.Ub(self.tspan[k])))
            
            # (II) - update (Dulong-Petit law)
            As_.vector()[range(Ns)]     = As[range(self.Ns), k+1]
            M_as                        = csc_matrix(assemble(As_ * self.alpha * self.va * dx).array())
            M_as_solver                 = umfpack.UmfpackLU(M_as)                   
            es[:,k+1]                   = M_as_solver.solve(L_CV)
                        
            # (III) - update (force)
            if not self.memory_constrained:
                fS[:,k+1]               = - MMGT @ es[:,k+1]
            else:
                fS[:,k+1]               = - MM_solver.solve(self.my_mult(self.G.T,es[:,k+1]))
            
            # (IV) - update (Fourier law)
            es_.vector()[range(Ns)]     = es[range(self.Ns), k+1]
            L_es                        = assemble( 1/(es_ * es_) * dot(self.Lambda * grad(es_), self.vA) * dx )[:]
            eS[:,k+1]                   = MM_solver.solve(L_es)                              

            perct = int(k/(Nt-1) * 100)  
            bar = ('Entropy forward-stepping : |' + '#' * int(perct/2) + ' ' + str(perct) + '%' + '|')
            sys.stdout.write('\r' + bar)
            
            
        print('Time integration completed ! \n')
        
        
        
        # Computing Entropy with S(t=0) = 0
        Ham         = np.zeros(Nt+1)
        sigma       = np.zeros(Nt+1)
        InOutput    = np.zeros(Nt+1)
        
        Ham[0]      = 0
        #sigma[0]    = -fS[:,0] @ self.MM @ eS[:,0]
        sigma[0]    = -self.my_mult(fS[:,0].T,self.my_mult(self.MM,eS[:,0]))
        
        #InOutput[0] = self.Ub(self.tspan[0]) @ self.BG.T @ es[:,0]
        InOutput[0] = self.my_mult(self.Ub(self.tspan[0]).T,self.my_mult(self.BG.T,es[:,0]))
        
        for k in range(Nt):
            #sigma[k+1]      = -fS[:,k+1] @ self.MM @ eS[:,k+1] 
            sigma[k+1]      = -self.my_mult(fS[:,k+1].T,self.my_mult(self.MM,eS[:,k+1]))
            
            #InOutput[k+1]   = self.Ub(self.tspan[k+1]) @ self.BG.T @ es[:,k+1]
            InOutput[k+1]   = self.my_mult(self.Ub(self.tspan[k+1]).T,self.my_mult(self.BG.T,es[:,k+1]))
            
            Ham[k+1]        = Ham[k] + self.dt *( InOutput[k+1] + sigma[k+1])

            perct = int(k/(Nt-1) * 100)  
            bar = ('Computing Entropy : |' + '#' * int(perct/2) + ' ' + str(perct) + '%' + '|')
            sys.stdout.write('\r' + bar)
            
        return As, fS, es, eS, Ham

   
class Lyapunov(Heat_2D):
 
    #%% PROBLEM DEFINITION
    ## Method.
    #  @param self The object pointer. 
    def Set_Initial_Data(self, eT_0=None, init_by_vector=False, eT0=None, **kwargs):
        """
        Set initial data on the FE spaces 
        """
        assert self.set_rectangular_domain == 1, 'The domain must be defined before.'
        
        # Expressions
        if not init_by_vector :
            self.eT_0  = Expression(eT_0, degree=2, x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL, **kwargs)
            self.init_by_vector = False
        else :
            # Vectors
            self.eT0            = eT0
            self.init_by_vector = True

        self.set_initial_data = 1
        
        if self.verbose: print(40*'-')
        if self.verbose: print('Initial data: OK')
        if self.verbose: print(40*'-')
        
        return self.set_initial_data 
    
    #%% SPACE-TIME DISCRETIZATION
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
            self.eT0 = interpolate(self.eT_0, self.Vs).vector()[:]
            
        if self.project_boundary_control == 1 : print('Project BC: OK')
       
        if self.verbose: print(40*'-')    
        if self.verbose: print('Project initial data: OK')
        if self.verbose: print(40*'-')
        
        self.project_initial_data = 1
        
        return self.project_initial_data  
    
    
    #%% TIME INTEGRATION
    ## Method.
    #  @param self The object pointer.     
    def Integration_DAE(self):
        """
        Solution of the DAE system related to the Lyapunov PFEM formulation
        """
        assert self.set_formulation == 1, "The PFEM formulation must be set before !"
        
        global order
        order = []
        
        # Defines the residual
        Ns  = self.Ns
        NV  = self.NV
        def DAE_grad(t,y,yd):
            res_0 = np.zeros(Ns)
            res_1 = np.zeros(NV)
            res_2 = np.zeros(NV)
            
            res_0 = self.MrhoCV @ yd[:Ns] - self.G @ y[Ns+NV:] - self.BG @ self.Ub(t) 
            res_1 = self.G.T @ y[:Ns] + self.MM @ y[Ns:Ns+NV]
            res_2 = - self.LAMBDA @ y[Ns:Ns+NV] + self.MM @ y[Ns+NV:]
            
            return np.concatenate((res_0, res_1, res_2))

        def DAE_div(t,y,yd):
            res_0 = np.zeros(Ns)
            res_1 = np.zeros(NV)
            res_2 = np.zeros(NV)
            
            res_0 = self.MrhoCV @ yd[:Ns] - self.D @ y[Ns+NV:] 
            res_1 = self.D.T @ y[:Ns] + self.MM @ y[Ns:Ns+NV] - self.BD @ self.Ub(t) 
            res_2 = - self.LAMBDA @ y[Ns:Ns+NV] + self.MM @ y[Ns+NV:]
            
            return np.concatenate((res_0, res_1, res_2))
        
        def handle_result(solver, t ,y, yd):
            global order
            order.append(solver.get_last_order())
            
            solver.t_sol.extend([t])
            solver.y_sol.extend([y])
            solver.yd_sol.extend([yd])   
        
        #The initial conditons
        MM_facto    = umfpack.UmfpackLU(csc_matrix(self.MM))    
        eT_0        = self.eT0
        fQ_0        = MM_facto.solve(csc_matrix(assemble( - dot(grad(self.alpha), self.vA) * dx).array())) @ eT_0 
        eQ_0        = MM_facto.solve(csc_matrix(assemble( - dot(self.Lambda * grad(self.alpha), self.vA) * dx).array())) @ eT_0 
        
        
        y0  = np.concatenate(( eT_0, fQ_0, eQ_0 ))#Initial conditions
        yd0 = np.zeros(Ns+NV+NV) #Initial conditions
            
        # Create an Assimulo implicit problem
        if self.formulation == 'grad':
            imp_mod = Implicit_Problem(DAE_grad, y0, yd0, name='Lyapunov-DAE')
        if self.formulation == 'div':
            imp_mod = Implicit_Problem(DAE_div, y0, yd0, name='Lyapunov-DAE')

        imp_mod.handle_result = handle_result
           
        #Set the algebraic components
        imp_mod.algvar = np.ndarray.tolist( np.concatenate(( np.ones(Ns), np.zeros(NV+NV) )) ) 
        
        #Create an Assimulo implicit solver (IDA)
        imp_sim = IDA(imp_mod) #Create a IDA solver
         
        #Sets the paramters
        imp_sim.atol = 1e-6 #Default 1e-6
        imp_sim.rtol = 1e-6 #Default 1e-6
        #imp_sim.maxh = dt
        #imp_sim.maxsteps = Nt
        #imp_sim.clock_step = dt
        imp_sim.suppress_alg = True #Suppres the algebraic variables on the error test
        imp_sim.report_continuously = True
             
        #Let Sundials find consistent initial conditions by use of 'IDA_YA_YDP_INIT'
        imp_sim.make_consistent('IDA_YA_YDP_INIT')
         
        #Simulate
        t, y, yd    = imp_sim.simulate(self.tfinal, self.tinit, self.tspan ) 
        eT          = y[:,:Ns].T
        eQ          = y[:,Ns+NV:].T
        Ham         = np.array([ 1/2 * eT[:,k] @ self.MrhoCV @ eT[:,k] for k in range(self.Nt+1) ])
        
        return eT, eQ, Ham

    
    ## Method.
    #  @param self The object pointer.  
    def Integration_DAE_FEM(self):
        """
        Solution of the DAE system related to the Lyapunov FEM formulation
        [To check]
        """
        M   = self.MrhoCV
        A   = assemble( dot(self.Lambda * grad(self.alpha), grad(self.va)) * dx ).array()

        if self.formulation == 'div':
            C   = assemble( self.ab * self.va * ds ).array()[:,self.b_ind]
        
        if self.formulation == 'grad':
            L0  = assemble( self.Ub_sp0_Expression * self.va * ds)[:]
            L1  = assemble( self.Ub_sp1_Expression * self.va * ds)[:]
        
        global order
        order = []
        
        # Defines the residual
        Ns  = self.Ns
        def Problem_div(t,y,yd):
            res_0 = np.zeros(Ns)
            res_1 = np.zeros(Ns)
            
            res_0 = M @ yd[:Ns] + A @ y[:Ns] + C @ y[Ns:] 
            res_1 = - C.T @ y[:Ns] + self.Mb @ self.Ub(t)
#            res_1 = - C.T @ yd[:Ns] + self.Mb @ self.Ubder(t)

            return np.concatenate((res_0, res_1))
        
        def Problem_grad(t,y,yd):
            res_0 = np.zeros(Ns)
            
            res_0 = M @ yd[:Ns] + A @ y[:Ns] - (L0 * self.Ub_tm0(t) + L1 + np.ones(Ns) * self.Ub_tm1(t))  

            return res_0

        #The initial conditons
        eT_0        = self.eT0
           
        
        if self.formulation == 'div':
            eT_0_       = Function(self.Vs);    eT_0_.vector()[range(Ns)] = eT_0[range(Ns)]
            lag_0       = linalg.solve(self.Mb.toarray(), assemble(dot(grad(eT_0_), self.norext) * self.vb * ds)[self.b_ind])

            y0 =  np.concatenate(( eT_0, lag_0))#Initial conditions
            yd0 = np.zeros(Ns+self.Nb) #Initial conditions
            
            #Create an Assimulo implicit problem
            imp_mod = Implicit_Problem(Problem_div, y0, yd0, name='Problem_div')
           
            #Set the algebraic components
            imp_mod.algvar = np.ndarray.tolist( np.concatenate(( np.ones(Ns), np.zeros(self.Nb) )) ) 
        
        if self.formulation == 'grad':
            y0  =  eT_0 #Initial conditions
            yd0 = np.zeros(Ns) #Initial conditions
       
            #Create an Assimulo implicit problem
            imp_mod = Implicit_Problem(Problem_grad, y0, yd0, name='Problem_grad')
           
        
        #Create an Assimulo implicit solver (IDA)
        imp_sim = IDA(imp_mod) #Create a IDA solver
         
        #Sets the paramters
        imp_sim.suppress_alg = True #Suppres the algebraic variables on the error test
        imp_sim.report_continuously = True
             
        #Let Sundials find consistent initial conditions by use of 'IDA_YA_YDP_INIT'
        imp_sim.make_consistent('IDA_YA_YDP_INIT')
         
        #Simulate
        t, y, yd    = imp_sim.simulate(self.tfinal, self.tinit, self.tspan ) 
        
        eT          = y[:,:Ns].T
        
        MM_facto    = umfpack.UmfpackLU(csc_matrix(self.MM))    
        eQ          = MM_facto.solve(csc_matrix(assemble( - dot(self.Lambda * grad(self.alpha), self.vA) * dx).array())) @ eT
        
        Ham         = np.array([ 1/2 * eT[:,k] @ self.MrhoCV @ eT[:,k] for k in range(self.Nt+1) ])
        
        return eT, eQ, Ham        
    

    
    ## Method.
    #  @param self The object pointer.       
    def Integration_ODE_RK4(self):
        """
        4th order Runge-Kutta scheme for the numerical integration of the ODE system
        related to the Lyapunov PFEM formulation 
        """
        assert self.set_time_setting == 1, 'Time discretization must be specified first'
        assert self.set_formulation  == 1, "The PFEM formulation must be set before !"

        
        tspan = self.tspan
        dt    = self.dt
        
        self.BD         = csr_matrix(self.BD)
        self.BG         = csr_matrix(self.BG)
        self.D          = csr_matrix(self.D)
        self.G          = csr_matrix(self.G)
        self.LAMBDA     = csr_matrix(self.LAMBDA)
        
        MrhoCV_sparse   = csc_matrix(self.MrhoCV)
        MM_sparse       = csc_matrix(self.MM)

        MrhoCV_solver   = umfpack.UmfpackLU(MrhoCV_sparse)
        MM_solver       = umfpack.UmfpackLU(MM_sparse)
        
        if not self.memory_constrained:   
            if self.formulation == 'grad' :
                temp    = MM_solver.solve(self.G.T)
                ODE_A   = - MrhoCV_solver.solve(temp.T @ self.LAMBDA @ temp) 
                ODE_B   = MrhoCV_solver.solve(self.BG)
                
                def dif_func(t,y):
                    return ODE_A @ y + ODE_B @ self.Ub(t)

            elif self.formulation == 'div' :
                temp    = MM_solver.solve(self.D.T)
                ODE_A   = - MrhoCV_solver.solve(temp.T @ self.LAMBDA @ temp) 
                ODE_B   = MrhoCV_solver.solve( temp.T @ MM_solver.solve(self.LAMBDA) @ self.BD )
                def dif_func(t,y):
                    return ODE_A @ y + ODE_B @ self.Ub(t)

        else:
            if self.formulation == 'grad' :
                def dif_func(t,y):
                    z       = MM_solver.solve(self.my_mult(self.G.T,y))
                    w       = MM_solver.solve(self.my_mult(self.LAMBDA,z))
                    return MrhoCV_solver.solve(self.my_mult(self.BG,self.Ub(t))-self.my_mult(self.G,w))

            elif self.formulation == 'div' :
                def dif_func(t,y):
                    w       = self.my_mult(self.BD,self.Ub(t)) - self.my_mult(self.D.T,y) 
                    z       = MM_solver.solve(w)
                    w       = self.my_mult(self.LAMBDA,z)
                    z       = MM_solver.solve(w)
                    return MrhoCV_solver.solve(self.my_mult(self.D,z))
            
        eT  = np.zeros((self.Ns, self.Nt+1))
        Ham = np.zeros(self.Nt+1)
        
        eT[:,0] = self.eT0
        #Ham[0] = 1/2 * eT[:,0] @ self.MrhoCV @  eT[:,0] 
        Ham[0]  = 1/2 * self.my_mult(eT[:,0].T,self.my_mult(self.MrhoCV,eT[:,0]))
         
        for k in range(self.Nt):
            k1 = dif_func(tspan[k], eT[:,k])
            k2 = dif_func(tspan[k] + dt/2, eT[:,k] + dt/2 * k1)
            k3 = dif_func(tspan[k] + dt/2, eT[:,k] + dt/2 * k2)
            k4 = dif_func(tspan[k] + dt, eT[:,k] + dt * k3)
            
            eT[:,k+1] = eT[:,k] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            #Ham[k+1] = 1/2 * eT[:,k+1] @ self.MrhoCV @ eT[:,k+1]
            Ham[k+1] = 1/2 * self.my_mult(eT[:,k+1].T,self.my_mult(self.MrhoCV,eT[:,k+1]))
        
            # Progress bar
            perct = int(k/(self.Nt-1) * 100)  
            bar   = ('Time-stepping RK4 : |' + '#' * int(perct/2) + ' ' + str(perct) + '%' + '|')
            sys.stdout.write('\r' + bar)
            
        
        
        eQ  = MM_solver.solve(self.my_mult(csc_matrix(assemble( - dot(self.Lambda * grad(self.alpha), self.vA) * dx).array()),eT))
        
        return eT, eQ, Ham
    
    
     
if __name__ == '__main__':
    
    from math import sin, pi

    E       = Energy()
    
    ### Geometric domain
    
    x0, xL = 0, 2
    y0, yL = 0, 1
    E.Set_Rectangular_Domain(x0, xL, y0, yL)

    ### Physical parameters
    
    rho         = 'x[0]*(xL-x[0]) + x[1]*(yL-x[1]) + 5'
    Lambda11    = '5 + x[0]*x[1]'
    Lambda12    = '(x[0]-x[1]) * (x[0]-x[1])'
    Lambda22    = '3 + x[1]/(1+x[0])'
    CV          = '3'
    
    E.Set_Physical_Parameters(rho=rho, Lambda11=Lambda11, Lambda12=Lambda12, Lambda22=Lambda22, CV=CV)

    ### Time final
    
    tf  = 10
    dt  = 1e-3
    E.Set_Initial_Final_Time(initial_time=0, final_time=tf)


    ### Mesh & FE families & Assembly & Time step
    
    E.Set_Gmsh_Mesh(xmlfile='rectangle.xml', rfn_num=1)
    E.Set_Finite_Element_Spaces(family_scalar='P', family_Vector='RT', family_boundary='P', rs=1, rV=0, rb=1)
    E.Assembly()
    E.Set_Time_Setting(time_step=dt)
    tspan  = E.tspan


    ### Initial data
    
    levinit, ampl   = 100, 25
    sX, sY, X0, Y0  = E.xL/4, E.yL/4, E.xL/2, E.yL/2 
    Gaussian_init   = Expression(' ampl * exp(- pow( (x[0]-X0)/sX, 2) - pow( (x[1]-Y0)/sY, 2) ) + levinit', 
                                     degree=2, levinit=levinit, ampl=ampl, sX=sX, sY=sY, X0=X0, Y0=Y0)
    Ginit           = interpolate(Gaussian_init, E.Vs)  
    E.Set_Initial_Data(init_by_vector=True, au0=100*np.ones(E.Ns), eu0=  Ginit.vector()[:]) 

    Ub_sp0  = ''' 
        ( abs(x[0]) <= DOLFIN_EPS ? - x[1] * (yL-x[1]) : 0 )            
        + ( abs(xL - x[0]) <= DOLFIN_EPS ? exp(x[1] * (yL-x[1])) -1  : 0 )      
        + ( abs(x[1]) <= DOLFIN_EPS ?  - x[0] * (xL-x[0])/3  : 0 )            
        + ( abs(yL - x[1]) <= DOLFIN_EPS ?  pow(x[0],2) * (xL-x[0])/4  : 0 )     
        '''
    Ub_tm0  = lambda t : sin(2 * 2*pi/tf * t) * 10 * 0

    E.Set_Boundary_Control(Ub_tm0=Ub_tm0, Ub_sp0=Ub_sp0, Ub_tm1=lambda t : 0, Ub_sp1='0', levinit=levinit)
    E.Project_Boundary_Control()

    ### Space-time simulation
    
    E.Set_Formulation('div')
    au, fU, fsig, eu, eU, esig, Ham = E.Integration_DAE()

    E.Plot_Hamiltonian(tspan, Ham)
    
