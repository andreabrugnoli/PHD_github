# -*- coding: utf-8 -*-
from fenics import *

QE = FiniteElement('CG',mesh.ufl_cell(),2)
BE = FiniteElement('B',mesh.ufl_cell(),3)
M = FunctionSpace(mesh,QE+BE)
