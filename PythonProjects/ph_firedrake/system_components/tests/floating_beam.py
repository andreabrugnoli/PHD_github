import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from modules_phdae.classes_phsystem import SysPhdaeRig
from system_components.classes_beam import FloatingEB
from scipy.io import savemat
from system_components.tests.parameters import n_el, rho1, EI1, L1, n_rig

beam = FloatingEB(n_el, rho1, EI1, L1, m_joint=0, J_joint=J_joint1)

J_ode = beam.J_e
M_ode = beam.M_e
B_ode = beam.B_e
Q_ode = la.inv(M_ode)

pathout = '/home/a.brugnoli/GitProjects/MatlabProjects/PH/PH_TITOP/Matrices_FreeEB/'
Qode_file = 'Q_pH'; Jode_file = 'J_pH'; Bode_file = 'B_pH'
savemat(pathout + Qode_file, mdict={Qode_file: Q_ode})
savemat(pathout + Jode_file, mdict={Jode_file: J_ode})
savemat(pathout + Bode_file, mdict={Bode_file: B_ode})