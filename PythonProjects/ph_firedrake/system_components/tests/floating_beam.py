import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from modules_phdae.classes_phsystem import SysPhdaeRig
from system_components.beams import FloatingPlanarEB
from scipy.io import savemat
from system_components.tests.manipulator_constants import n_el, rho1, EI1, L1, J_joint1, m_joint1

beam = FloatingPlanarEB(n_el, rho1, EI1, L1, m_joint=m_joint1, J_joint=J_joint1)

J_ode = beam.J_e
M_ode = beam.M_e
B_ode = beam.B_e
Q_ode = la.inv(M_ode)

pathout = '/home/a.brugnoli/GitProjects/MatlabProjects/PH/PH_TITOP/SimpleBeam/Matrices_FreeEB/'
Qode_file = 'Q_pH'; Jode_file = 'J_pH'; Bode_file = 'B_pH'
savemat(pathout + Qode_file, mdict={Qode_file: Q_ode})
savemat(pathout + Jode_file, mdict={Jode_file: J_ode})
savemat(pathout + Bode_file, mdict={Bode_file: B_ode})