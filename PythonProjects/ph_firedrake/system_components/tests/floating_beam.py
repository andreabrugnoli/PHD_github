import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import sys
sys.path.append("/home/a.brugnoli/GitProjects/PythonProjects/modules_phdae")
sys.path.append("/home/a.brugnoli/GitProjects/PythonProjects/ph_firedrake")
from classes_phsystem import SysPhdaeRig
from system_components.classes_beam import FloatingEB
from scipy.io import savemat

n_el = 2

rho1 = 0.2  # kg/m
EI1 = 1  # N m^2
L1 = 0.5  # m
m_joint1 = 0.5
J_joint1 = 1  # kg/m^2

rho2 = 0.2  # kg/m
EI2 = 1  # N m^2
L2 = 0.5  # m
m_joint2 = 1
J_joint2 = 0.1  # kg/m^2

m_payload = 0.1  # kg
J_payload = 0.5 * 10**-3  # kg/m^2

n_rig = 3

beam = FloatingEB(n_el, rho1, EI1, L1, m_joint=m_joint1, J_joint=J_joint1)

J_ode = beam.J_e
M_ode = beam.M_e
B_ode = beam.B_e
Q_ode = la.inv(M_ode)

pathout = '/home/a.brugnoli/GitProjects/MatlabProjects/PH/FloatingFramePH/PH_matrices/PlanarBeam/'
Qode_file = 'Q_pH'; Jode_file = 'J_pH'; Bode_file = 'B_pH'
savemat(pathout + Qode_file, mdict={Qode_file: Q_ode})
savemat(pathout + Jode_file, mdict={Jode_file: J_ode})
savemat(pathout + Bode_file, mdict={Bode_file: B_ode})