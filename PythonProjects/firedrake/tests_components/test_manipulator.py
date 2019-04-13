from FloatingEB import FloatingEB
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

n_el = 3

J_joint1 = 0.1  # kg/m^2
J_joint2 = 0.1  # kg/m^2

m_joint2 = 1
rho1 = 0.2  # kg/m
rho2 = 0.2  # kg/m

EI1 = 1  # N m^2
EI2 = 1  # N m^2

L1 = 0.5  # m
L2 = 0.5  # m

m_payload = 0.1  # kg
J_payload = 0.5 * 10**-3  # kg/m^2


beam1 = FloatingEB(n_el, rho1, EI1, L1, J_joint=J_joint1)

plt.spy(beam1.J); plt.show()