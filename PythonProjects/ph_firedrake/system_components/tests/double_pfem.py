import numpy as np
import scipy.linalg as la
from system_components.waves import NeumannWave
from modules_phdae.classes_phsystem import SysPhdaeRig
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
from firedrake import *

mesh = Mesh("./meshes/DoubleDom.msh")

wave  = NeumannWave(1, 1, 1, 1, 10, 10, modes=True)
