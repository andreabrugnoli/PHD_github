import numpy as np
import matplotlib.pyplot as plt
from fenics import *
import meshio
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
rcParams['text.usetex'] = True

parameters['allow_extrapolation'] = True

path_result = '/home/a.brugnoli/LargeFiles/results_ifacwc2_fenics/'


save = False 
ind_ref = 10

Hdae_file = 'H_dae_' + str(ind_ref) + '.npy'
Hpdae_file = 'Hp_dae_' + str(ind_ref) + '.npy'
Hqdae_file = 'Hq_dae_' + str(ind_ref) + '.npy'
tdae_file = 't_dae_' + str(ind_ref) + '.npy'


H_dae = np.load(path_result + Hdae_file)
Hp_dae = np.load(path_result + Hpdae_file)
Hq_dae = np.load(path_result + Hqdae_file)
t_dae = np.load(path_result + tdae_file)



Hode_file = 'H_ode_' + str(ind_ref) + '.npy'
Hpode_file = 'Hp_ode_' + str(ind_ref) + '.npy'
Hqode_file = 'Hq_ode_' + str(ind_ref) + '.npy'
tode_file = 't_ode_' + str(ind_ref) + '.npy'


H_ode = np.load(path_result + Hode_file)
Hp_ode = np.load(path_result + Hpode_file)
Hq_ode = np.load(path_result + Hqode_file)
t_ode = np.load(path_result + tode_file)


plt.figure()
plt.plot(t_dae, H_dae, label=r'$H DAE$')
plt.plot(t_ode, H_ode, label=r'$H ODE$')

plt.legend(loc='upper right')
plt.show()
if save:
    plt.savefig(path_figs + "H_10.eps", format="eps")