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

n_mesh = 7

ind_ref = 15
R_ext = 1
H_file = 'H_dae_' + str(ind_ref) + '.npy'
Hp_file = 'Hp_dae_' + str(ind_ref) + '.npy'
Hq_file = 'Hq_dae_' + str(ind_ref) + '.npy'

t_file = 't_dae_' + str(ind_ref) + '.npy'

path_result = '/home/a.brugnoli/LargeFiles/results_ifacwc_CN2/'
path_mesh = "/home/a.brugnoli/GitProjects/PythonProjects/ph_fenics/Wave_fenics/meshes_ifacwc/"
path_figs = "/home/a.brugnoli/Plots/Python/Plots/Waves/IFAC_WC2020_CN2/"

#path_result_ref = '/home/a.brugnoli/LargeFiles/results_ifacwc2_fenics/'
path_result_ref = path_result


H_ref = np.load(path_result_ref + H_file)
Hp_ref = np.load(path_result_ref + Hp_file)
Hq_ref = np.load(path_result_ref + Hq_file)
t_ref = np.load(path_result_ref + t_file)

n_t = len(t_ref)

ep_file = 'ep_dae_' + str(ind_ref) + '.npy'
eq_file = 'eq_dae_' + str(ind_ref) + '.npy'
ep_ref = np.load(path_result_ref + ep_file)
eq_ref = np.load(path_result_ref + eq_file)

mesh_file = 'duct_' + str(ind_ref) + '.xml'
mesh_ref = Mesh(path_mesh + mesh_file)
#plot(mesh_ref); plt.show()

Vp_ref = FunctionSpace(mesh_ref, "CG", 1)
Vq_ref = VectorFunctionSpace(mesh_ref, "DG", 0)

#Vq_ref = FunctionSpace(mesh_ref, "RT", 1)

dx_ref = Measure('dx', domain=mesh_ref)
ds_ref = Measure('ds', domain=mesh_ref)

mesh1_ref = Mesh(path_mesh + "duct_dom1_" + str(ind_ref) + ".xml")
Vp1_ref = FunctionSpace(mesh1_ref, "CG", 1)
#Vq1_ref = FunctionSpace(mesh1_ref, "RT", 1)
Vq1_ref = VectorFunctionSpace(mesh1_ref, "DG", 0)

mesh2_ref = Mesh(path_mesh + "duct_dom2_" + str(ind_ref) + ".xml")
Vp2_ref = FunctionSpace(mesh2_ref, "CG", 1)
#Vq2_ref = FunctionSpace(mesh2_ref, "RT", 1)
Vq2_ref = VectorFunctionSpace(mesh2_ref, "DG", 0)

n_Vp1_ref = Vp1_ref.dim()
n_Vp2_ref = Vp2_ref.dim()

n_Vp_ref = n_Vp1_ref + n_Vp2_ref

n_Vq1_ref = Vq1_ref.dim()
n_Vq2_ref = Vq2_ref.dim()

n_Vq_ref = n_Vq1_ref + n_Vq2_ref

dx1_ref = Measure('dx', domain=mesh1_ref)
ds1_ref = Measure('ds', domain=mesh1_ref)

dx2_ref = Measure('dx', domain=mesh2_ref)
ds2_ref = Measure('ds', domain=mesh2_ref)


t_elapsed_dae = np.zeros(n_mesh)
t_elapsed_ode = np.zeros(n_mesh)

errHdae = np.zeros(n_mesh)
errHode = np.zeros(n_mesh)

h_vec = np.zeros(n_mesh)
plt.figure(0)
plt.xlabel(r'Time $\mathrm{[s]}$')
plt.ylabel(r'$H_{DAE}$')
plt.title(r"Hamiltonian given by DAE")

plt.figure(1)
plt.xlabel(r'Time $\mathrm{[s]}$')
plt.ylabel(r'$H_{ODE}$')
plt.title(r"Hamiltonian given by ODE")

err_ep_ode = np.zeros(n_mesh)
err_eq_ode = np.zeros(n_mesh)

err_ep_dae = np.zeros(n_mesh)
err_eq_dae = np.zeros(n_mesh)

fun_ep_ref_t = Function(Vp_ref)
fun_eq_ref_t = Function(Vq_ref)

for i in range(n_mesh):

    mesh_ind = 4 + i

    mesh_i = Mesh(path_mesh + "duct_" + str(mesh_ind) + ".xml")

    dx_i = Measure('dx', domain=mesh_i)
    ds_i = Measure('ds', domain=mesh_i)

    Vp_i = FunctionSpace(mesh_i, "CG", 1)
    Vq_i = VectorFunctionSpace(mesh_i, "DG", 0)

#    Vq_i = FunctionSpace(mesh_i, "RT", 1)

    file_ep_dae_i = 'ep_dae_' + str(mesh_ind) + '.npy'
    file_eq_dae_i = 'eq_dae_' + str(mesh_ind) + '.npy'
    ep_dae_i = np.load(path_result + file_ep_dae_i)
    eq_dae_i = np.load(path_result + file_eq_dae_i)

    err_ep_dae_allt = np.zeros(n_t)
    err_eq_dae_allt = np.zeros(n_t)
    
    norm_ep_ref_allt = np.zeros(n_t)
    norm_eq_ref_allt = np.zeros(n_t)

    fun_ep_dae_t = Function(Vp_i)
    fun_eq_dae_t = Function(Vq_i)
    
    mesh1_i = Mesh(path_mesh + "duct_dom1_" + str(mesh_ind) + ".xml")
    Vp1_i = FunctionSpace(mesh1_i, "DG", 0)
    Vq1_i = FunctionSpace(mesh1_i, "RT", 1)

    mesh2_i = Mesh(path_mesh + "duct_dom2_" + str(mesh_ind) + ".xml")
    Vp2_i = FunctionSpace(mesh2_i, "CG", 1)
    Vq2_i = VectorFunctionSpace(mesh2_i, "DG", 0)
#    Vq2_i = FunctionSpace(mesh2_i, "RT", 1)
    
    dx1_i = Measure('dx', domain=mesh1_i)    
    dx2_i = Measure('dx', domain=mesh2_i)
    
#    ds1_i = Measure('ds', domain=mesh1_i)
#    ds2_i = Measure('ds', domain=mesh2_i)
    
#    plot(mesh1_i)
#    plot(mesh2_i)
#    plt.show()
    
    n_Vp1_i = Vp1_i.dim()
    n_Vp2_i = Vp2_i.dim()

    n_Vp_i = n_Vp1_i + n_Vp2_i
    
    n_Vq1_i = Vq1_i.dim()
    n_Vq2_i = Vq2_i.dim()

    n_Vq_i = n_Vq1_i + n_Vq2_i
    
    file_ep_ode_i = 'ep_ode_' + str(mesh_ind) + '.npy'
    file_eq_ode_i = 'eq_ode_' + str(mesh_ind) + '.npy'
    ep_ode_i = np.load(path_result + file_ep_ode_i)
    eq_ode_i = np.load(path_result + file_eq_ode_i)
    
    ep1_ode_i = ep_ode_i[:n_Vp1_i, :]
    ep2_ode_i = ep_ode_i[n_Vp1_i:, :]

    eq1_ode_i = eq_ode_i[:n_Vq1_i, :]
    eq2_ode_i = eq_ode_i[n_Vq1_i:, :]

    
    fun_ep1_ode_t = Function(Vp1_i)
    fun_ep2_ode_t = Function(Vp2_i)

    fun_eq1_ode_t = Function(Vq1_i)
    fun_eq2_ode_t = Function(Vq2_i)
    
    err_ep_ode_allt = np.zeros(n_t)
    err_eq_ode_allt = np.zeros(n_t)
        
    for j in range(n_t):
              
        fun_ep_dae_t.vector()[range(Vp_i.dim())] = ep_dae_i[range(Vp_i.dim()), j]
        fun_eq_dae_t.vector()[range(Vq_i.dim())] = eq_dae_i[range(Vq_i.dim()), j]
        fun_ep_ref_t.vector()[range(Vp_ref.dim())] = ep_ref[range(Vp_ref.dim()), j]     
        fun_eq_ref_t.vector()[range(Vq_ref.dim())] = eq_ref[range(Vq_ref.dim()), j]
        
#        fun_ep_dae_t.vector()[range(Vp_i.dim())] = np.ascontiguousarray(ep_dae_i[range(Vp_i.dim()), j])   
#        fun_eq_dae_t.vector()[range(Vq_i.dim())] = np.ascontiguousarray(eq_dae_i[range(Vq_i.dim()), j])
#        fun_ep_ref_t.vector()[range(Vp_ref.dim())] = np.ascontiguousarray(ep_ref[range(Vp_ref.dim()), j])       
#        fun_eq_ref_t.vector()[range(Vq_ref.dim())] = np.ascontiguousarray(eq_ref[range(Vq_ref.dim()), j])

#        fun_ep_dae_t.vector()[:] = np.ascontiguousarray(ep_dae_i[:, j])   
#        fun_eq_dae_t.vector()[:] = np.ascontiguousarray(eq_dae_i[:, j])
#        fun_ep_ref_t.vector()[:] = np.ascontiguousarray(ep_ref[:, j])       
#        fun_eq_ref_t.vector()[:] = np.ascontiguousarray(eq_ref[:, j])
#
        int_ep_ref_t = Function(Vp_ref)
        int_ep_ref_t.interpolate(fun_ep_ref_t)
        
        int_ep_dae_t = Function(Vp_ref)
        int_ep_dae_t.interpolate(fun_ep_dae_t)

        errp_dae_t = np.sqrt(assemble(inner(int_ep_dae_t - int_ep_ref_t, int_ep_dae_t - int_ep_ref_t) * dx_ref))
        normp_ref_t = np.sqrt(assemble(inner(int_ep_ref_t, int_ep_ref_t) * dx_ref))
        
        err_ep_dae_allt[j] = errp_dae_t
        norm_ep_ref_allt[j] = normp_ref_t
        
#        int_eq_ref_t = Function(Vq_ref)
        int_eq_ref_t = interpolate(fun_eq_ref_t, Vq_ref)
        
#        int_eq_ref_t = Function(Vq_ref)
        int_eq_dae_t = interpolate(fun_eq_dae_t, Vq_ref)

        errq_dae_t = np.sqrt(assemble(inner(int_eq_dae_t - int_eq_ref_t, int_eq_dae_t - int_eq_ref_t) * dx_ref))
        normq_ref_t = np.sqrt(assemble(inner(int_eq_ref_t, int_eq_ref_t) * dx_ref))        
        
        err_eq_dae_allt[j] = errq_dae_t
        norm_eq_ref_allt[j] = normq_ref_t
                
        fun_ep1_ode_t.vector()[range(n_Vp1_i)] = ep1_ode_i[range(n_Vp1_i), j]
        fun_ep2_ode_t.vector()[range(n_Vp2_i)] = ep2_ode_i[range(n_Vp2_i), j]

        fun_eq1_ode_t.vector()[range(n_Vq1_i)] = eq1_ode_i[range(n_Vq1_i), j]
        fun_eq2_ode_t.vector()[range(n_Vq2_i)] = eq2_ode_i[range(n_Vq2_i), j]
        
        int_ep1_ode_t = interpolate(fun_ep1_ode_t, Vp1_ref)
        int_ep2_ode_t = interpolate(fun_ep2_ode_t, Vp2_ref)
        
        int_ep1_ref_t = Function(Vp1_ref)
        int_ep2_ref_t = Function(Vp2_ref)
        
        int_ep1_ref_t = interpolate(fun_ep_ref_t, Vp1_ref)
        int_ep2_ref_t = interpolate(fun_ep_ref_t, Vp2_ref)
        
#        LagrangeInterpolator.interpolate(int_ep1_ref_t, int_ep_ref_t)
#        LagrangeInterpolator.interpolate(int_ep2_ref_t, int_ep_ref_t)
   
        int_eq1_ode_t = interpolate(fun_eq1_ode_t, Vq1_ref)
        int_eq2_ode_t = interpolate(fun_eq2_ode_t, Vq2_ref)
        
        int_eq1_ref_t = Function(Vq1_ref)
        int_eq2_ref_t = Function(Vq2_ref)
        
        int_eq1_ref_t = interpolate(fun_eq_ref_t, Vq1_ref)
        int_eq2_ref_t = interpolate(fun_eq_ref_t, Vq2_ref)
        
#        LagrangeInterpolator.interpolate(int_eq1_ref_t, int_eq_ref_t)
#        LagrangeInterpolator.interpolate(int_eq2_ref_t, int_eq_ref_t)
        
        
        errp1_ode_t = assemble(inner(int_ep1_ode_t - int_ep1_ref_t,\
                                            int_ep1_ode_t - int_ep1_ref_t) * dx1_ref)
        
        errp2_ode_t = assemble(inner(int_ep2_ode_t - int_ep2_ref_t,\
                                            int_ep2_ode_t - int_ep2_ref_t) * dx2_ref)
        
        errp_ode_t = np.sqrt(errp1_ode_t + errp2_ode_t)
        
        err_ep_ode_allt[j] = errp_ode_t
        
        errq1_ode_t = assemble(inner(int_eq1_ode_t - int_eq1_ref_t,\
                                            int_eq1_ode_t - int_eq1_ref_t) * dx1_ref)
        
        errq2_ode_t = assemble(inner(int_eq2_ode_t - int_eq2_ref_t,\
                                            int_eq2_ode_t - int_eq2_ref_t) * dx2_ref)
        
        errq_ode_t = np.sqrt(errq1_ode_t + errq2_ode_t)
        
        err_eq_ode_allt[j] = errq_ode_t
        

    nt_fin = n_t
    err_ep_dae[i] = np.linalg.norm(err_ep_dae_allt[:nt_fin])/np.linalg.norm(norm_ep_ref_allt[:nt_fin])
    err_eq_dae[i] = np.linalg.norm(err_eq_dae_allt[:nt_fin])/np.linalg.norm(norm_eq_ref_allt[:nt_fin])
    
    err_ep_ode[i] = np.linalg.norm(err_ep_ode_allt[:nt_fin])/np.linalg.norm(norm_ep_ref_allt[:nt_fin])
    err_eq_ode[i] = np.linalg.norm(err_eq_ode_allt[:nt_fin])/np.linalg.norm(norm_eq_ref_allt[:nt_fin])


    t_elapsed_dae[i] = np.load(path_result + "t_elapsed_dae_" + str(mesh_ind) + ".npy")
    t_elapsed_ode[i] = np.load(path_result + "t_elapsed_ode_" + str(mesh_ind) + ".npy")
    
    dae_file_i = 'H_dae_' + str(mesh_ind) + '.npy'
    ode_file_i = 'H_ode_' + str(mesh_ind) + '.npy'

    H_dae_i = np.load(path_result + dae_file_i)
    H_ode_i = np.load(path_result + ode_file_i)

    plt.figure(0)
    plt.plot(t_ref, H_dae_i, label=r'$h = R/$' + str(mesh_ind))

    plt.figure(1)
    plt.plot(t_ref, H_ode_i, label=r'$h = R/$' + str(mesh_ind))

    errHdae[i] = np.sqrt(np.linalg.norm(H_ref - H_dae_i, np.inf))/np.sqrt(np.linalg.norm(H_ref))
    errHode[i] = np.sqrt(np.linalg.norm(H_ref - H_ode_i, np.inf))/np.sqrt(np.linalg.norm(H_ref))

#    errHdae[i] = np.sum(np.abs(H_ref - H_dae_i))/np.sum(np.abs(H_ref))
#    errHode[i] = np.sum(np.abs(H_ref - H_ode_i))/np.sum(np.abs(H_ref))
    h_vec[i] = R_ext/mesh_ind



plt.figure(0)
plt.plot(t_ref, H_ref, label=r'$h_{REF} = R/$' + str(15))
plt.legend(loc='upper right')
plt.savefig(path_figs + "Hdae_all.eps", format="eps")

plt.figure(1)
plt.plot(t_ref, H_ref, label=r'$h_{REF} = R/$' + str(15))
plt.legend(loc='upper right')
plt.savefig(path_figs + "Hode_all.eps", format="eps")

plt.figure(2)
plt.plot(t_ref, H_ref, label=r'$H$')
plt.plot(t_ref, Hp_ref, label=r'$H_p$')
plt.plot(t_ref, Hq_ref, label=r'$H_v$')
plt.xlabel(r'Time $\mathrm{[s]}$')
plt.ylabel(r'Hamiltonian $\mathrm{[J]}$')
plt.title(r"Reference Hamiltonian")
plt.legend(loc='upper right')
plt.savefig(path_figs + "Href.eps", format="eps")

# np.save("Hdae_err.npy", errHdae)
# np.save("Hode_err.npy", errHode)

#fig = plt.figure()
#plt.plot(h_vec, errHdae, 'b-')
#plt.xlabel(r'Mesh size', fontsize=fntsize)
#plt.ylabel(r'$||H_{ref} - H_{dae}||_{L^2}$', fontsize=fntsize)
#plt.title(r"$L^2$ norm Hamiltonian difference", fontsize=fntsize)
# # plt.legend(loc='upper left')
##
#plt.savefig(path_figs + "Hdae_diff.eps", format="eps")
##
#fig = plt.figure()
#plt.plot(h_vec, errHode, 'b-')
#plt.xlabel(r'Mesh size', fontsize=fntsize)
#plt.ylabel(r'$||H_{ref} - H_{ode}||_{L^2}$', fontsize=fntsize)
#plt.title(r"$L^2$ norm Hamiltonian difference", fontsize=fntsize)
# # plt.legend(loc='upper left')
##
#plt.savefig(path_figs + "Hode_diff.eps", format="eps")

fig = plt.figure()
plt.plot(h_vec, errHode, 'b-', label="ODE")
plt.plot(h_vec, errHdae, 'r-', label="DAE")
plt.xlabel(r'Mesh size')
plt.ylabel(r'$||H_{REF} - H_{ODE/DAE}||_{L^2}$')
plt.title(r"$L^2$ norm Hamiltonian difference")
# plt.yscale('log')
plt.legend(loc='upper left')
plt.savefig(path_figs + "Hall_diff.eps", format="eps")
#
fig = plt.figure()
plt.plot(h_vec, err_ep_ode, 'b-', label="ODE")
plt.plot(h_vec, err_ep_dae, 'r-', label="DAE")
plt.xlabel(r'Mesh size')
plt.ylabel(r'$||p_{REF} - p_{DAE}||_{L^2}$')
plt.title(r"$L^2$ norm error on p")
# plt.yscale('log')
plt.legend(loc='upper left')
plt.savefig(path_figs + "err_ep.eps", format="eps")

fig = plt.figure()
plt.plot(h_vec, err_eq_ode, 'b-',  label="ODE")
plt.plot(h_vec, err_eq_dae, 'r-',  label="DAE")
plt.xlabel(r'Mesh size')
plt.ylabel(r'$||v_{REF} - v_{DAE}||_{L^2}$')
plt.title(r"$L^2$ norm error on v")
plt.legend(loc='upper left')

plt.savefig(path_figs + "err_eq.eps", format="eps")

#
plt.show()
