import numpy as np
import matplotlib.pyplot as plt
from fenics import *
import meshio

plt.rc('text', usetex=True)

parameters['allow_extrapolation'] = True
ind_ref = 10
R_ext = 1
ref_file = 'H_ode_' + str(ind_ref) + '.npy'
t_file = 't_ode_' + str(ind_ref) + '.npy'

path_result = '/home/a.brugnoli/GitProjects/PythonProjects/ph_fenics/Wave_fenics/results_ifacwc/'
path_mesh = "/home/a.brugnoli/GitProjects/PythonProjects/ph_fenics/Wave_fenics/meshes_ifacwc/"
H_ref = np.load(path_result + ref_file)
t_ref = np.load(path_result + t_file)

n_t = len(t_ref)

ep_file = 'ep_ode_' + str(ind_ref) + '.npy'
eq_file = 'eq_ode_' + str(ind_ref) + '.npy'
ep_ref = np.load(path_result + ep_file)
eq_ref = np.load(path_result + eq_file)

mesh1_ref = Mesh(path_mesh + "duct_dom1_" + str(ind_ref) + ".xml")
Vp1_ref = FunctionSpace(mesh1_ref, "CG", 1)
Vq1_ref = FunctionSpace(mesh1_ref, "RT", 2)

mesh2_ref = Mesh(path_mesh + "duct_dom2_" + str(ind_ref) + ".xml")
Vp2_ref = FunctionSpace(mesh2_ref, "CG", 1)
Vq2_ref = FunctionSpace(mesh2_ref, "RT", 2)

n_Vp1_ref = Vp1_ref.dim()
n_Vp2_ref = Vp2_ref.dim()

n_Vp_ref = n_Vp1_ref + n_Vp2_ref

n_Vq1_ref = Vq1_ref.dim()
n_Vq2_ref = Vq2_ref.dim()

n_Vq_ref = n_Vq1_ref + n_Vq2_ref

ep1_ref= ep_ref[:n_Vp1_ref, :]
ep2_ref= ep_ref[n_Vp1_ref:, :]

eq1_ref= eq_ref[:n_Vq1_ref, :]
eq2_ref= eq_ref[n_Vq1_ref:, :]

dx1_ref = Measure('dx', domain=mesh1_ref)
ds1_ref = Measure('ds', domain=mesh1_ref)

dx2_ref = Measure('dx', domain=mesh2_ref)
ds2_ref = Measure('ds', domain=mesh2_ref)

n_mesh = 7
fntsize = 16

t_elapsed_dae = np.zeros(n_mesh)
t_elapsed_ode = np.zeros(n_mesh)

errHode = np.zeros(n_mesh)

h_vec = np.zeros(n_mesh)
#plt.figure(0)
#plt.xlabel(r'Time (s)', fontsize=fntsize)
#plt.ylabel(r'$H_{DAE}$', fontsize=fntsize)
#plt.title(r"Hamiltonian given by DAE", fontsize=fntsize)
#
#plt.figure(1)
#plt.xlabel(r'Time (s)', fontsize=fntsize)
#plt.ylabel(r'$H_{ODE}$', fontsize=fntsize)
#plt.title(r"Hamiltonian given by ODE", fontsize=fntsize)

err_ep_ode = np.zeros(n_mesh)
err_eq_ode = np.zeros(n_mesh)


fun_ep1_ref_t = Function(Vp1_ref)
fun_eq1_ref_t = Function(Vq1_ref)

fun_ep2_ref_t = Function(Vp2_ref)
fun_eq2_ref_t = Function(Vq2_ref)

for i in range(n_mesh):

    mesh_ind = 4 + i
              
    mesh1_i = Mesh(path_mesh + "duct_dom1_" + str(mesh_ind) + ".xml")
    Vp1_i = FunctionSpace(mesh1_i, "CG", 1)
    Vq1_i = FunctionSpace(mesh1_i, "RT", 2)

    mesh2_i = Mesh(path_mesh + "duct_dom2_" + str(mesh_ind) + ".xml")
    Vp2_i = FunctionSpace(mesh2_i, "CG", 1)
    Vq2_i = FunctionSpace(mesh2_i, "RT", 2)
    
    dx1_i = Measure('dx', domain=mesh1_i)
    ds1_i = Measure('ds', domain=mesh1_i)
    
    dx2_i = Measure('dx', domain=mesh2_i)
    ds2_i = Measure('ds', domain=mesh2_i)
    
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
    
    norm_ep_ref_allt = np.zeros(n_t)
    norm_eq_ref_allt = np.zeros(n_t)
    
    for j in range(n_t):
              
        fun_ep1_ode_t.vector()[range(n_Vp1_i)] = ep1_ode_i[range(n_Vp1_i), j]
        fun_ep2_ode_t.vector()[range(n_Vp2_i)] = ep2_ode_i[range(n_Vp2_i), j]

        fun_eq1_ode_t.vector()[range(n_Vq1_i)] = eq1_ode_i[range(n_Vq1_i), j]
        fun_eq2_ode_t.vector()[range(n_Vq2_i)] = eq2_ode_i[range(n_Vq2_i), j]
        
        fun_ep1_ref_t.vector()[range(n_Vp1_ref)] = ep1_ref[range(n_Vp1_ref), j]         
        fun_ep2_ref_t.vector()[range(n_Vp2_ref)] = ep2_ref[range(n_Vp2_ref), j]     

        fun_eq1_ref_t.vector()[range(n_Vq1_ref)] = eq1_ref[range(n_Vq1_ref), j]
        fun_eq2_ref_t.vector()[range(n_Vq2_ref)] = eq2_ref[range(n_Vq2_ref), j]
        
        int_ep1_ref_t = Function(Vp1_ref)
        int_ep1_ode_t = Function(Vp1_ref)
        int_ep2_ref_t = Function(Vp2_ref)
        int_ep2_ode_t = Function(Vp2_ref)

        int_ep1_ref_t.interpolate(fun_ep1_ref_t)
        int_ep1_ode_t.interpolate(fun_ep1_ode_t)       
        int_ep2_ref_t.interpolate(fun_ep2_ref_t)        
        int_ep2_ode_t.interpolate(fun_ep2_ode_t)

        errp1_ode_t = assemble(inner(int_ep1_ode_t - int_ep1_ref_t,\
                                            int_ep1_ode_t - int_ep1_ref_t) * dx1_ref)
        normp1_ref_t = assemble(inner(int_ep1_ref_t, int_ep1_ref_t) * dx1_ref)
        
        errp2_ode_t = assemble(inner(int_ep2_ode_t - int_ep2_ref_t,\
                                            int_ep2_ode_t - int_ep2_ref_t) * dx2_ref)
        normp2_ref_t = assemble(inner(int_ep2_ref_t, int_ep2_ref_t) * dx2_ref)
        
        errp_ode_t = np.sqrt(errp1_ode_t + errp2_ode_t)
        normp_ref_t = np.sqrt(normp1_ref_t + normp2_ref_t)
        
        err_ep_ode_allt[j] = errp_ode_t
        norm_ep_ref_allt[j] = normp_ref_t
        
#        int_eq1_ref_t = Function(Vq1_ref)
#        int_eq1_ode_t = Function(Vq1_ref)
#        int_eq2_ref_t = Function(Vq2_ref)
#        int_eq2_ode_t = Function(Vq2_ref)

        int_eq1_ref_t = interpolate(fun_eq1_ref_t, Vq1_ref)        
        int_eq1_ode_t = interpolate(fun_eq1_ode_t, Vq1_ref)        
        int_eq2_ref_t = interpolate(fun_eq2_ref_t, Vq2_ref)        
        int_eq2_ode_t = interpolate(fun_eq2_ode_t, Vq2_ref)

        errq1_ode_t = assemble(inner(int_eq1_ode_t - int_eq1_ref_t,\
                                            int_eq1_ode_t - int_eq1_ref_t) * dx1_ref)
        normq1_ref_t = assemble(inner(int_eq1_ref_t, int_eq1_ref_t) * dx1_ref)
        
        errq2_ode_t = assemble(inner(int_eq2_ode_t - int_eq2_ref_t,\
                                            int_eq2_ode_t - int_eq2_ref_t) * dx2_ref)
        normq2_ref_t = assemble(inner(int_eq2_ref_t, int_eq2_ref_t) * dx2_ref)
        
        errq_ode_t = np.sqrt(errq1_ode_t + errq2_ode_t)
        normq_ref_t = np.sqrt(normq1_ref_t + normq2_ref_t)
        
        err_eq_ode_allt[j] = errq_ode_t
        norm_eq_ref_allt[j] = normq_ref_t
        
    nt_fin = 500
    err_ep_ode[i] = np.linalg.norm(err_ep_ode_allt[:nt_fin])/np.linalg.norm(norm_ep_ref_allt[:nt_fin])
    err_eq_ode[i] = np.linalg.norm(err_eq_ode_allt[:nt_fin])/np.linalg.norm(norm_eq_ref_allt[:nt_fin])

#
#    dae_file_i = 'H_dae_' + str(mesh_ind) + '.npy'
#    ode_file_i = 'H_ode_' + str(mesh_ind) + '.npy'
#
#    H_dae_i = np.load(path_result + dae_file_i)
#    H_ode_i = np.load(path_result + ode_file_i)

#    t_elapsed_dae[i] = np.load(path_result + "t_elapsed_dae_" + str(mesh_ind) + ".npy")
#    t_elapsed_ode[i] = np.load(path_result + "t_elapsed_ode_" + str(mesh_ind) + ".npy")

#    plt.figure(0)
#    plt.plot(t_ref, H_dae_i, label=r'$h = R/$' + str(mesh_ind))
#
#    plt.figure(1)
#    plt.plot(t_ref, H_ode_i, label=r'$h = R/$' + str(mesh_ind))

#    errHdae[i] = np.sqrt(np.linalg.norm(H_ref - H_dae_i))#/np.sqrt(np.linalg.norm(H_ref))
#    errHode[i] = np.sqrt(np.linalg.norm(H_ref - H_ode_i))#/np.sqrt(np.linalg.norm(H_ref))

    h_vec[i] = R_ext/mesh_ind


path_figs = "/home/a.brugnoli/Plots_Videos/Python/Plots/Waves/IFAC_WC2020/"

#plt.figure(0)
#plt.plot(t_ref, H_ref, label=r'$h_{REF} = R/$' + str(15))
#plt.legend(loc='upper right')
## plt.savefig(path_figs + "Hdae_all.eps", format="eps")
#
#plt.figure(1)
#plt.plot(t_ref, H_ref, label=r'$h_{REF} = R/$' + str(15))
#plt.legend(loc='upper right')
## plt.savefig(path_figs + "Hode_all.eps", format="eps")
#
#
## np.save("Hdae_err.npy", errHdae)
## np.save("Hode_err.npy", errHode)
#
## fig = plt.figure()
## plt.plot(h_vec, errHdae, 'b-')
## plt.xlabel(r'Mesh size', fontsize=fntsize)
## plt.ylabel(r'$||H_{ref} - H_{dae}||_{L^2}$', fontsize=fntsize)
## plt.title(r"$L^2$ norm Hamiltonian difference", fontsize=fntsize)
## # plt.legend(loc='upper left')
##
## plt.savefig(path_figs + "Hdae_diff.eps", format="eps")
##
## fig = plt.figure()
## plt.plot(h_vec, errHode, 'b-')
## plt.xlabel(r'Mesh size', fontsize=fntsize)
## plt.ylabel(r'$||H_{ref} - H_{ode}||_{L^2}$', fontsize=fntsize)
## plt.title(r"$L^2$ norm Hamiltonian difference", fontsize=fntsize)
## # plt.legend(loc='upper left')
##
## # plt.savefig(path_figs + "Hode_diff.eps", format="eps")
#
#fig = plt.figure()
#plt.plot(h_vec, errHode, 'b-', label="ODE")
#plt.plot(h_vec, errHdae, 'r-', label="DAE")
#plt.xlabel(r'Mesh size', fontsize=fntsize)
#plt.ylabel(r'$||H_{REF} - H_{ODE/DAE}||_{L^2}$', fontsize=fntsize)
#plt.title(r"$L^2$ norm Hamiltonian difference", fontsize=fntsize)
## plt.yscale('log')
#plt.legend(loc='upper left')

fig = plt.figure()
plt.plot(h_vec, err_ep_ode, 'b-', )
plt.xlabel(r'Mesh size', fontsize=fntsize)
plt.ylabel(r'$||p_{REF} - p_{DAE}||_{L^2}$', fontsize=fntsize)
plt.title(r"$L^2$ norm error on p", fontsize=fntsize)
# plt.yscale('log')
#plt.legend(loc='upper left')
# plt.savefig(path_figs + "err_ep.eps", format="eps")

fig = plt.figure()
plt.plot(h_vec, err_eq_ode, 'b-', )
plt.xlabel(r'Mesh size', fontsize=fntsize)
plt.ylabel(r'$||v_{REF} - v_{DAE}||_{L^2}$', fontsize=fntsize)
plt.title(r"$L^2$ norm error on q", fontsize=fntsize)
#
plt.show()