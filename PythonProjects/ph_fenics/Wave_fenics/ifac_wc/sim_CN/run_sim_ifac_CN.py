from Wave_fenics.ifac_wc.sim_CN.wavedae_wc_func_CN import computeH_dae
from Wave_fenics.ifac_wc.sim_CN.waveode_wc_func_CN import computeH_ode

for ind in range(4, 11):
    print("DAE: " +str(ind))
    computeH_dae(ind)
    print("DAE " +str(ind) + "completed" )

    print("ODE: " +str(ind))   
    computeH_ode(ind)
    print("ODE " +str(ind) + "completed" )

     
#
computeH_dae(15)
# computeH_ode(15)