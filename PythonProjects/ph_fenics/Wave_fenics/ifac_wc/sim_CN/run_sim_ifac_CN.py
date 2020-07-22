from Wave_fenics.ifac_wc.sim_CN.wavedae_wc_func_CN import computeH_dae
from Wave_fenics.ifac_wc.sim_CN.waveode_wc_func_CN import computeH_ode

for ind in range(4, 5):
    print("DAE: " +str(ind))
    computeH_dae(ind)
    print("DAE " +str(ind) + " completed" )

    print("ODE: " +str(ind))   
    computeH_ode(ind)
    print("ODE " +str(ind) + " completed" )

     
#print("DAE: " +str(15))
#computeH_dae(15)
#print("DAE " +str(15) + " completed" )
# computeH_ode(15)