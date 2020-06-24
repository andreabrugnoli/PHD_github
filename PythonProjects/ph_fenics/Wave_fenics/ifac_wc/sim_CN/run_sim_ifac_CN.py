from Wave_fenics.ifac_wc.sim_CN.wavedae_wc_func_CN import computeH_dae
from Wave_fenics.ifac_wc.sim_CN.waveode_wc_func_CN import computeH_ode

for ind in range(4, 8):
     computeH_dae(ind)
     computeH_ode(ind)
#
# computeH_dae(15)
# computeH_ode(15)