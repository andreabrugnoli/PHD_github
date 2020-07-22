from Wave_fenics.ifac_wc.wavedae_wc_func import computeH_dae
from Wave_fenics.ifac_wc.waveode_wc_func_rk import computeH_ode

for ind in range(4, 5):
     computeH_dae(ind)
     computeH_ode(ind)

# computeH_dae(15)
# computeH_ode(15)