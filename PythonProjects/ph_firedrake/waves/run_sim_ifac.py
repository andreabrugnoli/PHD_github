from waves.wavedae_wc_func import computeH_dae
from waves.waveode_wc_func_rk import computeH_ode

for ind in range(4, 11):
    computeH_dae(ind)
    computeH_ode(ind)

computeH_dae(15)
#computeH_ode(15)
