geo_case = "2D"
# geo_case = "3D"

if geo_case=="2D":
    from FEEC.wave_eq.compute_err_wave2D import compute_err
else:
    from FEEC.wave_eq.compute_err_wave3D import compute_err
import numpy as np

save_res = True
bc_input = "D"
n_test = 2

n_vec = np.array([2 ** (i+3) for i in range(n_test)])
h_vec = 1./n_vec

vp_err_deg1 = np.zeros((n_test,))
vp_err_deg2 = np.zeros((n_test,))
vp_err_deg3 = np.zeros((n_test,))

sigp_err_deg1 = np.zeros((n_test,))
sigp_err_deg2 = np.zeros((n_test,))
sigp_err_deg3 = np.zeros((n_test,))

vd_err_deg1 = np.zeros((n_test,))
vd_err_deg2 = np.zeros((n_test,))
vd_err_deg3 = np.zeros((n_test,))

sigd_err_deg1 = np.zeros((n_test,))
sigd_err_deg2 = np.zeros((n_test,))
sigd_err_deg3 = np.zeros((n_test,))

order_vp_deg1 = np.zeros((n_test-1,))
order_vp_deg2 = np.zeros((n_test-1,))
order_vp_deg3 = np.zeros((n_test-1,))

order_sigp_deg1 = np.zeros((n_test-1,))
order_sigp_deg2 = np.zeros((n_test-1,))
order_sigp_deg3 = np.zeros((n_test-1,))

order_vd_deg1 = np.zeros((n_test-1,))
order_vd_deg2 = np.zeros((n_test-1,))
order_vd_deg3 = np.zeros((n_test-1,))

order_sigd_deg1 = np.zeros((n_test-1,))
order_sigd_deg2 = np.zeros((n_test-1,))
order_sigd_deg3 = np.zeros((n_test-1,))

for i in range(n_test):
    res_deg1 = compute_err(n_vec[i], 100, deg=1, bd_cond=bc_input)

    vp_err_deg1[i] = res_deg1["p_err"]
    sigp_err_deg1[i] = res_deg1["q_err"]
    vd_err_deg1[i] = res_deg1["pd_err"]
    sigd_err_deg1[i] = res_deg1["qd_err"]

    if i>0:
        order_vp_deg1[i-1] = np.log(vp_err_deg1[i]/vp_err_deg1[i-1])/np.log(h_vec[i]/h_vec[i-1])
        order_sigp_deg1[i-1] = np.log(sigp_err_deg1[i]/sigp_err_deg1[i-1])/np.log(h_vec[i]/h_vec[i-1])

        order_vd_deg1[i-1] = np.log(vd_err_deg1[i]/vd_err_deg1[i-1])/np.log(h_vec[i]/h_vec[i-1])
        order_sigd_deg1[i-1] = np.log(sigd_err_deg1[i]/sigd_err_deg1[i-1])/np.log(h_vec[i]/h_vec[i-1])

print("Estimated order of convergence for v primal: " + str(order_vp_deg1))
print("Estimated order of convergence for sigma primal: " + str(order_sigp_deg1))

print("Estimated order of convergence for v dual: " + str(order_vd_deg1))
print("Estimated order of convergence for sigma dual: " + str(order_sigd_deg1))

path_res = "results_wave/"
if save_res:
    np.save(path_res + "h_" + bc_input + "_" + geo_case, h_vec)

    np.save(path_res + "vp_err_deg1_" + bc_input + "_" + geo_case, vp_err_deg1)
    np.save(path_res + "sigp_err_deg1_" + bc_input + "_" + geo_case, sigp_err_deg1)

    np.save(path_res + "vd_err_deg1_" + bc_input + "_" + geo_case, vd_err_deg1)
    np.save(path_res + "sigd_err_deg1_" + bc_input + "_" + geo_case, sigd_err_deg1)

    np.save(path_res + "order_vp_deg1_" + bc_input + "_" + geo_case, order_vp_deg1)
    np.save(path_res + "order_sigp_deg1_" + bc_input + "_" + geo_case, order_sigp_deg1)

    np.save(path_res + "order_vd_deg1_" + bc_input + "_" + geo_case, order_vd_deg1)
    np.save(path_res + "order_sigd_deg1_" + bc_input + "_" + geo_case, order_sigd_deg1)
