DEG=2

geo_case = "2D"
# geo_case = "3D"

if geo_case=="2D":
    from FEEC.wave_eq.compute_err_wave2D import compute_err
    # from FEEC.wave_eq.staggering_wave2D import compute_err
else:
    # from FEEC.wave_eq.compute_err_wave3D import compute_err
    from FEEC.wave_eq.staggering_wave3D import compute_err

import numpy as np

save_res = False
bc_input = "D"
n_test_deg2 = 4

n_vec_deg2 = np.array([2 ** (i+1) for i in range(n_test_deg2)])
h_vec_deg2 = 1./n_vec_deg2

p3_err_deg2 = np.zeros((n_test_deg2,))
u1_err_deg2 = np.zeros((n_test_deg2,))
p0_err_deg2 = np.zeros((n_test_deg2,))
u2_err_deg2 = np.zeros((n_test_deg2,))

p30_err_deg2 = np.zeros((n_test_deg2,))
u12_err_deg2 = np.zeros((n_test_deg2,))

order_p3_deg2 = np.zeros((n_test_deg2 - 1,))
order_u1_deg2 = np.zeros((n_test_deg2 - 1,))
order_p0_deg2 = np.zeros((n_test_deg2 - 1,))
order_u2_deg2 = np.zeros((n_test_deg2 - 1,))

order_p30_deg2 = np.zeros((n_test_deg2 - 1,))
order_u12_deg2 = np.zeros((n_test_deg2 - 1,))

for i in range(n_test_deg2):
    res_deg2 = compute_err(n_vec_deg2[i], 200, deg=DEG, bd_cond=bc_input)

    p3_err_deg2[i] = res_deg2["err_p3"]
    u1_err_deg2[i] = res_deg2["err_u1"]
    p0_err_deg2[i] = res_deg2["err_p0"]
    u2_err_deg2[i] = res_deg2["err_u2"]

    p30_err_deg2[i] = res_deg2["err_p30"]
    u12_err_deg2[i] = res_deg2["err_u12"]

    if i>0:
        order_p3_deg2[i - 1] = np.log(p3_err_deg2[i] / p3_err_deg2[i - 1]) / np.log(h_vec_deg2[i] / h_vec_deg2[i - 1])
        order_u1_deg2[i - 1] = np.log(u1_err_deg2[i] / u1_err_deg2[i - 1]) / np.log(h_vec_deg2[i] / h_vec_deg2[i - 1])

        order_p0_deg2[i - 1] = np.log(p0_err_deg2[i] / p0_err_deg2[i - 1]) / np.log(h_vec_deg2[i] / h_vec_deg2[i - 1])
        order_u2_deg2[i - 1] = np.log(u2_err_deg2[i] / u2_err_deg2[i - 1]) / np.log(h_vec_deg2[i] / h_vec_deg2[i - 1])

        order_p30_deg2[i - 1] = np.log(p30_err_deg2[i] / p30_err_deg2[i - 1]) / np.log(h_vec_deg2[i] / h_vec_deg2[i - 1])
        order_u12_deg2[i - 1] = np.log(u12_err_deg2[i] / u12_err_deg2[i - 1]) / np.log(h_vec_deg2[i] / h_vec_deg2[i - 1])

print("Estimated order of convergence for p_3: " + str(order_p3_deg2))
print("Estimated order of convergence for u_1: " + str(order_u1_deg2))

print("Estimated order of convergence for p_0: " + str(order_p0_deg2))
print("Estimated order of convergence for u_2: " + str(order_u2_deg2))

print("Estimated order of convergence for p_0 - p_3: " + str(order_p30_deg2))
print("Estimated order of convergence for u_2 - u_1: " + str(order_u12_deg2))

path_res = "results_wave/"
if save_res:
    np.save(path_res + "h_deg2_" + bc_input + "_" + geo_case, h_vec_deg2)

    np.save(path_res + "p3_err_deg2_" + bc_input + "_" + geo_case, p3_err_deg2)
    np.save(path_res + "u1_err_deg2_" + bc_input + "_" + geo_case, u1_err_deg2)

    np.save(path_res + "p0_err_deg2_" + bc_input + "_" + geo_case, p0_err_deg2)
    np.save(path_res + "u2_err_deg2_" + bc_input + "_" + geo_case, u2_err_deg2)

    np.save(path_res + "p30_err_deg2_" + bc_input + "_" + geo_case, p30_err_deg2)
    np.save(path_res + "u12_err_deg2_" + bc_input + "_" + geo_case, u12_err_deg2)

    np.save(path_res + "order_p3_deg2_" + bc_input + "_" + geo_case, order_p3_deg2)
    np.save(path_res + "order_u1_deg2_" + bc_input + "_" + geo_case, order_u1_deg2)

    np.save(path_res + "order_p0_deg2_" + bc_input + "_" + geo_case, order_p0_deg2)
    np.save(path_res + "order_u2_deg2_" + bc_input + "_" + geo_case, order_u2_deg2)

    np.save(path_res + "order_p30_deg2_" + bc_input + "_" + geo_case, order_p30_deg2)
    np.save(path_res + "order_u12_deg2_" + bc_input + "_" + geo_case, order_u12_deg2)
