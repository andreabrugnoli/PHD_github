DEG=3

# geo_case = "2D"
geo_case = "3D"

if geo_case=="2D":
    # from FEEC.wave_eq.compute_err_wave2D import compute_err
    # from FEEC.wave_eq.compute_err_wave2D_divgrad import compute_err
    from FEEC.wave_eq.staggering_wave2D import compute_err
else:
    # from FEEC.wave_eq.compute_err_wave3D import compute_err
    from FEEC.wave_eq.staggering_wave3D import compute_err

import numpy as np
save_res = False
path_res = "results_wave/"
bc_input = "DN"
n_test_deg3 = 2

n_vec_deg3 = np.array([2 ** (i) for i in range(n_test_deg3)])
h_vec_deg3 = 1./n_vec_deg3

p3_err_deg3 = np.zeros((n_test_deg3,))
u1_err_deg3 = np.zeros((n_test_deg3, 2))
p0_err_deg3 = np.zeros((n_test_deg3, 2))
u2_err_deg3 = np.zeros((n_test_deg3, 2))

p30_err_deg3 = np.zeros((n_test_deg3,))
u12_err_deg3 = np.zeros((n_test_deg3,))

H_err_deg3 = np.zeros((n_test_deg3, 3))

order_p3_deg3 = np.zeros((n_test_deg3 - 1,))
order_u1_deg3 = np.zeros((n_test_deg3 - 1, 2))
order_p0_deg3 = np.zeros((n_test_deg3 - 1, 2))
order_u2_deg3 = np.zeros((n_test_deg3 - 1, 2))

order_p30_deg3 = np.zeros((n_test_deg3 - 1,))
order_u12_deg3 = np.zeros((n_test_deg3 - 1,))

order_H_deg3 = np.zeros((n_test_deg3 - 1, 3))

for i in range(n_test_deg3):
    res_deg3 = compute_err(n_vec_deg3[i], 100, deg=DEG, bd_cond=bc_input)

    # vp_err_deg3[i] = res_deg3["p_err"]
    # sigp_err_deg3[i] = res_deg3["q_err"]
    # vd_err_deg3[i] = res_deg3["pd_err"]
    # sigd_err_deg3[i] = res_deg3["qd_err"]

    p3_err_deg3[i] = res_deg3["err_p3"]
    u1_err_deg3[i, :] = res_deg3["err_u1"]
    p0_err_deg3[i, :] = res_deg3["err_p0"]
    u2_err_deg3[i, :] = res_deg3["err_u2"]

    p30_err_deg3[i] = res_deg3["err_p30"]
    u12_err_deg3[i] = res_deg3["err_u12"]

    H_err_deg3[i] = res_deg3["err_H"]

    if i>0:
        order_p3_deg3[i - 1] = np.log(p3_err_deg3[i] / p3_err_deg3[i - 1]) / np.log(h_vec_deg3[i] / h_vec_deg3[i - 1])

        order_u1_deg3[i - 1, 0] = np.log(u1_err_deg3[i, 0] / u1_err_deg3[i - 1, 0]) / np.log(h_vec_deg3[i] / h_vec_deg3[i - 1])
        order_u1_deg3[i - 1, 1] = np.log(u1_err_deg3[i, 1] / u1_err_deg3[i - 1, 1]) / np.log(h_vec_deg3[i] / h_vec_deg3[i - 1])

        order_p0_deg3[i - 1, 0] = np.log(p0_err_deg3[i, 0] / p0_err_deg3[i - 1, 0]) / np.log(h_vec_deg3[i] / h_vec_deg3[i - 1])
        order_p0_deg3[i - 1, 1] = np.log(p0_err_deg3[i, 1] / p0_err_deg3[i - 1, 1]) / np.log(h_vec_deg3[i] / h_vec_deg3[i - 1])

        order_u2_deg3[i - 1, 0] = np.log(u2_err_deg3[i, 0] / u2_err_deg3[i - 1, 0]) / np.log(h_vec_deg3[i] / h_vec_deg3[i - 1])
        order_u2_deg3[i - 1, 1] = np.log(u2_err_deg3[i, 1] / u2_err_deg3[i - 1, 1]) / np.log(h_vec_deg3[i] / h_vec_deg3[i - 1])

        order_p30_deg3[i - 1] = np.log(p30_err_deg3[i] / p30_err_deg3[i - 1]) / np.log(h_vec_deg3[i] / h_vec_deg3[i - 1])
        order_u12_deg3[i - 1] = np.log(u12_err_deg3[i] / u12_err_deg3[i - 1]) / np.log(h_vec_deg3[i] / h_vec_deg3[i - 1])

        order_H_deg3[i - 1, 0] = np.log(H_err_deg3[i, 0] / H_err_deg3[i - 1, 0]) / np.log(h_vec_deg3[i] / h_vec_deg3[i - 1])
        order_H_deg3[i - 1, 1] = np.log(H_err_deg3[i, 1] / H_err_deg3[i - 1, 1]) / np.log(h_vec_deg3[i] / h_vec_deg3[i - 1])
        order_H_deg3[i - 1, 2] = np.log(H_err_deg3[i, 2] / H_err_deg3[i - 1, 2]) / np.log(h_vec_deg3[i] / h_vec_deg3[i - 1])

print("Estimated L2 order of convergence for p_3: " + str(order_p3_deg3))
print("Estimated L2, Hcurl order of convergence for u_1: " + str(order_u1_deg3))

print("Estimated L2, H1 order of convergence for p_0: " + str(order_p0_deg3))
print("Estimated L2, Hdiv order of convergence for u_2: " + str(order_u2_deg3))

print("Estimated L2 order of convergence for p_0 - p_3: " + str(order_p30_deg3))
print("Estimated L2 order of convergence for u_2 - u_1: " + str(order_u12_deg3))

print("Estimated order of convergence for H_s: " + str(order_H_deg3[:, 0]))
print("Estimated L2 order of convergence for H_10: " + str(order_H_deg3[:, 1]))
print("Estimated L2 order of convergence for H_32: " + str(order_H_deg3[:, 2]))

if save_res:
    np.save(path_res + "h_deg3_" + bc_input + "_" + geo_case, h_vec_deg3)

    np.save(path_res + "p3_err_deg3_" + bc_input + "_" + geo_case, p3_err_deg3)
    np.save(path_res + "u1_err_deg3_" + bc_input + "_" + geo_case, u1_err_deg3)
    np.save(path_res + "p0_err_deg3_" + bc_input + "_" + geo_case, p0_err_deg3)
    np.save(path_res + "u2_err_deg3_" + bc_input + "_" + geo_case, u2_err_deg3)
    np.save(path_res + "p30_err_deg3_" + bc_input + "_" + geo_case, p30_err_deg3)
    np.save(path_res + "u12_err_deg3_" + bc_input + "_" + geo_case, u12_err_deg3)

    np.save(path_res + "H_err_deg3_" + bc_input + "_" + geo_case, H_err_deg3)

    np.save(path_res + "order_p3_deg3_" + bc_input + "_" + geo_case, order_p3_deg3)
    np.save(path_res + "order_u1_deg3_" + bc_input + "_" + geo_case, order_u1_deg3)
    np.save(path_res + "order_p0_deg3_" + bc_input + "_" + geo_case, order_p0_deg3)
    np.save(path_res + "order_u2_deg3_" + bc_input + "_" + geo_case, order_u2_deg3)
    np.save(path_res + "order_p30_deg3_" + bc_input + "_" + geo_case, order_p30_deg3)
    np.save(path_res + "order_u12_deg3_" + bc_input + "_" + geo_case, order_u12_deg3)

    np.save(path_res + "order_H_deg3_" + bc_input + "_" + geo_case, order_H_deg3)
