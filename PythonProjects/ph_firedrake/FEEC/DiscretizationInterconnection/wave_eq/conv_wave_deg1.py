DEG=1

geo_case = "2D"
bc_input = "DN"

if geo_case=="2D":
    from FEEC.DiscretizationInterconnection.wave_eq.gyrator_wave2D import compute_err
else:
    from FEEC.DiscretizationInterconnection.wave_eq.gyrator_wave3D import compute_err
import numpy as np
import os
save_res = True # input("Save results: ")

# path_project = "~/GitProjects/PHD_github/PythonProjects/ph_firedrake/FEEC/DiscretizationInterconnection/wave_eq/"
path_res = "results_wave/"
# path_res = os.path.join(path_project, folder_res)
# os.mkdir(path_res)

n_test_deg1 = 5


n_vec_deg1 = np.array([2 ** (i+2) for i in range(n_test_deg1)])
h_vec_deg1 = 1./n_vec_deg1

p3_err_deg1 = np.zeros((n_test_deg1, ))
u1_err_deg1 = np.zeros((n_test_deg1, 2))
p0_err_deg1 = np.zeros((n_test_deg1, 2))
u2_err_deg1 = np.zeros((n_test_deg1, 2))

p30_err_deg1 = np.zeros((n_test_deg1,))
u12_err_deg1 = np.zeros((n_test_deg1,))

H_err_deg1 = np.zeros((n_test_deg1, 3))

order_p3_deg1 = np.zeros((n_test_deg1 - 1,))
order_u1_deg1 = np.zeros((n_test_deg1 - 1, 2))
order_p0_deg1 = np.zeros((n_test_deg1 - 1, 2))
order_u2_deg1 = np.zeros((n_test_deg1 - 1, 2))

order_p30_deg1 = np.zeros((n_test_deg1 - 1,))
order_u12_deg1 = np.zeros((n_test_deg1 - 1,))

order_H_deg1 = np.zeros((n_test_deg1 - 1, 3))

for i in range(n_test_deg1):
    res_deg1 = compute_err(n_vec_deg1[i], 100, deg=DEG, bd_cond=bc_input)

    # vp_err_deg1[i] = res_deg1["p_err"]
    # sigp_err_deg1[i] = res_deg1["q_err"]
    # vd_err_deg1[i] = res_deg1["pd_err"]
    # sigd_err_deg1[i] = res_deg1["qd_err"]

    p3_err_deg1[i] = res_deg1["err_p3"]
    u1_err_deg1[i, :] = res_deg1["err_u1"]
    p0_err_deg1[i, :] = res_deg1["err_p0"]
    u2_err_deg1[i, :] = res_deg1["err_u2"]

    p30_err_deg1[i] = res_deg1["err_p30"]
    u12_err_deg1[i] = res_deg1["err_u12"]

    H_err_deg1[i] = res_deg1["err_H"]

    if i>0:
        order_p3_deg1[i - 1] = np.log(p3_err_deg1[i] / p3_err_deg1[i - 1]) / np.log(h_vec_deg1[i] / h_vec_deg1[i - 1])

        order_u1_deg1[i - 1, 0] = np.log(u1_err_deg1[i, 0] / u1_err_deg1[i - 1, 0]) / np.log(h_vec_deg1[i] / h_vec_deg1[i - 1])
        order_u1_deg1[i - 1, 1] = np.log(u1_err_deg1[i, 1] / u1_err_deg1[i - 1, 1]) / np.log(h_vec_deg1[i] / h_vec_deg1[i - 1])

        order_p0_deg1[i - 1, 0] = np.log(p0_err_deg1[i, 0] / p0_err_deg1[i - 1, 0]) / np.log(h_vec_deg1[i] / h_vec_deg1[i - 1])
        order_p0_deg1[i - 1, 1] = np.log(p0_err_deg1[i, 1] / p0_err_deg1[i - 1, 1]) / np.log(h_vec_deg1[i] / h_vec_deg1[i - 1])

        order_u2_deg1[i - 1, 0] = np.log(u2_err_deg1[i, 0] / u2_err_deg1[i - 1, 0]) / np.log(h_vec_deg1[i] / h_vec_deg1[i - 1])
        order_u2_deg1[i - 1, 1] = np.log(u2_err_deg1[i, 1] / u2_err_deg1[i - 1, 1]) / np.log(h_vec_deg1[i] / h_vec_deg1[i - 1])

        order_p30_deg1[i - 1] = np.log(p30_err_deg1[i] / p30_err_deg1[i - 1]) / np.log(h_vec_deg1[i] / h_vec_deg1[i - 1])
        order_u12_deg1[i - 1] = np.log(u12_err_deg1[i] / u12_err_deg1[i - 1]) / np.log(h_vec_deg1[i] / h_vec_deg1[i - 1])

        order_H_deg1[i - 1, 0] = np.log(H_err_deg1[i, 0] / H_err_deg1[i - 1, 0]) / np.log(h_vec_deg1[i] / h_vec_deg1[i - 1])
        order_H_deg1[i - 1, 1] = np.log(H_err_deg1[i, 1] / H_err_deg1[i - 1, 1]) / np.log(h_vec_deg1[i] / h_vec_deg1[i - 1])
        order_H_deg1[i - 1, 2] = np.log(H_err_deg1[i, 2] / H_err_deg1[i - 1, 2]) / np.log(h_vec_deg1[i] / h_vec_deg1[i - 1])


print("Estimated L2 order of convergence for p_3: " + str(order_p3_deg1))
print("Estimated L2, Hcurl order of convergence for u_1: " + str(order_u1_deg1))

print("Estimated L2, H1 order of convergence for p_0: " + str(order_p0_deg1))
print("Estimated L2, Hdiv order of convergence for u_2: " + str(order_u2_deg1))

print("Estimated L2 order of convergence for p_0 - p_3: " + str(order_p30_deg1))
print("Estimated L2 order of convergence for u_2 - u_1: " + str(order_u12_deg1))

print("Estimated order of convergence for H_s: " + str(order_H_deg1[:, 0]))
print("Estimated L2 order of convergence for H_10: " + str(order_H_deg1[:, 1]))
print("Estimated L2 order of convergence for H_32: " + str(order_H_deg1[:, 2]))


if save_res:
    np.save(path_res + "h_deg1_" + bc_input + "_" + geo_case, h_vec_deg1)

    np.save(path_res + "p3_err_deg1_" + bc_input + "_" + geo_case, p3_err_deg1)
    np.save(path_res + "u1_err_deg1_" + bc_input + "_" + geo_case, u1_err_deg1)
    np.save(path_res + "p0_err_deg1_" + bc_input + "_" + geo_case, p0_err_deg1)
    np.save(path_res + "u2_err_deg1_" + bc_input + "_" + geo_case, u2_err_deg1)
    np.save(path_res + "p30_err_deg1_" + bc_input + "_" + geo_case, p30_err_deg1)
    np.save(path_res + "u12_err_deg1_" + bc_input + "_" + geo_case, u12_err_deg1)

    np.save(path_res + "H_err_deg1_" + bc_input + "_" + geo_case, H_err_deg1)

    np.save(path_res + "order_p3_deg1_" + bc_input + "_" + geo_case, order_p3_deg1)
    np.save(path_res + "order_u1_deg1_" + bc_input + "_" + geo_case, order_u1_deg1)
    np.save(path_res + "order_p0_deg1_" + bc_input + "_" + geo_case, order_p0_deg1)
    np.save(path_res + "order_u2_deg1_" + bc_input + "_" + geo_case, order_u2_deg1)

    np.save(path_res + "order_p30_deg1_" + bc_input + "_" + geo_case, order_p30_deg1)
    np.save(path_res + "order_u12_deg1_" + bc_input + "_" + geo_case, order_u12_deg1)

    np.save(path_res + "order_H_deg1_" + bc_input + "_" + geo_case, order_H_deg1)
