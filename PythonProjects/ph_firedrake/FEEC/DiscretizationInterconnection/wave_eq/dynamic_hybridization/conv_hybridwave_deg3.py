DEG = 3

geo_case = "2D"
bc_input = "DN"

if geo_case=="2D":
    from FEEC.DiscretizationInterconnection.wave_eq.dynamic_hybridization.hybrid_wave2D_grad import compute_err
else:
    from FEEC.DiscretizationInterconnection.wave_eq.dynamic_hybridization.hybrid_wave3D import compute_err
import numpy as np

save_res = True # input("Save results: ")

# path_project = "~/GitProjects/PHD_github/PythonProjects/ph_firedrake/FEEC/DiscretizationInterconnection/wave_eq/"
path_res = "results_hybrid/"
# path_res = os.path.join(path_project, folder_res)
# os.mkdir(path_res)

n_test_deg3 = 3

n_vec_deg3 = np.array([2 ** (i+2) for i in range(n_test_deg3)])
h_vec_deg3 = 1./n_vec_deg3

u1_err_deg3 = np.zeros((n_test_deg3, 2))
p0_err_deg3 = np.zeros((n_test_deg3, 2))

p0tan_err_deg3 = np.zeros((n_test_deg3, ))
lambda0nor_err_deg3 = np.zeros((n_test_deg3, ))

H_err_deg3 = np.zeros((n_test_deg3,))

order_u1_deg3 = np.zeros((n_test_deg3 - 1, 2))
order_p0_deg3 = np.zeros((n_test_deg3 - 1, 2))

order_p0tan_deg3 = np.zeros((n_test_deg3 - 1, ))
order_lambda0nor_deg3 = np.zeros((n_test_deg3 - 1, ))

order_H_deg3 = np.zeros((n_test_deg3 - 1,))

for i in range(n_test_deg3):
    res_deg3 = compute_err(n_vec_deg3[i], 100, deg=DEG, bd_cond=bc_input)

    u1_err_deg3[i, :] = res_deg3["err_u1"]
    p0_err_deg3[i, :] = res_deg3["err_p0"]

    p0tan_err_deg3[i] = res_deg3["err_p0tan"]
    lambda0nor_err_deg3[i] = res_deg3["err_lambdanor"]

    H_err_deg3[i] = res_deg3["err_H"]

    if i>0:

        order_u1_deg3[i - 1, 0] = np.log(u1_err_deg3[i, 0] / u1_err_deg3[i - 1, 0]) / np.log(h_vec_deg3[i] / h_vec_deg3[i - 1])
        order_u1_deg3[i - 1, 1] = np.log(u1_err_deg3[i, 1] / u1_err_deg3[i - 1, 1]) / np.log(h_vec_deg3[i] / h_vec_deg3[i - 1])

        order_p0_deg3[i - 1, 0] = np.log(p0_err_deg3[i, 0] / p0_err_deg3[i - 1, 0]) / np.log(h_vec_deg3[i] / h_vec_deg3[i - 1])
        order_p0_deg3[i - 1, 1] = np.log(p0_err_deg3[i, 1] / p0_err_deg3[i - 1, 1]) / np.log(h_vec_deg3[i] / h_vec_deg3[i - 1])

        order_p0tan_deg3[i - 1] = np.log(p0tan_err_deg3[i] / p0tan_err_deg3[i - 1]) / np.log(
            h_vec_deg3[i] / h_vec_deg3[i - 1])

        order_lambda0nor_deg3[i - 1] = np.log(lambda0nor_err_deg3[i] / lambda0nor_err_deg3[i - 1]) / np.log(
            h_vec_deg3[i] / h_vec_deg3[i - 1])

        order_H_deg3[i - 1] = np.log(H_err_deg3[i] / H_err_deg3[i - 1]) / np.log(h_vec_deg3[i] / h_vec_deg3[i - 1])


print("Estimated L2, Hcurl order of convergence for u_1: " + str(order_u1_deg3))
print("Estimated L2, H1 order of convergence for p_0: " + str(order_p0_deg3))

print("Estimated L2 order of convergence for lambda0nor: " + str(order_lambda0nor_deg3))
print("Estimated L2 order of convergence for p_0tan: " + str(order_p0tan_deg3))

print("Estimated order of convergence for H_10: " + str(order_H_deg3))


if save_res:
    np.save(path_res + "h_deg3_" + bc_input + "_" + geo_case, h_vec_deg3)

    np.save(path_res + "u1_err_deg3_" + bc_input + "_" + geo_case, u1_err_deg3)
    np.save(path_res + "p0_err_deg3_" + bc_input + "_" + geo_case, p0_err_deg3)
    np.save(path_res + "p0tan_err_deg3_" + bc_input + "_" + geo_case, p0tan_err_deg3)
    np.save(path_res + "lambdanor_err_deg3_" + bc_input + "_" + geo_case, lambda0nor_err_deg3)

    np.save(path_res + "H_err_deg3_" + bc_input + "_" + geo_case, H_err_deg3)

    np.save(path_res + "order_u1_deg3_" + bc_input + "_" + geo_case, order_u1_deg3)
    np.save(path_res + "order_p0_deg3_" + bc_input + "_" + geo_case, order_p0_deg3)
    np.save(path_res + "order_p0tan_deg3_" + bc_input + "_" + geo_case, order_p0tan_deg3)
    np.save(path_res + "order_lambda0nor_deg3_" + bc_input + "_" + geo_case, order_lambda0nor_deg3)

    np.save(path_res + "order_H_deg3_" + bc_input + "_" + geo_case, order_H_deg3)
