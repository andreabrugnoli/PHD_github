DEG=1

geo_case = "3D"

from FEEC.maxwell_eq.dual_field_maxwell3D import compute_err


import numpy as np

path_res = "results_maxwell/"
bc_input = "EH"

save_res = input("Save results: ")
n_test_deg1 = int(input("Number tests: "))

n_vec_deg1 = np.array([2 ** (i) for i in range(n_test_deg1)])
h_vec_deg1 = 1. / n_vec_deg1

E2_err_deg1 = np.zeros((n_test_deg1, 2))
H2_err_deg1 = np.zeros((n_test_deg1, 2))
E1_err_deg1 = np.zeros((n_test_deg1, 2))
H1_err_deg1 = np.zeros((n_test_deg1, 2))

E21_err_deg1 = np.zeros((n_test_deg1,))
H21_err_deg1 = np.zeros((n_test_deg1,))

H_err_deg1 = np.zeros((n_test_deg1, 3))

order_E2_deg1 = np.zeros((n_test_deg1 - 1, 2))
order_H2_deg1 = np.zeros((n_test_deg1 - 1, 2))
order_E1_deg1 = np.zeros((n_test_deg1 - 1, 2))
order_H1_deg1 = np.zeros((n_test_deg1 - 1, 2))

order_E21_deg1 = np.zeros((n_test_deg1 - 1,))
order_H21_deg1 = np.zeros((n_test_deg1 - 1,))

order_H_deg1 = np.zeros((n_test_deg1 - 1, 3))

for i in range(n_test_deg1):
    res_deg1 = compute_err(n_vec_deg1[i], 100, deg=DEG, bd_cond=bc_input)

    E2_err_deg1[i, :] = res_deg1["err_E2"]
    E1_err_deg1[i, :] = res_deg1["err_E1"]
    H2_err_deg1[i, :] = res_deg1["err_H2"]
    H1_err_deg1[i, :] = res_deg1["err_H1"]

    E21_err_deg1[i] = res_deg1["err_E21"]
    H21_err_deg1[i] = res_deg1["err_H21"]

    H_err_deg1[i] = res_deg1["err_H"]

    if i > 0:
        order_E2_deg1[i - 1, 0] = np.log(E2_err_deg1[i, 0] / E2_err_deg1[i - 1, 0]) / np.log(
            h_vec_deg1[i] / h_vec_deg1[i - 1])
        order_E2_deg1[i - 1, 1] = np.log(E2_err_deg1[i, 1] / E2_err_deg1[i - 1, 1]) / np.log(
            h_vec_deg1[i] / h_vec_deg1[i - 1])

        order_E1_deg1[i - 1, 0] = np.log(E1_err_deg1[i, 0] / E1_err_deg1[i - 1, 0]) / np.log(
            h_vec_deg1[i] / h_vec_deg1[i - 1])
        order_E1_deg1[i - 1, 1] = np.log(E1_err_deg1[i, 1] / E1_err_deg1[i - 1, 1]) / np.log(
            h_vec_deg1[i] / h_vec_deg1[i - 1])

        order_H2_deg1[i - 1, 0] = np.log(H2_err_deg1[i, 0] / H2_err_deg1[i - 1, 0]) / np.log(
            h_vec_deg1[i] / h_vec_deg1[i - 1])
        order_H2_deg1[i - 1, 1] = np.log(H2_err_deg1[i, 1] / H2_err_deg1[i - 1, 1]) / np.log(
            h_vec_deg1[i] / h_vec_deg1[i - 1])

        order_H1_deg1[i - 1, 0] = np.log(H1_err_deg1[i, 0] / H1_err_deg1[i - 1, 0]) / np.log(
            h_vec_deg1[i] / h_vec_deg1[i - 1])
        order_H1_deg1[i - 1, 1] = np.log(H1_err_deg1[i, 1] / H1_err_deg1[i - 1, 1]) / np.log(
            h_vec_deg1[i] / h_vec_deg1[i - 1])

        order_E21_deg1[i - 1] = np.log(E21_err_deg1[i] / E21_err_deg1[i - 1]) / np.log(
            h_vec_deg1[i] / h_vec_deg1[i - 1])

        order_H21_deg1[i - 1] = np.log(H21_err_deg1[i] / H21_err_deg1[i - 1]) / np.log(
            h_vec_deg1[i] / h_vec_deg1[i - 1])

        order_H_deg1[i - 1, 0] = np.log(H_err_deg1[i, 0] / H_err_deg1[i - 1, 0]) / np.log(
            h_vec_deg1[i] / h_vec_deg1[i - 1])
        order_H_deg1[i - 1, 1] = np.log(H_err_deg1[i, 1] / H_err_deg1[i - 1, 1]) / np.log(
            h_vec_deg1[i] / h_vec_deg1[i - 1])
        order_H_deg1[i - 1, 2] = np.log(H_err_deg1[i, 2] / H_err_deg1[i - 1, 2]) / np.log(
            h_vec_deg1[i] / h_vec_deg1[i - 1])

print("Estimated L2, Hdiv order of convergence for E2: " + str(order_E2_deg1))
print("Estimated L2, Hdiv order of convergence for H2: " + str(order_H2_deg1))

print("Estimated L2, Hcurl order of convergence for E1: " + str(order_E1_deg1))
print("Estimated L2, Hcurl order of convergence for H1: " + str(order_H1_deg1))

print("Estimated L2 order of convergence for E_2 - E_1: " + str(order_E21_deg1))
print("Estimated L2 order of convergence for H_2 - H_1: " + str(order_H21_deg1))

print("Estimated order of convergence for H_s: " + str(order_H_deg1[:, 0]))
print("Estimated L2 order of convergence for H_E2H1: " + str(order_H_deg1[:, 1]))
print("Estimated L2 order of convergence for H_H2E1: " + str(order_H_deg1[:, 2]))

if save_res:
    np.save(path_res + "h_deg1_" + bc_input + "_" + geo_case, h_vec_deg1)

    np.save(path_res + "E2_err_deg1_" + bc_input + "_" + geo_case, E2_err_deg1)
    np.save(path_res + "E1_err_deg1_" + bc_input + "_" + geo_case, E1_err_deg1)
    np.save(path_res + "H2_err_deg1_" + bc_input + "_" + geo_case, H2_err_deg1)
    np.save(path_res + "H1_err_deg1_" + bc_input + "_" + geo_case, H1_err_deg1)
    np.save(path_res + "E21_err_deg1_" + bc_input + "_" + geo_case, E21_err_deg1)
    np.save(path_res + "H21_err_deg1_" + bc_input + "_" + geo_case, H21_err_deg1)

    np.save(path_res + "H_err_deg1_" + bc_input + "_" + geo_case, H_err_deg1)

    np.save(path_res + "order_E2_deg1_" + bc_input + "_" + geo_case, order_E2_deg1)
    np.save(path_res + "order_E1_deg1_" + bc_input + "_" + geo_case, order_E1_deg1)
    np.save(path_res + "order_H2_deg1_" + bc_input + "_" + geo_case, order_H2_deg1)
    np.save(path_res + "order_H1_deg1_" + bc_input + "_" + geo_case, order_H1_deg1)

    np.save(path_res + "order_E21_deg1_" + bc_input + "_" + geo_case, order_E21_deg1)
    np.save(path_res + "order_H21_deg1_" + bc_input + "_" + geo_case, order_H21_deg1)

    np.save(path_res + "order_H_deg1_" + bc_input + "_" + geo_case, order_H_deg1)
