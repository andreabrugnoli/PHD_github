from firedrake import *

from spaces_forms_hybridwave import spaces01, spaces32

n_test_deg1 = 5
n_test_deg2 = 4
n_test_deg3 = 3

n_vec_deg1 = np.array([2 ** (i) for i in range(n_test_deg1)])
n_vec_deg2 = np.array([2 ** (i) for i in range(n_test_deg2)])
n_vec_deg3 = np.array([2 ** (i) for i in range(n_test_deg3)])

n_dict = {1: n_vec_deg1, 2: n_vec_deg2, 3: n_vec_deg3}

kk=1
for n_deg in n_dict.values():

    for n_el in n_deg:

        mesh = BoxMesh(n_el, n_el, n_el, 1, 1, 1)

        W01_loc, V0_tan, V01 = spaces01(mesh, kk)
        V_grad = W01_loc * V0_tan

        # print("Conforming Galerkin 01 dim for n_el " + str(n_el) + " and degree " + str(kk) + ": " + str(V01.dim()))
        print("Conforming Galerkin 01 (1 broken) dim for n_el " + str(n_el) + " and degree " + str(kk) + ": "\
              + str(V01.sub(0).dim() + W01_loc.sub(1).dim()))
        print("Hybrid 01 dim for n_el " + str(n_el) + " and degree " + str(kk) + ": " + str(V0_tan.dim()))

        W32_loc, V2_tan, V32 = spaces32(mesh, kk)

        print("Conforming Galerkin 32 dim for n_el " + str(n_el) + " and degree " + str(kk) + ": " + str(V32.dim()))
        print("Hybrid 32 dim for n_el " + str(n_el) + " and degree " + str(kk) + ": " + str(V2_tan.dim()))

    kk = kk+1