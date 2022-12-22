from firedrake import *
import numpy as np

from FEEC.DiscretizationInterconnection.maxwell_eq.dynamic_hybridization.spaces_forms_hybridmaxwell \
    import spacesE1H2, spacesE2H1

n_test_deg1 = 5
n_test_deg2 = 4
n_test_deg3 = 3

n_vec_deg1 = np.array([2 ** (i) for i in range(n_test_deg1)])
n_vec_deg2 = np.array([2 ** (i) for i in range(n_test_deg2)])
n_vec_deg3 = np.array([2 ** (i) for i in range(n_test_deg3)])

n_dict = {1: n_vec_deg1, 2: n_vec_deg2, 3: n_vec_deg3}

kk=1

print('E1H2 Spaces')

for n_deg in n_dict.values():

    for n_el in n_deg:

        mesh = BoxMesh(n_el, n_el, n_el, 1, 1, 1)

        WE1H2_loc, VE1_tan, V12 = spacesE1H2(mesh, kk)

        print("Conforming Galerkin E1H2  dim for n_el " + str(n_el) + " and degree " + str(kk) + ": " + str(V12.dim()))
        print("Conforming Galerkin E1H2 (2 broken) dim for n_el " + str(n_el) + \
              " and degree " + str(kk) + ": " + str(V12.sub(0).dim() + WE1H2_loc.sub(1).dim()))

        print("Hybrid E1H2 dim for n_el " + str(n_el) + " and degree " + str(kk) + ": " + str(VE1_tan.dim()))

        print("Ratio hybrid continous " + str(n_el) + " and degree " + str(kk) + ": " + \
              str(VE1_tan.dim()/(V12.sub(0).dim() + WE1H2_loc.sub(1).dim())))
    kk = kk+1

print('E2H1 Spaces')

kk=1

for n_deg in n_dict.values():

    for n_el in n_deg:

        mesh = BoxMesh(n_el, n_el, n_el, 1, 1, 1)

        WE2H1_loc, VH1_tan, V21 = spacesE2H1(mesh, kk)

        print("Conforming Galerkin E2H1  dim for n_el " + str(n_el) + " and degree " + str(kk) + ": " + str(V21.dim()))
        print("Conforming Galerkin E2H1 (2 broken) dim for n_el " + str(n_el) + \
              " and degree " + str(kk) + ": " + str(WE2H1_loc.sub(0).dim()+V21.sub(1).dim()))
        print("Hybrid E2H1 dim for n_el " + str(n_el) + " and degree " + str(kk) + ": " + str(VH1_tan.dim()))

        print("Ratio hybrid continous " + str(n_el) + " and degree " + str(kk) + ": " + \
              str(VH1_tan.dim() / (WE2H1_loc.sub(0).dim()+V21.sub(1).dim())))

    kk = kk+1