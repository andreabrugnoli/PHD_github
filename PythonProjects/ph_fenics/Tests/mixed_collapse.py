# -*- coding: utf-8 -*-

CollToMix_index = []
for i in range(0,N_T):
    for j in range(0,N_T):
        if np.all( np.abs(coord_T[i,:] - coord_T_collapse[j,:]) <= 1e-16 ) :
            CollToMix_index.append(j)
           
MixToColl_index = []
for i in range(0,N_T):
    for j in range(0,N_T):
        if np.all( np.abs(coord_T[j,:] - coord_T_collapse[i,:]) <= 1e-16 ) :
            MixToColl_index.append(j)