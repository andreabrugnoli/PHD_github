% Equivalent Matlab script to the Simulink model
clear all
close all
clc

t=0.003; %m
rho = 2015; % kg/m^3
E = 70*10^9; % Pa
ni = 0.35; 

lx_1=3; %m
ly_1=3; %m

lx_2=3; %m
ly_2=3; %m

lx_3=3; %m
ly_3=3; %m


M_SA = 1147;
Jzz_SA = 212;

xi = 0.003;

P_id_1 = 1;
P_id_2 = 1;
P_id_3 = 1;

Cs_xy_1 = [0,    1/2*ly_1;
           lx_1, 1/4*ly_1;
           lx_1, 3/4*ly_1];

Cs_xy_2 = [0,    2/3*ly_2;
           0,    3/4*ly_2;
           lx_2, 1/4*ly_2;
           lx_2, 3/4*ly_2];
       
Cs_xy_3 = [0,    2/3*ly_3;
           0,    3/4*ly_3;
           lx_3, 1/4*ly_3;
           lx_3, 3/4*ly_3];
       

ind_1 = [1:3];
ind_2 = [1:6];
ind_3 = [1:6];

[MtzRyRx_1] = NPort_Kirchh_1elem_TzRyRx_Cs(lx_1, ly_1, t, rho, E, ni, P_id_1, Cs_xy_1, xi)

if ind_1~=0
    MtzRyRx_1 = invio(MtzRyRx_1, ind_1);
end

[MtzRyRx_2] = NPort_Kirchh_1elem_TzRyRx_Cs(lx_2, ly_2, t, rho, E, ni, P_id_2, Cs_xy_2, xi)

if ind_2~=0
    MtzRyRx_2 = invio(MtzRyRx_2, ind_2);
end

[MtzRyRx_3] = NPort_Kirchh_1elem_TzRyRx_Cs(lx_3, ly_3, t, rho, E, ni, P_id_3, Cs_xy_3, xi)

if ind_3~=0
    MtzRyRx_3 = invio(MtzRyRx_3, ind_3);
end

