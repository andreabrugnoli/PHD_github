% Equivalent Matlab script to the Simulink model
clear all
close all
clc

l1=3; %m
l2=3; %m
t=0.003; %m
rho = 2015; % kg/m^3
E = 70*10^9; % Pa
ni = 0.35; 
nx = 1;
ny = 1;

M_SA = 1147;
Jzz_SA = 212;

xi = 0.003;

P_id_1 = 1;%22;%
P_id_2 = 1;%8;%
P_id_3 = 1;%8;%
C_ids_1 = [2 4];%[14, 42];%


ind_1 = 0;
ind_2 = [1:3];
ind_3 = [1:3];

MtzRyRx_Fra = NPort_KirchhoffTzRyRx(l1, l2, t, rho, E, ni, nx, ny, P_id_1, C_ids_1, xi);

Cs_xy = [l1, 1/3*l2;
         l1, 2/3*l2];


MtzRyRx_And = NPort_Kirchh_1elem_TzRyRx_Cs(l1, l2, t, rho, E, ni, Cs_xy, xi);

figure(1)
bode(MtzRyRx_Fra(7:9,4:6),'r')
figure(2)
bode(MtzRyRx_And(7:9,4:6),'b')

