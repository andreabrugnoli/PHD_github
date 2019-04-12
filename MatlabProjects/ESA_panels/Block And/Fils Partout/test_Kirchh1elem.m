% Test Kirchhoff one element 
% Equivalent Matlab script to the Simulink model
clear all
close all
clc

lx=1; %m
ly=1; %m
t=0.003; %m
rho = 2015; % kg/m^3
E = 70*10^9; % Pa
ni = 0.35; 
nx = 1;
ny = 1;
xi = 0.003;

Cs_xy = [lx, 1/4*ly;
         lx, 3/4*ly;];

[MtzRyRx_And] = NPort_Kirchh_1elem_TzRyRx_Cs(lx, ly, t, rho, E, ni, 1, Cs_xy, xi);

[MtzRyRx_Fra] = NPort_KirchhoffTzRyRx(lx, ly, t, rho, E, ni, 1,1, 1, [2,4], xi)