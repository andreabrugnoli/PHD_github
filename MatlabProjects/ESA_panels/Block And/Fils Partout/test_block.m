% Test function
clear all
close all
clc

M_SA = 1147;
Jzz_SA = 212;
PO = [0 0 0.5]';
[A,B,C,D] = linmod('Sentinel_model');

format long
Res_M = D
size(A)

% Test function

% l1=1; %m
% l2=1; %m
% t=0.005; %m
% rho = 2700; % kg/m^3
% E = 70*10^9; % Pa
% ni = 0.35; 
% nx = 1;
% ny = 1;
% P_id = 1;
% C_ids = [2, 4];
% xi = 0;
% 
% [MtzRyRx] = NPort_KirchhoffTzRyRx(l1, l2, t, rho, E, ni, nx, ny, P_id, C_ids, xi)
