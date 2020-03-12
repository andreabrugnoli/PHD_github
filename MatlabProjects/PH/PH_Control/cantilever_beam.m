clc
close all
clear all
addpath('/home/a.brugnoli/GitProjects/MatlabProjects/PH/Functions/')
path_mat = './Matrices_EB/'
addpath(path_mat)
renameFiles(path_mat)


load A; load B; load C; load D;

% [Abar,Bbar,Cbar,T,k] = ctrbf(A,B,C);

sys = ss(A, B, C, D);
A = sys.A;
sysr = minreal(sys);
Ar= sysr.A;

% eigs = eig(Ar);
% omega = imag(eigs)/(2*pi)
% omega_p = omega(omega>=0);
% 
% format longG
% disp(sort(omega_p))