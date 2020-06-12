clc
close all
clear all
addpath('./TITOP_Kirchh/')

rho = 7810;  % Kg/m^3
E = 10^9;  % Pa
nu = 0.3;
h = 0.01;  % m
Lx = 0.6;  % m
Ly = 0.3;   % m

g = 9.81;

m_plate = rho*h*Lx*Ly;

J_zz = m_plate*(Lx^2+Ly^2)/3

nx = 2;
ny = 2;

n_P = 1; % (nx+1)*(ny/6)+1;
n_C_gr = (nx+1)*(ny/2)+nx/2+1;
n_C_y = nx+1
plate_titop = NPort_KirchhoffTzRyRx(Lx,Ly,h,rho,E,nu,nx,ny,n_P,[n_C_gr, n_C_y],0);

plate_pendulum = invio(plate_titop,[8, 9]);

plate_pendulum = minreal(plate_pendulum);
plate_pendulum = balred(plate_pendulum, 10);



