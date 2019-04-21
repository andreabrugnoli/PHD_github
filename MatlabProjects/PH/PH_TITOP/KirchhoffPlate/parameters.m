% Constant for Kirchhoff plate

% rho = 2015;  % Kg/m^3
% E = 69.8692 * 10^9;  % Pa
% G = 22.1615 * 10^9;  % Pa
% nu = E/2/G-1;
% Lx = 1;  % m
% Ly = 1;  % m
% h = 0.003; % m

rho = 1;  % Kg/m^3
E = 1;  % Pa
G = 1;  % Pa
nu = 0.3;
Lx = 1;  % m
Ly = 1;  % m
h = 0.1; % m

x_P = 0;
y_P = 0;

nx = 10;
ny = 10;

m_plate = rho*h*Lx*Ly;

Jxx = rho*h*Lx*(Ly - y_P)^3/3;
Jyy = rho*h*Ly*(Lx - x_P)^3/3;
Jzz = Jxx + Jyy;

w0 = 1e-5; wf=1e6;