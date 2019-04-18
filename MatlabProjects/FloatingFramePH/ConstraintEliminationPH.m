clc
close all
clear all
addpath('./PH_matrices/')
addpath('./TITOP_utils/')
load Jode_pH; load Mode_pH; load Bode_pH

E = 2e11;
rho = 7900;
nu = 0.3;

b = 0.05;
h = 0.01;
A = b * h;

I = 1./12 * b * h^3;

EI = E * I;
L = 1;

Qode_pH = inv(Mode_pH);
sys_pH = ss(Jode_pH * Qode_pH, Bode_pH, Bode_pH' * Qode_pH, 0);
sys_TITOP = TwoPort_NElementsBeamTyRz(2, rho, A, L, E, I, 0);

sys_TITOPvel = sys_TITOP(1:2,1:2)*tf(1, [1 0]);
% sys_pH = minreal(sys_pH);
% sys_TITOPvel = minreal(sys_TITOPvel);

figure()
pzmap(sys_pH, 'r', sys_TITOPvel, 'b')

% sigma(sys_pH, 'b', sys_TITOPvel(1:2,1:2), 'r', {1e-10, 1e6})
% legend('PH', 'TITOP')


