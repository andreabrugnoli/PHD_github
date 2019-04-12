clc
close all
clear all
addpath('./PH_matrices/')
addpath('./TITOP_utils/')

load J_pH; load Q_pH; load B_pH

E = 2e11;
rho = 7900;
nu = 0.3;

b = 0.05;
h = 0.01;
A = b * h;

I = 1./12 * b * h^3;

EI = E * I;
L = 1;

sys_pH = ss(J_pH*Q_pH, B_pH, B_pH'*Q_pH, 0);

MtyRz = TwoPort_NElementsBeamTyRz(3, rho, A, L, E, I, 0);
sys_TITOP = invio(MtyRz, [3, 4]) * tf(1, [1, 0]);

% sys_pH = minreal(sys_pH);
% sys_TITOP = minreal(sys_TITOP);

sigma(sys_pH(:, :), 'b', sys_TITOP(:, :), 'r', {1e-3, 1e6})
legend('PH', 'TITOP')



