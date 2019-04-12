clc
close all
clear all
addpath('./PH_matrices/')
addpath('./TITOP_utils/')
load J_pH; load M_pH; load B_pH

E = 2e11;
rho = 7900;
nu = 0.3;

b = 0.05;
h = 0.01;
A = b * h;

I = 1./12 * b * h^3;

EI = E * I;
L = 1;

sys_pH = dss(J_pH, B_pH, B_pH', 0, M_pH);
sys_TITOP = TwoPort_NElementsBeamTyRz(3, rho, A, L, E, I, 0);

der = tf([1, 0], 1);
int = tf(1, [1, 0]);
sysder = [1 0 0 0;
          0 1 0 0;
          0 0 der 0;
          0 0 0 der];
sysint = [int 0 0 0;
          0 int 0 0;
          0 0 1 0;
          0 0 0 1];
sys_TITOPvel = sysder * sys_TITOP * sysint;


sigma(sys_pH, 'b', sys_TITOPvel, 'r', {1e-6, 1e6})
legend('PH', 'TITOP')



