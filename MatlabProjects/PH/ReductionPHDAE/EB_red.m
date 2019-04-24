clc
close all
clear all

addpath('./EB_Matrices/')
load E; load J; load B; 
% load F; load G; load H; load L; 
% load M_22P; load J_22P; load Q_22P; load B_22P; load T_z2u.mat; load T_z2u.mat;
% load Etil; load Jtil; load Btil; 
load Er; load Jr; load Br;
% load Er1; load Jr1; load Br1;
% load Er2; load Jr2; load Br2;

% sys_fullDAE = dss(J,B,B',0,E);
% sys_fullDAEmin = minreal(sys_fullDAE);
% 
% sys_fullODE = ss(F, G, H, L);
% sys_fullODEmin = minreal(sys_fullODE);
% 
% sys_22P = ss(J_22P*Q_22P, B_22P, B_22P'*Q_22P, 0);
% sys_22Pmin = minreal(sys_22P);

sys_full = dss(J,B,B',0,E);
sys_full = minreal(sys_full);
% [isp, sys_full] = isproper(sys_full);

% sys_reg = dss(Jtil,Btil,Btil',0,Etil);
% sys_reg = minreal(sys_reg);
% [isp, sys_red] = isproper(sys_red);

sys_red = dss(Jr,Br,Br',0,Er);
sys_red = minreal(sys_red);

% sys_red1 = dss(Jr1,Br1,Br1',0,Er1);
% sys_red1 = minreal(sys_red1);

% sys_red2 = dss(Jr2,Br2,Br2',0,Er2);
% sys_red2 = minreal(sys_red2);
 
% figure(1)
% bode(sys_fullODEmin, 'b', sys_22P, 'r', sys_fullDAEmin, 'y')

figure()
sigma(sys_full, 'b', sys_red, 'r')
legend('EB n = 30', 'EB n = 10')

% t_fin = 1;
% n_ev = 1000;
% t_ev = linspace(0, t_fin, n_ev);
% x_DAE = step(sys_fullDAEmin, t_ev);
% x_ODE = step(sys_fullODEmin, t_ev);
% x_22P = step(sys_22Pmin, t_ev);
% 
% figure();
% plot(t_ev, x_DAE, 'b', t_ev, x_ODE, 'r', t_ev, x_22P, 'y') 


% figure(3)
% bode(sys_full, sys_red2)


%eig_full = eig(E, J)
%eig_red = eig(Er, Jr)

