clear all; close all; clc;
manipulator_phpfem
manipulatorSanfe_titop

w0 = 1e-5; wf = 1e6;


figure(); sigma(sys_phode, 'r', sys_titop, 'g', {w0, wf}) 
% figure(); sigma(sys_dae, 'b', {w0, wf}) 
% figure(); sigma(sys_titop, 'g', {w0, wf}) 

kp1 = 160;
kv1 = 11;
kp2 = 60;
Kv2 = 1.1;
alpha_rel = 60*pi/180;
rad_deg = 180/pi;