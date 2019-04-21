close all; clc;
addpath('/home/a.brugnoli/GitProjects/MatlabProjects/PH/Settings/')
addpath('./DeLuca/')
model_constants;
ph_model;
titop_model;

path_fig = '/home/a.brugnoli/Plots_Videos/Matlab/Manipulator/';
fntsize = 13;
figure(); sigma(sys_phode, 'r', sys_titop, 'b', {w0, wf})
set_graphics_sigma(gca, fntsize)
legend('PHODE', 'TITOP')
% print(strcat(path_fig,'Sigma_TT_PH'),'-depsc')
figure(); sigma(sys_phode, 'r', sys_phdae, 'g', {w0, wf})
set_graphics_sigma(gca, fntsize)
legend('PHODE', 'PHDAE')
% print(strcat(path_fig,'Sigma_2PH'),'-depsc')

t0 = 0; tfin = 4;
paramNameValStruct.Solver = 'ode45'; 
paramNameValStruct.StartTime = num2str(t0);
paramNameValStruct.StopTime = num2str(tfin);
% paramNameValStruct.AbsTol = '1e-6';
% paramNameValStruct.RelTol = '1e-6';
sim_out = sim('manipulator_cl', paramNameValStruct);

t_out = sim_out.tout;
y_out = sim_out.yout;

alpha1_tt = y_out{1}.Values.Data;
alpha2_tt = y_out{2}.Values.Data;
dalpha1_tt = y_out{3}.Values.Data;
dalpha2_tt = y_out{4}.Values.Data;

alpha1_ph = y_out{5}.Values.Data;
alpha2_ph = y_out{6}.Values.Data;
dalpha1_ph = y_out{7}.Values.Data;
dalpha2_ph = y_out{8}.Values.Data;

alpha1_dl = y_out{9}.Values.Data;
alpha2_dl = y_out{10}.Values.Data;
dalpha1_dl = y_out{11}.Values.Data;
dalpha2_dl = y_out{12}.Values.Data;

col_tt = 'c.'
col_ph = 'b-'
col_dl = 'r--'
leg = {'Titop', 'pH', 'de Luca'}
figure(); plot(t_out, alpha1_tt, col_tt, t_out, alpha1_ph, col_ph, t_out, alpha1_dl, col_dl);
set_graphics(gca, 'Time (s)', '$\theta_1 (t)$ Degrees/s', leg, fntsize,'Angular displacement');
% print(strcat(path_fig,'alpha1'),'-depsc')

figure(); plot(t_out, alpha2_tt, col_tt, t_out, alpha2_ph, col_ph, t_out, alpha2_dl, col_dl);
set_graphics(gca, 'Time (s)', '$\theta_2 (t)$ Degrees/s', leg, fntsize,'Angular displacement');
% print(strcat(path_fig,'alpha2'),'-depsc')

figure(); plot(t_out, dalpha1_tt, col_tt, t_out, dalpha1_ph, col_ph, t_out, dalpha1_dl, col_dl);
set_graphics(gca, 'Time (s)', '$\dot{\theta}_1 (t)$ Degrees/s', leg, fntsize,'Angular velocity');
% print(strcat(path_fig,'dalpha1'),'-depsc')

figure(); plot(t_out, dalpha2_tt, col_tt, t_out, dalpha2_ph, col_ph, t_out, dalpha2_dl, col_dl);
set_graphics(gca, 'Time (s)', '$\dot{\theta}_2 (t)$ Degrees/s', leg, fntsize,'Angular velocity');
% print(strcat(path_fig,'dalpha2'),'-depsc')



