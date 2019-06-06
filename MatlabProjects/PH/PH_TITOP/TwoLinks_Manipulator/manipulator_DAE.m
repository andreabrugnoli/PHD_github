close all; clc;
% clear all;
addpath('/home/a.brugnoli/GitProjects/MatlabProjects/PH/Settings/')
parameters;
ph_model;

path_fig = '/home/a.brugnoli/Plots_Videos/Matlab/Manipulator/';
fntsize = 13;
% figure(); sigma(sys_phode, 'r', sys_phdae, 'g', {w0, wf})
% set_graphics_sigma(gca, fntsize)
% legend('PHODE', 'PHDAE')
% print(strcat(path_fig,'Sigma_2PH'),'-depsc')

t0 = 0; tfin = 4;
paramNameValStruct.Solver = 'ode23t'; 
paramNameValStruct.StartTime = num2str(t0);
paramNameValStruct.StopTime = num2str(tfin);
paramNameValStruct.SaveState      = 'on';
paramNameValStruct.StateSaveName  = 'xout';
% paramNameValStruct.AbsTol = '1e-6';
% paramNameValStruct.RelTol = '1e-6';
sim_out = sim('manipulator_dae', paramNameValStruct);

t_out = sim_out.tout;
y_out = sim_out.yout;
x_out = sim_out.xout;

alpha1_ph = y_out{1}.Values.Data;
alpha2_ph = y_out{2}.Values.Data;
dalpha1_ph = y_out{3}.Values.Data;
dalpha2_ph = y_out{4}.Values.Data;


col_ph = 'b-'
dal2_ph = sim_out.xout{1}.Values.Data(:, 4)-sim_out.xout{1}.Values.Data(:, 1);
figure(); plot(t_out, dal2_ph*rad_deg, col_ph);

dal2_ph = sim_out.xout{1}.Values.Data * sys_phdae.B(:,2);
figure(); plot(t_out, dal2_ph*rad_deg, col_ph);

% for i=1:7
%     x1_ph = sim_out.xout{1}.Values.Data(:, i);
%     figure(i); plot(t_out, x1_ph, col_ph);
% %     leg = {'pHDAE'};
% %   set_graphics(gca, 'Time (s)', '$\theta_1 (t)$ Degrees/s', leg, fntsize,'Angular displacement');
% end

% figure(); plot(t_out, alpha1_ph, col_ph);
% leg = {'pHDAE'};
% set_graphics(gca, 'Time (s)', '$\theta_1 (t)$ Degrees/s', leg, fntsize,'Angular displacement');
% % print(strcat(path_fig,'alpha2'),'-depsc')
% 
% figure(); plot(t_out, alpha2_ph, col_ph);
% leg = {'pHDAE'};
% set_graphics(gca, 'Time (s)', '$\theta_2 (t)$ Degrees/s', leg, fntsize,'Angular displacement');
% % print(strcat(path_fig,'alpha2'),'-depsc')
% 
% figure(); plot(t_out, dalpha1_ph, col_ph);
% leg = {'pHDAE'};
% set_graphics(gca, 'Time (s)', '$\dot{\theta}_1 (t)$ Degrees/s', leg, fntsize,'Angular velocity');
% % print(strcat(path_fig,'alpha2'),'-depsc')
% 
% figure(); plot(t_out, dalpha2_ph, col_ph);
% leg = {'pHDAE'};
% set_graphics(gca, 'Time (s)', '$\dot{\theta}_2 (t)$ Degrees/s', leg, fntsize,'Angular velocity');
% % print(strcat(path_fig,'alpha2'),'-depsc')

