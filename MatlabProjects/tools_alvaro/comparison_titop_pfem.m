% Comparison TITOP PH
clear all; close all; clc;
manipulator_titop
manipulator_phpfem

% width = 15; height = 10; x0 = 0; y0 = 0; fntsize = 10;
% 
% w0 = 1e-5; wf = 1e6;
% figure('Units','centimeters','Position',[x0 y0 width height],'PaperPositionMode','auto');
% sigma(sys_ode, 'r', sys_dae, 'b', sys_titop, 'g', {w0, wf}) 
% 
% set(gca,...
% 'Units','normalized',...
% 'FontWeight','normal',...
% 'FontUnits','points',...
% 'FontSize',fntsize,...
% 'FontName','Times')
% ylabel('$\sigma(\omega)$',...
% 'FontUnits','points',...
% 'interpreter','latex',...
% 'FontSize',fntsize,...
% 'FontName','Times')
% xlabel('Frequency',...
% 'interpreter','latex',...
% 'FontUnits','points',...
% 'FontWeight','normal',...
% 'FontSize',fntsize,...
% 'FontName','Times')
% legend({'pHODE', 'pHDAE ', 'TITOP'},...
% 'interpreter','latex',...
% 'FontSize',fntsize,...
% 'FontName','Times',...
% 'Location','NorthEast')
% title('Singular values',...
% 'FontUnits','points',...
% 'interpreter','latex',...
% 'FontWeight','normal',...
% 'FontSize',fntsize,...
% 'FontName','Times')

figure(); sigma(sys_ode, 'r', {w0, wf})
figure(); sigma(sys_dae, 'b', {w0, wf})
figure(); sigma(sys_titop, 'g', {w0, wf}) 