addpath('./Matrices_manipulator/')
load B_ode; load J_ode; load Q_ode;
load B_dae; load J_dae; load E_dae;

sys_phode = ss(J_ode * Q_ode, B_ode, B_ode'* Q_ode, 0);
sys_phdae = dss(J_dae, B_dae, B_dae', 0, E_dae);

% width = 15; height = 10; x0 = 0; y0 = 0; fntsize = 20;
% 
% figure('Units','centimeters','Position',[x0 y0 width height],'PaperPositionMode','auto');
% sigma(sys_ode, 'r', {w0, wf}) 
% figure('Units','centimeters','Position',[x0 y0 width height],'PaperPositionMode','auto');
% sigma(sys_dae, 'b', {w0, wf})



% rad_deg = 180/pi;
% 
% 
% kp1 = 160;
% kv1 = 11;
% kp2 = 60;
% Kv2 = 1.1;
% 
% % kp1=61.71; kp2=13.3833; kv1=8.72; kv2=1.89;
% alpha_rel = 60*pi/180;