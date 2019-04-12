% Equivalent Matlab script to the Simulink model
clear all
close all
clc

l1=1; %m
l2=1; %m
t=0.003; %m
rho = 2015; % kg/m^3
E = 70*10^9; % Pa
ni = 0.35; 
nx = 1;
ny = 4;

M_SA = 1147;
Jzz_SA = 212;

xi = 0.003;

P_id_1 = 5;
P_id_2 = 3;
P_id_3 = 3;
C_ids_1 = [4 8];
C_ids_2 = [7 4 8];
C_ids_3 = 7;

ind_1 = 0;
ind_2 = [1:3];
ind_3 = [1:3];

MtzRyRx_1 = NPort_KirchhoffTzRyRx(l1, l2, t, rho, E, ni, nx, ny, P_id_1, C_ids_1, xi);
% if ind_1~=0
%     MtzRyRx = invio(MtzRyRx_1, ind_1);
% end



MtzRyRx_2 = NPort_KirchhoffTzRyRx(l1, l2, t, rho, E, ni, nx, ny, P_id_2, C_ids_2, xi);
if ind_2~=0
    MtzRyRx_2 = invio(MtzRyRx_2, ind_2);
end

MtzRyRx_3 = NPort_KirchhoffTzRyRx(l1, l2, t, rho, E, ni, nx, ny, P_id_3, C_ids_3, xi);
if ind_3~=0
    MtzRyRx_3 = invio(MtzRyRx_3, ind_3);
end

A_sys3 = MtzRyRx_3.a;
B_sys3 = MtzRyRx_3.b;
C_sys3 = [MtzRyRx_3.c(1:3,:); -MtzRyRx_3.c(4:6,:)];
D_sys3 = [MtzRyRx_3.d(1:3,:); -MtzRyRx_3.d(4:6,:)];
sys3 = ss(A_sys3,B_sys3,C_sys3,D_sys3);

indices_sys2 = [7,8,9,4,5,6];
sys2_fb_sys3 = feedback( MtzRyRx_2, sys3,  indices_sys2, indices_sys2);

sys2_fb_sys3 = sys2_fb_sys3([1:3,10:12], [1:3,10:12]);

A_sys2 = sys2_fb_sys3.a;
B_sys2 = sys2_fb_sys3.b;
C_sys2 = [sys2_fb_sys3.c(1:3,:); -sys2_fb_sys3.c(4:6,:)];
D_sys2 = [sys2_fb_sys3.d(1:3,:); -sys2_fb_sys3.d(4:6,:)];
sys2 = ss(A_sys2,B_sys2,C_sys2,D_sys2);

indices_sys1 = [4,5,6,1,2,3];
sys1_fb_sys2 = feedback(MtzRyRx_1, sys2, indices_sys1, indices_sys1);

final_sys = sys1_fb_sys2(7:9,7:9);
final_sys_red = minreal(final_sys);

[A_sim,B_sim,C_sim,D_sim] = linmod('Sentinel_model');
sys_Simu = ss(A_sim,B_sim,C_sim,D_sim);
sys_Simu_red = minreal(sys_Simu);
Masses = dcgain(sys_Simu_red);
lambda_S = eig(sys_Simu_red.a);
lambda_M = eig(final_sys_red.a);

figure(1)
plot(real(lambda_S),imag(lambda_S),'bo',real(lambda_M),imag(lambda_M),'r*')
hold on 
plot([0 0],[min([imag(lambda_S);imag(lambda_M)]) max([imag(lambda_S);imag(lambda_M)]) ],'g')
legend('Simulink','Matlab','Imag Axis')

figure(2)
plot(real(lambda_M),imag(lambda_M),'r*')
hold on 
plot([0 0],[min([imag(lambda_M)]) max([imag(lambda_M)]) ],'g')
legend('Matlab','Imag Axis')

figure(3)
plot(real(lambda_S),imag(lambda_S),'bo')
hold on 
plot([0 0],[min([imag(lambda_S)]) max([imag(lambda_S)]) ],'g')
legend('Simulink','Imag Axis')
