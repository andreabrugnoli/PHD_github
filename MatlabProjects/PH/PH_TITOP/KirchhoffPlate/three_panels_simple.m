% Equivalent Matlab script to the Simulink model
addpath('./TITOP_Kirchh/')
parameters;

xi = 0.00001;

P_id_1 = 3;
P_id_2 = 1;
P_id_3 = 1;
C_ids_1 = [2 6];
C_ids_2 = [2 3 4];
C_ids_3 = 3;

ind_1 = 0;
ind_2 = [1:3];
ind_3 = [1:3];

MtzRyRx_1 = NPort_KirchhoffTzRyRx(Lx, Ly, h, rho, E, nu, 1, 2, P_id_1, C_ids_1, xi);

MtzRyRx_2 = NPort_KirchhoffTzRyRx(Lx, Ly, h, rho, E, nu, 1, 1, P_id_2, C_ids_2, xi);
if ind_2~=0
    MtzRyRx_2 = invio(MtzRyRx_2, ind_2);
end

MtzRyRx_3 = NPort_KirchhoffTzRyRx(Lx, Ly, h, rho, E, nu, 1, 1, P_id_3, C_ids_3, xi);
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
final_sys_red = minreal(final_sys,10^-8);
lambda_M = eig(final_sys_red.a);

sys_titop = invio(final_sys,[1,2,3])*tf(1, [1,0]);
