clear all
close all
clc


addpath('D:\f.sanfedino\Synthesis\inversemodel\KirchhoffPlate');
l1 = 1;
l2 = 1;
t = 0.003;
rho = 2015;%*1.2;
E = 69.8692e9;%*0.8;
G = 22.1615e9;
ni = E/2/G - 1;
xi = 0.001;
nx = 6;
ny = 6;

% First appendage
P_id = 22;
C_ids = [14;42];

[M1] = NPort_KirchhoffTzRyRx(l1,l2,t,rho,E,ni,nx,ny,P_id,C_ids,xi);
 [M1,h] = balreal(M1);
 M1r = modred(M1,21:length(h)); 
% M1r2 = balred(M1,501:length(h));

% Middle appendage
P_id = 8;
C_ids = [36;14;42];
[M2] = NPort_KirchhoffTzRyRx(l1,l2,t,rho,E,ni,nx,ny,P_id,C_ids,xi);
%M2 = makeScrew(M2,0.003,1/12,3);
M2_inv = invio(M2,1:3);
M2_inv2 = invio(M2,10:12);

% Final appendage
P_id = 8;
C_ids = [36];
[M3] = NPort_KirchhoffTzRyRx(l1,l2,t,rho,E,ni,nx,ny,P_id,C_ids,xi);
%M3 = makeScrew(M3,0.003,1/12,1);
M3_inv = invio(M3,1:3);
M3_invMin = minreal(M3_inv*[1 0.5 0;0 1 0;0 0 1;eye(3,3)]);
M3_inv2 = invio(M3,4:6);

% angle 1 (between first appendage and middle appendage)
pivotdir1 = [0;1;0];
angle1 = 135;
V_rev1 = pivotdir1/norm(pivotdir1,2);
theta1 = angle1*pi/360;
Mpass1=quat2dcm([cos(theta1) V_rev1(1).*sin(theta1) V_rev1(2).*sin(theta1) V_rev1(3).*sin(theta1)])';
Mpass1 = [Mpass1 zeros(3,3);zeros(3,3) Mpass1];

% angle 2 (between middle appendage and last appendage)
pivotdir2 = [0;1;0];
angle2 = 0;
V_rev2 = pivotdir2/norm(pivotdir2,2);
theta2 = angle2*pi/360;
Mpass2=quat2dcm([cos(theta2) V_rev2(1).*sin(theta2) V_rev2(2).*sin(theta2) V_rev2(3).*sin(theta2)])';
Mpass2 = [Mpass2 zeros(3,3);zeros(3,3) Mpass2];



% Modèle Total 3 panneaux solaires

[aa,bb,cc,dd] = linmod('ThreecellsSolarArray');
Mdyn = ss(aa,bb,cc,dd);
Mdynr = minreal(Mdyn);

% % Modèle Total 3 panneaux solaires with spring/damping
% K = 1e2;
% C = 1;
% [aa,bb,cc,dd] = linmod('ThreecellsSolarArrayWithSprings_damp');
% MdynK_damp = ss(aa,bb,cc,dd);
% MdynrK_damp = minreal(MdynK_damp);
% rsys2damp = modred(MdynK_damp,50:length(h)); % State-space model with 3 outputs, 3 inputs, and 29 states -> miglior comportamento su y
% %rsys3 = balred(Mdyn,20);
% Mdyn_SA_damp = rsys2damp;

% Model reduction
[Mdyn,h] = balreal(Mdyn);
%rsys1 = modred(Mdyn,21:length(h)); % State-space model with 3 outputs, 3 inputs, and 19 states

rsys2 = modred(Mdyn,31:length(h)); % State-space model with 3 outputs, 3 inputs, and 29 states -> miglior comportamento su y
%rsys3 = balred(Mdyn,20);
Mdyn_SA = rsys2;