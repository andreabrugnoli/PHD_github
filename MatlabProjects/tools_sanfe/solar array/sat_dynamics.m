clear all
close all
clc

l1 = 1;
l2 = 1;
t = 0.003;
rho1 = 2015;
rho2 = 2015;
load('Mdyn_SA1.mat')
load('Mdyn_SA2.mat')
%%

% Satellite
MB.cg = [0;0;0];                 % position of gravity center
                                 % in (0,X,Y,Z) (m)
MB.m = 100;                      % mass (kg)
MB.Ixx = 10;                     % Main Inertia in
MB.Iyy = 10;                     % (cg,X,Y,Z)
MB.Izz = 20;                     % (kgm^2)
MB.Ixy = 0;                      % Cross Inertia in
MB.Ixz = 0;                      % (cg,X,Y,Z)
MB.Iyz = 0;                      % (kgm^2)
MB.I = [MB.Ixx MB.Ixy MB.Ixz;MB.Ixy MB.Iyy MB.Iyz;MB.Ixz MB.Iyz MB.Izz];
DBG = [MB.m*eye(3) zeros(3,3);zeros(3,3) MB.I];


% South Panel
SA{1}.dyn = Mdyn_SA1;
SA{1}.cg = [1.5;0;0];            % position of gravity center CG1
                               % in (P1,X1,Y1,Z1) (m)
SA{1}.m = 3*rho1*l1*l2*t;
SA{1}.Jz = SA{1}.m/12*(1^2 + 3^2) + SA{1}.m*1.5^2;

SA{1}.TM = [ cos(90*pi/180)   -sin(90*pi/180)   0    % this is the transformation matrix associated with 
             sin(90*pi/180)    cos(90*pi/180)   0    % the rotation from main body reference frame (X,Y,Z)
             0                 0                1];  % to appendage frame (X1, Y1, Z1). (that is the 3x3 matrix
         
SA{1}.P = [0;1;0.5];             % position of connection point P1
                               % in (0,X,Y,Z) (m)
CGPvec=SA{1}.P-MB.cg;
SA{1}.tauCGP=[eye(3) -[0 CGPvec(3) -CGPvec(2);
       -CGPvec(3) 0 CGPvec(1);
       CGPvec(2) -CGPvec(1) 0] ;
       zeros(3) eye(3)];
                               
% North Panel
SA{2}.dyn = Mdyn_SA1;
SA{2}.cg = [1.5;0;0];            % position of gravity center CG1
                               % in (P1,X1,Y1,Z1) (m)
SA{2}.m = 3*rho2*l1*l2*t;
SA{2}.Jz = SA{2}.m/12*(1^2 + 3^2) + SA{2}.m*1.5^2;

SA{2}.P = [0;-1;0.5];             % position of connection point P1
                               % in (0,X,Y,Z) (m)
SA{2}.TM = [ cos(270*pi/180)   -sin(270*pi/180)   0    % this is the transformation matrix associated with 
             sin(270*pi/180)    cos(270*pi/180)   0    % the rotation from main body reference frame (X,Y,Z)
             0                 0                -1];  % to appendage frame (X1, Y1, Z1). (that is the 3x3 matrix
                                                     % of vectors X1, Y1 and Z1 coordinates in frame (X,Y,Z).
CGPvec=SA{2}.P-MB.cg;
SA{2}.tauCGP=[eye(3) -[0 CGPvec(3) -CGPvec(2);
       -CGPvec(3) 0 CGPvec(1);
       CGPvec(2) -CGPvec(1) 0] ;
       zeros(3) eye(3)];
   
   
%% CG TOT
nappend = 2;
masses=MB.m; 
cgs=MB.cg;
for i= 1: nappend,
    Rotcg_SA{i} = SA{i}.cg;
    Rotcg_SA{i} = SA{i}.TM * Rotcg_SA{i};
    masses=[masses SA{i}.m];
    cgs=[cgs SA{i}.P + Rotcg_SA{i}];
end
TotalMass = masses*ones(size(masses,2),1); % this is the total mass of the system
xcg_tot = cgs*masses'/TotalMass;

CGPvec=MB.cg-xcg_tot;
SCtauCGP=[eye(3) -[0 CGPvec(3) -CGPvec(2);
       -CGPvec(3) 0 CGPvec(1);
       CGPvec(2) -CGPvec(1) 0] ;
       zeros(3) eye(3)];


%% Dynamic Model

[a,b,c,d] = linmod('satellite_dynamics');
Dyna= ss(a,b,c,d);
figure;sigma(Dyna)




                                                     
