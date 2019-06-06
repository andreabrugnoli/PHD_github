

f06_file = 'D:\f.sanfedino\Synthesis\solar_array\array_attached\solar_pan_atch.f06';
bdf_file = 'D:\f.sanfedino\Synthesis\solar_array\array_attached\solar_pan_atch.bdf';
type = 'f'; % Type of analysis: 'r' for rigid, 'f' for flexible
dmp = 0.001;
id_clamped = 56; % id clamped node 
ids_free = [44,88]; % id free nodes
n_free = length(ids_free);
nb_mode = 20; % number of considered modes

pivotdir = [1;0;0];
channel = 3;

[M, J, xcg, LS, omega, EV_C, zeta, DPa0, cdP, cdC, tauCP, flagFatal] = getDataFromNASTRAN(f06_file,bdf_file,type,dmp,id_clamped,ids_free,nb_mode);
Mdyn1 = Nport_dynamics(nb_mode, LS, omega, EV_C, zeta, DPa0, tauCP, n_free); % Direct Model


% Middle appendage

f06_file = 'D:\f.sanfedino\Synthesis\solar_array\array_midle\solar_array_middle.f06';
bdf_file = 'D:\f.sanfedino\Synthesis\solar_array\array_midle\solar_array_middle.bdf';
type = 'f'; % Type of analysis: 'r' for rigid, 'f' for flexible
dmp = 0.001;
id_clamped = 34; % id clamped node 
ids_free = [78,44,88]; % id free nodes
n_free = length(ids_free);
nb_mode = 20; % number of considered modes

[M, J, xcg, LS, omega, EV_C, zeta, DPa0, cdP, cdC, tauCP, flagFatal] = getDataFromNASTRAN(f06_file,bdf_file,type,dmp,id_clamped,ids_free,nb_mode);
Mdyn2 = Nport_dynamics(nb_mode, LS, omega, EV_C, zeta, DPa0, tauCP, n_free); % Direct Model
Mdyn2screw = Nport_dynamics_withScrew(nb_mode, LS, omega, EV_C, zeta, DPa0, tauCP, n_free, M, J);

Mdyn2_123456 = invio(Mdyn2,[1:6]);
Mdyn2_19_24 = invio(Mdyn2,[19:24]);
Mdyn2screw_123456 = invio(Mdyn2screw,[1:6]);
Mdyn2screw_19_24 = invio(Mdyn2screw,[19:24]);
% Final appendage

f06_file = 'D:\f.sanfedino\Synthesis\solar_array\array_midle\solar_array_middle.f06';
bdf_file = 'D:\f.sanfedino\Synthesis\solar_array\array_midle\solar_array_middle.bdf';
type = 'f'; % Type of analysis: 'r' for rigid, 'f' for flexible
dmp = 0.001;
id_clamped = 34; % id clamped node 
ids_free = [78,98]; % id free nodes
n_free = length(ids_free);
nb_mode = 20; % number of considered modes

[M, J, xcg, LS, omega, EV_C, zeta, DPa0, cdP, cdC, tauCP, flagFatal] = getDataFromNASTRAN(f06_file,bdf_file,type,dmp,id_clamped,ids_free,nb_mode);
Mdyn3 = Nport_dynamics(nb_mode, LS, omega, EV_C, zeta, DPa0, tauCP, n_free);
Mdyn3screw = Nport_dynamics_withScrew(nb_mode, LS, omega, EV_C, zeta, DPa0, tauCP, n_free, M, J);
Mdyn3_16 = invio(Mdyn3,[1:6]);
Mdyn3_7_12 = invio(Mdyn3,[7:12]);
Mdyn3screw_16 = invio(Mdyn3screw,[1:6]);
Mdyn3screw_7_12 = invio(Mdyn3screw,[7:12]);

[aa,bb,cc,dd] = linmod('ThreecellsSolarArray_debris');
%[aa,bb,cc,dd] = linmod('ThreecellsSolarArray_screw_without_pivot');
SA_Dyn = ss(aa,bb,cc,dd);
%%

% Satellite
MB.cg = [-0.06;-0.03;2.51];                 % position of gravity center
                                 % in (0,X,Y,Z) (m)
MB.m = 5160/6.45;                      % mass (kg)
MB.Ixx = 25541/6.45;                     % Main Inertia in
MB.Iyy = 26514/6.45;                     % (cg,X,Y,Z)
MB.Izz = 11997/6.45;                     % (kgm^2)
MB.Ixy = -406.45/6.45;                      % Cross Inertia in
MB.Ixz = 20486.45/6.45;                      % (cg,X,Y,Z)
MB.Iyz = 256.45/6.45;                      % (kgm^2)
MB.I = [MB.Ixx MB.Ixy MB.Ixz;MB.Ixy MB.Iyy MB.Iyz;MB.Ixz MB.Iyz MB.Izz];
DBG = [MB.m*eye(3) zeros(3,3);zeros(3,3) MB.I];


% North Panel
SA{1}.cg = [1.5;0;0];            % position of gravity center CG1
                               % in (P1,X1,Y1,Z1) (m)
SA{1}.m = 3*56.2710;

SA{1}.TM = [ cos(90*pi/180)   -sin(90*pi/180)   0    % this is the transformation matrix associated with 
             sin(90*pi/180)    cos(90*pi/180)   0    % the rotation from main body reference frame (X,Y,Z)
             0                 0                1];  % to appendage frame (X1, Y1, Z1). (that is the 3x3 matrix
         
SA{1}.P = [0;1.76;0.655];             % position of connection point P1
                               % in (0,X,Y,Z) (m)
CGPvec=SA{1}.P-MB.cg;
SA{1}.tauCGP=[eye(3) -[0 CGPvec(3) -CGPvec(2);
       -CGPvec(3) 0 CGPvec(1);
       CGPvec(2) -CGPvec(1) 0] ;
       zeros(3) eye(3)];
                               
% South Panel
SA{2}.cg = [1.5;0;0];            % position of gravity center CG1
                               % in (P1,X1,Y1,Z1) (m)
SA{2}.m = 3*56.2710;

SA{2}.P = [0;-1.76;0.655];             % position of connection point P1
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

[a,b,c,d] = linmod('debris_model');
DynDebris = ss(a,b,c,d);




                                                     
