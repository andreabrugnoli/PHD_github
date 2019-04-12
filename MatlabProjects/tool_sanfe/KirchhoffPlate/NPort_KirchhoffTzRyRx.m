function [MtzRyRx] = NPort_KirchhoffTzRyRx(l1,l2,t,rho,E,ni,nx,ny,P_id,C_ids,xi)
% [MtzRyRx] = NPort_KirchhoffTzRyRx(l1,l2,t,rho,E,ni,nx,ny,P_id,C_ids,xi)
% Finite Element Model MtzRyRx (6x6) of a uniform Kirchhoff plate characterized by:
%   * plate length along x: l1 (m),
%   * plate length along y: l2 (m),
%   * plate thikness along z: t (m),
%   * mass density: rho  (Kg/m^3),
%   * Young modulus: E (Pascal or N/m^2),
%   * Poisson ratio: ni,
%   * number of 4-node plate elements along x: nx,
%   * number of 4-node plate elements along y: ny,
%   * ID number of parent node P: P_id (see NOTE for nodal convention),
%   * ID number of children nodes (vector): C_ids (see NOTE for nodal convention),
%   * xi: arbitrary damping ratio for all flexible modes.
%
%          ^ z    y
%          |     /  
%          |    /             C (x=0, y=ly) 
%          |   -----------------------/
%          |  /                  *   / 
%          | * P(xp = 0,yp)         / 
%          |/                      / 
%          .----------------------/----------> x
%     (x=0, y=0)                
%       
%       
%   Only pure flexion in the plane (P,x,z) and torsion in the plane (P,y,z) are considered.
%
%   The 6 inputs of MtzRyRx are:
%      * the external force F_Cz (along z) and torques T_Cx (around x) 
%        and T_Cy (around y) applied to the plate at point C,
%      * the linear ddot(z)_P (along z) and angular ddot(theta_x)_P
%        (around x) and ddot(theta_y)_P (around y) accelerations at point P.
%   The 6 outputs of M are:
%      * the linear ddot(z)_C (along z) and angular ddot(theta_x)_C
%        (around x) and ddot(theta_y)_C (around y) accelerations at point C,
%      * the external force F_Pz (along z) and torques T_Px (around x) 
%        and T_Py (around y) applied to the plate at point P.
%
%  NOTE: Example for a 4x5 elements plate: global node convention
%
%  25----26----27----28----29----30
%   | 16 |  17 |  18 |  19 |  20 |
%  19----20----21----22----23----24
%   | 11 |  12 |  13 |  14 |  15 |
%  13----14----15----16----17----18
%   | 6  |  7  |  8  |  9  |  10 |
%   7----8-----9-----10----11----12
%   | 1  |  2  |  3  |  4  |  5  |
%   1----2-----3-----4-----5-----6
%


% Element dimensions
lx = l1/nx;
ly = l2/ny;

nnodes = (nx+1)*(ny+1); % number of global nodes
n_free = length(C_ids);

% Global nodes Coordinates
GN_coord = gl_nodeCoord(nnodes,nx,lx,ly);

% Element Kinetic and Mass Matrices
[Kel,Mel] = makeKirchhoffElement(rho,E,ni,lx,ly,t);

% Assembling
[Kas,Mas,N] = makeAssembling(nx,ny,Kel,Mel);

% Constraints
[K2,M2] = makeClumpP(GN_coord,Kas,Mas,P_id);

% Eigenvectors and EigenValues

Mll = [M2(1:3*(P_id-1),1:3*(P_id-1)) M2(1:3*(P_id-1),3*(P_id-1)+4:N);...
       M2(3*(P_id-1)+4:N,1:3*(P_id-1)) M2(3*(P_id-1)+4:N,3*(P_id-1)+4:N)];
Kll = [K2(1:3*(P_id-1),1:3*(P_id-1)) K2(1:3*(P_id-1),3*(P_id-1)+4:N);...
       K2(3*(P_id-1)+4:N,1:3*(P_id-1)) K2(3*(P_id-1)+4:N,3*(P_id-1)+4:N)];

[V,D] = eig(Kll,Mll);
nb_mode = length(D(1,:));

% Eigenvectors for points C
EV_C = makeEigenVectC(V,P_id,C_ids);


% Modal Participation Factors
Mrl = [M2(3*(P_id-1)+1:3*(P_id-1)+3,1:3*(P_id-1)) M2(3*(P_id-1)+1:3*(P_id-1)+3,3*(P_id-1)+4:N)];
Lp = V'*Mrl';

% Residual Mass of Rigid Model
DPa = M2(3*(P_id-1)+1:3*(P_id-1)+3,3*(P_id-1)+1:3*(P_id-1)+3);
DPa0 = DPa - Lp'*Lp;

% Kinematic model from P to C
tauCP = makeKinematicPC(GN_coord,P_id,C_ids);

% N-port Dynamic model
MtzRyRx = Nport_dynamics(nb_mode, Lp, D, EV_C, xi, DPa0, tauCP, n_free);


fprintf('Number of modes found: %d \n',nb_mode)

prompt = 'How many mode shapes do you want to visalize? \n';
nb_mode_shapes = input(prompt);
if nb_mode_shapes == 0
    return;
else
    printModes(nb_mode_shapes, P_id, GN_coord, V, D);
end

end 
