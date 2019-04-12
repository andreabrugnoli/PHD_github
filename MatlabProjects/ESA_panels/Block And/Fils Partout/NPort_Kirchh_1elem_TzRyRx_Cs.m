function [MtzRyRx] = NPort_Kirchh_1elem_TzRyRx_Cs(lx, ly, t, rho, E, ni, P_id, Cs_xy, xi)
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
%   * ID number of parent node P: 1,
%   * Location of children inside the element: Cs_xy (N_children*2 matrix),
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

% Global Coordinates
GN_coord = [0,  0;
            lx, 0;
            0,  ly;
            lx, ly];
        
% Number of children
[n_free, ~] = size(Cs_xy); % number of global nodes

% Element Kinetic and Mass Matrices
[Kel,Mel] = makeKirchhoffElement(rho,E,ni,lx,ly,t);

[Kas,Mas] = makeAssembling(1,1,Kel,Mel);

% Constraints
[K2,M2] = makeClampP(GN_coord,Kas,Mas,P_id);

% Eigenvectors and EigenValues
N = 12;

Mff = [M2(1:3*(P_id-1),1:3*(P_id-1)) M2(1:3*(P_id-1),3*(P_id-1)+4:N);...
       M2(3*(P_id-1)+4:N,1:3*(P_id-1)) M2(3*(P_id-1)+4:N,3*(P_id-1)+4:N)];
Kff = [K2(1:3*(P_id-1),1:3*(P_id-1)) K2(1:3*(P_id-1),3*(P_id-1)+4:N);...
       K2(3*(P_id-1)+4:N,1:3*(P_id-1)) K2(3*(P_id-1)+4:N,3*(P_id-1)+4:N)];

[Phi,omega2] = eig(Kff,Mff);
nb_mode = length(omega2(1,:));

% Modal Participation Factors
Mrf = [M2(3*(P_id-1)+1:3*(P_id-1)+3,1:3*(P_id-1)) M2(3*(P_id-1)+1:3*(P_id-1)+3,3*(P_id-1)+4:N)];
Lp = Phi'*Mrf';  

% Residual Mass of Rigid Model
M_rr = M2(3*(P_id-1)+1:3*(P_id-1)+3,3*(P_id-1)+1:3*(P_id-1)+3);
DPa0 = M_rr - Lp'*Lp;

% Locating closest node to C points
C_ids = fromC2nodes(Cs_xy, GN_coord, P_id);

% Eigenvectors for points C
EV_C = makeEigenVectC(Phi,P_id,C_ids);

% Kinematic model from P to C
tauCP = makeKinematicPC(GN_coord,P_id,C_ids);
% Kinematic model from P to C
tauCN = makeKinematicCN(GN_coord,C_ids,Cs_xy);

% Shape matrix for C points
U_C = evalShapeMatrix(GN_coord, Cs_xy, P_id);

% N-port Dynamic model
[a,b,c,d] = Nport_dynamics_Cxy(nb_mode, Lp, omega2, EV_C, Phi, xi, DPa0, tauCP, tauCN, U_C, n_free);

MtzRyRx = ss(a,b,c,d);

end















