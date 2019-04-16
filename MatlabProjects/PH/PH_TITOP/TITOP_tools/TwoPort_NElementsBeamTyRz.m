function [MtyRz] = TwoPort_NElementsBeamTyRz(ne, rho, S, ll, E, I, xi)
% [MtyRz] = TwoPort_NElementsBeamTyRz(ne, rho, S, ll, E, I, xi)
% model M (4x4) of a uniform beam characterized by:
%   * finite elements number: ne
%   * mass density: rho  (Kg/m^3),
%   * section: S (m^2),
%   * lenght: ll (m),
%   * Young modulus: E (Pascal or N/m^2),
%   * second moment of aera w.r.t z axis: I (m^4),
%   * xi: arbitrary damping ratio for all flexible modes.
%          ^ y(x)
%          |
%          |
%          x=======================x---------> x
%         /P(x=0)                 C(x=l)
%        /
%       z
%   Only pure flexion in the plane (P,x,y) is considered.
%
%   The 4 inputs of M are:
%      * the external force F_Cy (along y) and torque T_Cz (around z) 
%        applied to the beam at point C,
%      * the linear ddot(y)_P (along y) and angular ddot(theta)_P
%        (around z) accelerations at point P.
%   The 4 outputs of M are:
%      * the linear ddot(y)_C (along y) and angular ddot(theta)_C
%        (around z) accelerations at point C,
%      * the external force F_Py (along y) and torque T_Pz (around z) 
%        applied to the beam at point P.
% Global nodes Coordinates

nnodes = ne+1;
GN_coord = gl_nodeCoord(nnodes,ne,ll);


l = ll/ne;
Me = (rho*S*l/420).*[156 22*l 54 -13*l; ...
    22*l 4*l^2 13*l -3*l^2 ; ....
    54 13*l 156 -22*l; ...
    -13*l -3*l^2 -22*l 4*l^2 ];

Ke = (E*I/l^3).*[12 6*l -12 6*l; ...
    6*l 4*l^2 -6*l 2*l^2; ...
    -12 -6*l 12 -6*l; ...
    6*l 2*l^2 -6*l 4*l^2];

% FE Assembling

M = zeros(2*(ne+1),2*(ne+1));
M(1:4,1:4) = Me;
K = zeros(2*(ne+1),2*(ne+1));
K(1:4,1:4) = Ke;
for i=1:ne-1
    M(2*i+1:2*i+4,2*i+1:2*i+4) = M(2*i+1:2*i+4,2*i+1:2*i+4) + Me;
    K(2*i+1:2*i+4,2*i+1:2*i+4) = K(2*i+1:2*i+4,2*i+1:2*i+4) + Ke;
end


P = eye(2,2);
for i=1:ne
    P = [P; [1 i*l;0 1]];
end
P = [P [zeros(2,2*ne);eye(2*ne)]];


M2 = P'*M*P;
K2 = P'*K*P;

[V,D,W] = eig(K2(3:2*(ne+1),3:2*(ne+1)),M2(3:2*(ne+1),3:2*(ne+1)));
nb_mode = length(D(1,:));

EV_C = V(2*ne-1:2*ne,:);

% V'*M2(3:end,3:end)*V
% V'*K2(3:end,3:end)*V

cdP = [0;0;0];
cdC = [ll;0;0];


tauCP =[1 ll;0 1];

Lp = V'*M2(1:2,3:2*(ne+1))';

m = rho*S*ll;
xcg = [ll/2;0;0];
CGPvec=cdP-xcg;

tauCGP=[eye(3) -[0 CGPvec(3) -CGPvec(2);
    -CGPvec(3) 0 CGPvec(1);
    CGPvec(2) -CGPvec(1) 0] ;
    zeros(3) eye(3)];

% 
% psi = [1 0;0 1;1 l;0 1;1 2*l;0 1];
% psi'*M*psi
% [V2,D2,W2] = eig(K,M);
% Lp = V2'*M*psi


% 
% MO_xcg = [m.*eye(3) zeros(3);zeros(3) blkdiag(0,0,I)];
% 
% Dpa = tauCGP'*MO_xcg*tauCGP
% Dpa = Dpa([2,6],[2,6])

Dpa = M2(1:2,1:2);
Dpa0 = Dpa - Lp'*Lp;


aa = [zeros(2*ne,2*ne) eye(2*ne,2*ne);-D zeros(2*ne,2*ne)];
bb = [zeros(2*ne,4);EV_C' -Lp];
cc = [-EV_C*D zeros(2,2*ne);Lp'*D zeros(2,2*ne)];
dd = [EV_C*EV_C' tauCP-EV_C*Lp;(tauCP-EV_C*Lp)' -Dpa0];

MtyRz = ss(aa,bb,cc,dd);

% fprintf('Number of modes found: %d \n',nb_mode)
% 
% prompt = 'How many mode shapes do you want to visalize? \n';
% nb_mode_shapes = input(prompt);
% if nb_mode_shapes == 0
%     return;
% else
%     printModes(nb_mode_shapes, GN_coord, V, D);
% end

end 
