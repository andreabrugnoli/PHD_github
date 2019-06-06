function [MtyRz] = TwoElementsBeamTyRz_modal(rho, S, ll, E, I, xi)
% M=TwoElementsBeamTyRz(ro,s,l,e,iz,xi) computes the 2 input/output
% model M (4x4) of a uniform beam characterized by:
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

l = ll/2;
M = (rho*S*l/420).*[156 22*l 54 -13*l 0 0; ...
    22*l 4*l^2 13*l -3*l^2 0 0; ....
    54 13*l 312 0 54 -13*l; ...
    -13*l -3*l^2 0 8*l^2 13*l -3*l^2; ...
    0 0 54 13*l 156 -22*l; ...
    0 0 -13*l -3*l^2 -22*l 4*l^2];

K = (E*I/l^3).*[12 6*l -12 6*l 0 0; ...
    6*l 4*l^2 -6*l 2*l^2 0 0; ...
    -12 -6*l 24 0 -12 6*l; ...
    6*l 2*l^2 0 8*l^2 -6*l 2*l^2; ...
    0 0 -12 -6*l 12 -6*l; ...
    0 0 6*l 2*l^2 -6*l 4*l^2];

P = [1 0 0 0 0 0; ...
0 1 0 0 0 0; ...
1 l 1 0 0 0; ...
0 1 0 1 0 0; ...
1 2*l 0 0 1 0; ...
0 1 0 0 0 1]

M2 = P'*M*P;
K2 = P'*K*P;


[V,D,W] = eig(K2(3:6,3:6),M2(3:6,3:6));


EV_C = V(3:4,:);

cdP = [0;0;0];
cdC = [ll;0;0];


tauCP =[1 ll;0 1];

Lp = V'*M2(1:2,3:6)'

m = rho*S*ll;
xcg = [l;0;0];
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


aa = [zeros(4,4) eye(4,4);-D zeros(4,4)];
bb = [zeros(4,4);EV_C' -Lp];
cc = [-EV_C*D zeros(2,4);Lp'*D zeros(2,4)];
dd = [EV_C*EV_C' tauCP-EV_C*Lp;(tauCP-EV_C*Lp)' -Dpa0];

MtyRz = ss(aa,bb,cc,dd);

end

