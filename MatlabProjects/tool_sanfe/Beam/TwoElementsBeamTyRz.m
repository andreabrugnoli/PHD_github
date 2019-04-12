function [MtyRz] = TwoElementsBeamTyRz(rho, S, ll, E, I, xi)
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
0 1 0 0 0 1];

Mz = P'*M*P;

Kz = P'*K*P;
phiC = [0 0 1 0;...
    0 0 0 1];

tau = [1 2*l;0 1];

Mm1=inv(Mz(3:6,3:6));
Mm1K=Mm1*Kz(3:6,3:6);
Mm1K0=Mm1K;


L=Mm1*Mz(3:6,1:2);
[V,D]=eig(Mm1K0);
a=[zeros(4) eye(4);-Mm1K -V*2*xi*sqrt(D)*inv(V)];
b=[zeros(4,4);[Mm1*phiC' -L]];
c=[phiC; -Mz(1:2,3:6)]*a(5:8,:);
d=[phiC*Mm1*phiC' tau-phiC*L;(tau-phiC*L)' -Mz(1:2,1:2)+Mz(1:2,3:6)*L];
MtyRz=ss(a,b,c,d);


end