function [M] = TwoPortBeamTxTyRz(ro,s,l,e,iz,xi)
% M=TwoPortBeamTxTyRz(ro,s,l,e,iz,xi) computes the 2 port
% model M (6x6) of a uniform beam characterized by:
%   * mass density: ro  (Kg/m^3),
%   * section: s (m^2),
%   * lenght: l (m),
%   * Young modulus: e (Pascal or N/m^2),
%   * second moment of aera w.r.t z axis: iz (m^4),
%   * xi: arbitrary damping ratio for all flexible modes.
%          ^ y(x)
%          |
%          |
%          x=======================x---------> x
%         /P(x=0)                 C(x=l)
%        /
%       z
%   Only pure flexion and traction in the plane (P,x,y) are 
%   considered.
%
%   The 6 inputs of M are:
%      * the external forces F_Cx (along x), F_Cy (along y) and 
%        torque T_Cz (around z) applied to the beam at point C,
%      * the linear ddot(x)_P (along x) ddot(y)_P (along y) and 
%        angular ddot(theta)_P (around z) accelerations at point P.
%   The 4 outputs of M are:
%      * the linear ddot(x)_C (along x) ddot(y)_C (along y) and 
%        angular ddot(theta)_C (around z) accelerations at point C,
%      * the external force F_Px (along x), F_Py (along y) and 
%        torque T_Pz (around z) applied by the beam at point P.
%
%   This fonction supports uncertain parameters (see ureal).
%
%   See also: MKelemu, TwoPortBeam

%  D. Alazard (05/2014)
phiC=[0 1 0 0;0 0 1 0];
tau=[1 l;0 1];
[Mz,Kz] = MKelemu(ro,s,l,e,iz);
Mm1=inv(Mz(3:6,3:6));
if isuncertain(Mm1), Mm1 = simplify(Mm1,'full');end
Mm1K=Mm1*Kz(3:6,3:6);
if isuncertain(Mm1K), 
    Mm1K = simplify(Mm1K,'full');
    Mm1K0=Mm1K.NominalValue;
else
    Mm1K0=Mm1K;
end
L=Mm1*Mz(3:6,1:2);
if isuncertain(L), L = simplify(L,'full');end
[V,D]=eig(Mm1K0);
a=[zeros(4) eye(4);-Mm1K -V*2*xi*sqrt(D)*inv(V)];
b=[zeros(4,4);[Mm1*phiC' -L]];
if isuncertain(b), b = simplify(b,'full');end
%c=[[-phiC*Mm1K; Mz(1:2,3:6)*Mm1K] [-phiC;Mz(1:2,3:6)]*V*2*xi*sqrt(D)*inv(V)];
c=[phiC; -Mz(1:2,3:6)]*a(5:8,:);
if isuncertain(c), c = simplify(c,'full');end
d=[phiC*Mm1*phiC' tau-phiC*L;(tau-phiC*L)' -Mz(1:2,1:2)+Mz(1:2,3:6)*L];
%d=[zeros(2) tau;tau' -Mz(1:2,1:2)]+[-phiC;Mz(1:2,3:6)]*b(5:8,:);
if isuncertain(d), d = simplify(d,'full');end
MtyRz=ss(a,b,c,d);
if isuncertain(MtyRz), MtyRz = simplify(MtyRz,'full');end

m=ro*l*s;
we2=3*e/(l^2*ro);
if isuncertain(we2), 
    we20=we2.NominalValue;
else
    we20=we2;
end
Mtx=ss([0 1;-we2 -2*xi*sqrt(we20)],[0 0;3/m -3/2],...
       [-we2 -2*xi*sqrt(we20);1.5*e*s/l m*xi*sqrt(we20)],[3/m -1/2;-1/2 -m/4]);
if isuncertain(Mtx), Mtx = simplify(Mtx,'full');end

M=append(MtyRz,Mtx);

P=zeros(6,6);    % Permutation Matrix
P(1,2)=1; P(2,3)=1; P(3,5)=1; P(4,6)=1; P(5,1)=1; P(6,4)=1;
M=P'*M*P;

end

