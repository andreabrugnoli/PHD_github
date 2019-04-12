function [M] = TwoPortBeam(ro,s,l,e,g,iz,iy,ipx,xi)
% M=TwoPortBeam(ro,s,l,e,g,iz,iy,ipx,xi) computes the 2 input/output
% model M (12x12) of a uniform beam characterized by:
%   * mass density: ro  (Kg/m^3),
%   * section: s (m^2),
%   * lenght: l (m),
%   * Young modulus: e (Pascal or N/m^2),
%   * shear modulus: g (Pascal or N/m^2),
%   * second moment of aera w.r.t z axis: iz (m^4),
%   * second moment of aera w.r.t y axis: iy (m^4),
%   * polar second moment of aera w.r.t x axis: ipx (m^4),
%   * xi: arbitrary damping ratio for all flexible modes.
%          ^ y(x)
%          |
%          |
%          x=======================x---------> x
%         /P(x=0)                 C(x=l)
%        /
%       z
%
%   The 12 inputs of M are:
%      * the 6 components of the external force/torque vector applied
%        to the beam at point C (in the frame (P,x,y,z)),
%      * the 6 components of the linear/angular acceleration vector at
%        point P (in the frame (P,x,y,z)).
%   The 12 outputs of M are:
%      * the 6 components of the linear/angular acceleration vector at
%        point C (in the frame (P,x,y,z)),
%      * the 6 components of the external force/torque vector applied
%        by the beam to outside at point P (in the frame (P,x,y,z)).
%   Only pure flexion in the plane (P,x,y), in the plane (P,x,z) traction and
%   torsion along (P,x) axis are considered.
%   This fonction supports uncertain parameters (see ureal).
%
%   See also: MKelemu, TwoPortBeamTyRz

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

[My,Ky] = MKelemu(ro,s,l,e,iy);
Mm1=inv(My(3:6,3:6));
if isuncertain(Mm1), Mm1 = simplify(Mm1,'full');end
Mm1K=Mm1*Ky(3:6,3:6);
if isuncertain(Mm1K), 
    Mm1K = simplify(Mm1K,'full');
    Mm1K0=Mm1K.NominalValue;
else
    Mm1K0=Mm1K;
end
L=Mm1*My(3:6,1:2);
if isuncertain(L), L = simplify(L,'full');end
[V,D]=eig(Mm1K0);
a=[zeros(4) eye(4);-Mm1K -V*2*xi*sqrt(D)*inv(V)];
b=[zeros(4,4);[Mm1*phiC' -L]];
if isuncertain(b), b = simplify(b,'full');end
%c=[[-phiC*Mm1K; My(1:2,3:6)*Mm1K] [-phiC;My(1:2,3:6)]*V*2*xi*sqrt(D)*inv(V)];
c=[phiC; -My(1:2,3:6)]*a(5:8,:);
if isuncertain(c), c = simplify(c,'full');end
d=[phiC*Mm1*phiC' tau-phiC*L;(tau-phiC*L)' -My(1:2,1:2)+My(1:2,3:6)*L];
%d=[zeros(2) tau;tau' -My(1:2,1:2)]+[-phiC;My(1:2,3:6)]*b(5:8,:);
if isuncertain(d), d = simplify(d,'full');end
MtzRy=ss(a,b,c,d);
if isuncertain(MtzRy), MtzRy = simplify(MtzRy,'full');end


Jx=ro*l*ipx;
we2=3*g/(l^2*ro);
if isuncertain(we2), 
    we20=we2.NominalValue;
else
    we20=we2;
end
MRx=ss([0 1;-we2 -2*xi*sqrt(we20)],[0 0;3/Jx -3/2],...
       [-we2 -2*xi*sqrt(we20);1.5*g*ipx/l Jx*xi*sqrt(we20)],[3/Jx -1/2;-1/2 -Jx/4]);
if isuncertain(MRx), MRx = simplify(MRx,'full');end

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

%M=append(MtyRz,MtzRy,MRx,Mtx);
M=append(MtyRz,diag([1,-1,1,-1])*MtzRy*diag([1,-1,1,-1]),MRx,Mtx);

P=zeros(12,12);    % Permutation Matrix
P(1,2)=1; P(2,6)=1;  P(3,8)=1; P(4,12)=1;
P(5,3)=1; P(6,5)=1;  P(7,9)=1; P(8,11)=1;
P(9,4)=1; P(10,10)=1;P(11,1)=1;P(12,7)=1;
M=P'*M*P;

end

