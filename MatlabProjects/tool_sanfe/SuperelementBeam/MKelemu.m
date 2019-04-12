function [M,K] = MKelemu(ro,s,l,e,i)
% [M,K] = MKelemu(ro,s,l,e,i) computes the mass matrix M (6x6) and the
% stiffness matrix K (6x6) of a uniform beam characterized by:
%   * mass density: ro  (Kg/m^3),
%   * section: s (m^2),
%   * lenght: l (m),
%   * Young modulus: e (Pascal or N/m^2),
%   * second moment of aera: i (m^4).
% M and K are relative to the vector Q of generalized coordinates:
%   Q=[y(0) dy/dx|0 d^2y/dx^2|0 y(l) dy/dx|l d^2y/dx^2|l]^T with
%     * q3=d^2y/dx^2|0;
%     * q4=y(l) - y(0) - l*dy/dx|0;
%     * q5=dy/dx|l - d^2y/dx^2|0;
%     * q6=d^2y/dx^2|l;
%   where y(x) is the lateral deflection at the point of abcisse x along 
%   the beam.
%   Only pure flexion in the plane (x,y) is considered.
%          ^ y(x)
%          |
%          |
%          ========================----------> x
%         /0                      l
%        /
%       z
%   This fonction supports uncertain parameters (see ureal).
%
%   See also: TwoPortBeamTyRz, TwoPortBeam

%  D. Alazard (05/2014)
m=ro*s*l;
M=m/55440*[21720 3732*l   281*l^2  6000  -1812*l   181*l^2;
         3732*l 832*l^2 69*l^3 1812*l -532*l^2 52*l^3;
         281*l^2 69*l^3   6*l^4   181*l^2  -52*l^3    5*l^4;
         6000   1812*l   181*l^2  21720  -3732*l   281*l^2;
         -1812*l -532*l^2 -52*l^3 -3732*l 832*l^2   -69*l^3;
         181*l^2 52*l^3    5*l^4  281*l^2 -69*l^3     6*l^4];
if isuncertain(M), M = simplify(M,'full') ;end
K=e*i/(70*l^3)*[1200  600*l  30*l^2  -1200  600*l  -30*l^2;
                600*l 384*l^2 22*l^3 -600*l 216*l^2 -8*l^3;
                30*l^2 22*l^3 6*l^4  -30*l^2 8*l^3     l^4;
                -1200 -600*l -30*l^2  1200  -600*l  30*l^2;
                600*l 216*l^2 8*l^3 -600*l 384*l^2 -22*l^3;
                -30*l^2 -8*l^3 l^4  30*l^2 -22*l^3   6*l^4];
if isuncertain(K), K = simplify(K,'full');end
end

