function [K,E]=rsqlocus(a,b,q,r,nn,e)
%RSQLOCUS  lieu du carré des racines pour les systèmes continus.
%       RSQLOCUS(A,B,Q,R) calcule et trace le lieu des
%       valeurs propres de la matrice hamiltonienne 
%       correspondante à la minimisation du critere:
%
%               J = Integral {x'Qx + ku'Ru} dt
%
%       sous les contraintes:
%               .
%               x = Ax + Bu 
%
%       lorsque k varie (variations calculees automatiquement). La
%       dynamique obtenue pour la valeur nominale k=1 est repérée
%       par de '+' bleux).
%          
%       RSQLOCUS(A,B,Q,R,N) inclus les termes couplés
%       états/commandes dans le critère:
%
%               J = Integral {x'Qx + ku'Ru + 2*x'Nu}
%          
%       [GAIN,E]=RSQLOCUS(A,B,Q,R,N,K) permet de fixer dans le
%       vecteur K les variations de k et retourne:
%           -Une matrice GAIN avec LENGTH(K) lignes et LENGTH(A)
%            colonnes contenant les valeurs des gains de retour d'état
%               u=-GAIN.x satisfaisant le critère pour les valeurs de k
%               correspondantes. 
%             - une matrice E avec LENGTH(K) colonnes et LENGTH(A)
%               lignes contenant les valeurs propres stables de la matrice 
%               hamiltonienne pour les valeurs de k correspondantes.
%          
%       Dans le cas general (k=1), la matrice hamiltonienne
%       s'ecrit:
%                   |    -1           -1  |
%                   |A-BR  N'      -BR  B'|
%               H = |                     |
%                   |    -1        -1     |
%                   |-Q+NR  N'   NR  B'-A'|
%          
%       Voir aussi: LQR, RLOCUSP


% D. Alazard 01/94
% Copyright (c) 1993-2000 ONERA/DCSD, All Rights Reserved.

error(nargchk(4,6,nargin));
error(abcdchk(a,b));
if ~length(a) | ~length(b)
        error('A and B matrices cannot be empty.')
end

[m,n] = size(a);
[mb,nb] = size(b);
[mq,nq] = size(q);
if (m ~= mq) | (n ~= nq) 
	error('A and Q must be the same size');
end
[mr,nr] = size(r);
if (mr ~= nr) | (nb ~= mr)
	error('B and R must be consistent');
end

if (nargin == 5) | (nargin == 6)
	[mn,nnn] = size(nn);
	if (mn ~= m) | (nnn ~= nr)
		error('N must be consistent with Q and R');
	end
else
	nn = zeros(m,nb);
end


aa=[a 0*a;-q -a'];
rm1=inv(r);
bb=[b;-nn]*sqrtm(rm1);
cc=sqrtm(rm1)*[nn' b'];
dd=0*ones(size(cc*bb));
if (nargin==4) | (nargin==5)
	rlocusp(aa,bb,cc,dd)
end
if (nargin==6)
	rlocusp(aa,bb,cc,dd,1./e);
	for ii=1:length(e)
		kk=1+nb*(ii-1);
		[v,d]=eig(aa-1./e(ii)*bb*cc);
		d=diag(d);
		[F,index]=sort(real(d));
                E(:,ii)=d(index(1:n));
		chi=v(1:n,index(1:n));
		lambda = v((n+1):(2*n),index(1:n));
		s = -real(lambda/chi);
		K(kk:kk+nb-1,:) = (e(ii)*r)\(nn'+b'*s);
	end;
end;
