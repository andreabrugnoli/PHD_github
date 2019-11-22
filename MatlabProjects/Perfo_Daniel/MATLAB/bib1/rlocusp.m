function [rout]=rlocusp(a,b,c,d,kk)
%RLOCUSP Lieu des racines points par points pour un système carré
%       (autant d'entrées que de sorties).
%       Ne présente un intéret que dans le cas d'un système
%       décrit par sa représentation d'état.
%       (sinon utiliser RLOCUS).
%
%       RLOCUSP(A,B,C,D) calcule et trace le lieu des valeurs
%       propres de la matrice:
%
%            Abf=A-kBC
%
%       pour un jeu de gains k calcule automatiquement.
%       La dynamique en boucle fermée obtenue pour le gain de 
%       boucle nominal (k=1) est reprérée par des '+' bleus.
%
%       R=RLOCUSP(A,B,C,D,K) permet de fixer dans le vecteur K
%       les variations du gain et retourne une matrice R avec 
%       LENGTH(K) colonnes et LENGTH(A) lignes contenant les 
%       valeurs propres pour les valeurs de gain correspondantes.
%
% Autres syntaxes:
%       R=RLOCUSP(SYSIN) ou R=RLOCUS(SYSIN,K);
%       SYSIN: objet LTI ou PCK.
%
%       Voir aussi RLOCUS, RSQLOCUS, PZMAP

% D. Alazard 01/94
% Copyright (c) 1993-2000 ONERA/DCSD, All Rights Reserved.
if nargin==0, eval('exresp(''rlocusp'');'), return, end
if nargin==1, [a,b,c,d]=test_natss(a);kk=[];end
if nargin==2, kk=b;[a,b,c,d]=test_natss(a);end
if nargin==4, kk=[];end
	
[m,n] = size(a);
[mb,nb] = size(b);
[mc,nc] = size(c);
if (nb ~= mc) 
	error('Le systeme doit etre carre. ');
end;

ep=eig(a);
p=length(ep);
tz=tzero(a,b,c,d);
tz=tz(abs(tz)<1.e6);		% Ignore zeros greater than 1.e6
z=length(tz);
epbf=eig(a-b*((eye(size(d))+d)\...
		(eye(size(d))))*c);
%mep=max([eps;abs([real(ep);real(tz);imag(ep);imag(tz)])]);
mep=max([eps;abs([real(ep);real(epbf);imag(ep);imag(epbf)])]);
	if z==p
		ax=1.2*mep;        
	else
		% Round graph axis    
		exponent=5*10^(floor(log10(mep))-1);
		ax=2*round(mep/exponent)*exponent;    
	end
holdon = ishold;
newplot;
if ~holdon
	plot([-ax,ax],[0,0],'b:',[0,0],[-ax,ax],'b:');
	axis([-ax,ax,-ax,ax])
else
	ax4 = axis;
	ax = sum(abs(ax4))/4;
	plot([ax4(1:2)],[0,0],'b:',[0,0],[ax4(3:4)],'b:');
	axis(ax4)
end
hold on
plot(real(ep),imag(ep),'x');
if ~isempty(tz)
	plot(real(tz),imag(tz),'o');
end
xlabel('Real Axis')
ylabel('Imag Axis')
erasemode = 'none';
drawnow
%
%Recherche des variations de gains
if isempty(kk),
	ii=0;
	k=eps;
	r(:,1)=ep;
	abf=a-b*((eye(size(d))+k*d)\(k*eye(size(d))))*c;
	[V,D]=eig(abf);
	sp=sum([1:length(a)].^2);
	while ii<=500,
		ii=ii+1;
		if cond(V)<100/eps,
		   U=inv(V);
		   dlamdai=[];
		   for jj=1:p
		   	dlamdai=[dlamdai U(jj,:)*b*c*V(:,jj)];
		   end;
		   deltak=ax/100/max(abs(dlamdai));
		   k=k+deltak;
		   if isinf(k),k=1/eps;end;
		else,
		   k=k+1000*eps;
		end;
	        abf=a-b*((eye(size(d))+k*d)\(k*eye(size(d))))*c;
		[V,D]=eig(abf);
		r(:,ii+1)=vsort(r(:,ii),diag(D),sp);
	end;
%
	for ii=1:length(a),
 	  plot(real(r(ii,:)),imag(r(ii,:)),'r-');
   end;
   rlocusp(a,b,c,d,1);
else
	for ii=1:length(kk)
		rout(:,ii)=eig(a-b*((eye(size(d))+kk(ii)*d)\...
		(kk(ii)*eye(size(d))))*c);
                plot(real(rout(:,ii)),imag(rout(:,ii)),'b+');
	end
end
%
if ~holdon, hold off, end

% Internal function VSORT.M
function [v2,pind]=vsort(v1,v2,sp)
% VSORT Matches two vectors.  Used in RLOCUS.
%	VS2 = VSORT(V1,V2) matches two complex vectors 
%	V1 and V2, returning VS2 with consists of the elements 
%	of V2 sorted so that they correspond to the elements 
%	in V1 in a least squares sense.

%     	[v2,pind]=VSORT(v1,v2,sp) 
%       sp is used to test a quick sort method and is equal to
%       sp=sum([1:length(indr)].^2); pind=1 is returned if
%       a slow sort method has to be applied.

%	Copyright (c) 1986-93, by the MathWorks, Inc.

pind=0;
if nargin < 3, sp = sum([1:length(v1)].^2); end
% Quick Sort 
p=length(v2);
vones=ones(p,1);
[dum,indr1]=min(abs(vones*v2.'-v1*vones'));
indr(indr1)=[1:p];

% Long (accurate) sort
if (indr*indr' ~= sp) 
	[dum,jnx]=sort(abs(v2));
	pind=1;
	for j=jnx' 
		[dum,inx]=min(abs(v2(j)-v1));
		indr(inx)=j;
		v1(inx)=1e30;
	end

end
v2=v2(indr);
