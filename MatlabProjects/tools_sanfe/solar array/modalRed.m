function [sysr]=modalred(sys,keep)
% 
% [SYSR] = MODALRED(SYS) : Interactive Modal Reduction.
%
% This function plots the poles-zeros maps and the frequency-
% domain response (BODEMAG)  of the system SYS (time-continuous
% LTI system).
% Then, a user graphics interface is proposed to select some modes
% in the system SYS to be eliminated in the reduced system SYSR. 
% The reduction is performed in the real block diagonal form of
% SYS obtained with function CANON.
%
% [SYSR] = MODALRED(SYS,1) then, the selected modes of SYS are 
% the modes of SYSR, all other modes are eliminated.
% 
% See also: CANON, MODRED

%   D. Alazard
%   Copyright (c) 2000-2015 ISAE-SUAPERO, All Rights Reserved.


if nargin ~=1 & nargin ~=2, 
    error('Wrong input arguments: type HELP modalRed');
    return
end;
if nargin==1, keep=0;end,

sysm=canon(sys,'modal');
figure
subplot(1,2,2);
%bodemag(sysm);
sigma(sysm);
subplot(1,2,1);
pzmap(sysm);
hold on
index = polefind(sysm.a);

    if keep==0,
        sysr=modred(sysm,index);
        disp(' ');
        disp(['Reduction of ',num2str(length(index)),' pole(s) in the model.']);
    else
        indexelim=indexsub([1:size(sysm.a,1)],index);
        sysr=modred(sysm,indexelim);
        disp(' ');
        disp(['Reduced model order: ',num2str(length(index))]);        
    end

function index=polefind(a)
% INDEX = POLEFIND(A) allowes to select interactively (through
% a user-graphic interface) some eigenvalues of the matrix A.
% 
% The vector INDEX are the indexes of the state variables 
% associated with the selected eigenvalues. 
% WARNING: its is assumed that A is a matrix with real block
% diagonal form (obtanined with function CANON for instance).
%
% See also: canon

% ONERA/DCSD S. Delannoy, D.Alazard 1/99
% Revised by D. Alazard 12/03

index = [];
%zoom on
%hold on
lafigurecourante=gca;

KK = 1;

index=[];
handler=[];
while KK ~= 5
        KK=menu({'Do your choice:'},'select a mode (left plot)','select a cloud of modes (left plot)','select a mode from the frequency response (right plot)','unselect a mode (left plot)','stop');
	if KK == 1, 
         axes(lafigurecourante);
         ind=nouveau_point(a);
	     message = redondance(ind(1),index);
	     if ~message,
	            handi=plot(real(eig(a(ind,ind))),imag(eig(a(ind,ind))),'c*','LineWidth',2);
	            index=[index;[ind]];
	            handler=[handler;handi];
	            if length(ind)==2,
	               handler=[handler;handi];
	            end;
	     end
           
    elseif KK == 2, 
             axes(lafigurecourante);
             ind=nuage(a,index);
             index=[index;[ind]];
             hh=1;
             while hh<length(ind),
                 jj=ind(hh);
                 if a(jj,jj+1)==0,
                    handi=plot(a(jj,jj),0,'c*','LineWidth',2);
                    handler=[handler;handi];
                    hh=hh+1;
                 else
                    handi=plot(real(eig(a(jj:jj+1,jj:jj+1))),imag(eig(a(jj:jj+1,jj:jj+1))),'c*','LineWidth',2);
                    handler=[handler;handi];
                    handler=[handler;handi];
                    hh=hh+2;
                 end
             end
             if hh==length(ind),
                 jj=ind(hh);
                 handi=plot(a(jj,jj),0,'c*','LineWidth',2);
                 handler=[handler;handi];
             end   
    elseif KK==3, ind=nouveau_point_fr(a);
	     message = redondance(ind(1),index);
	     if ~message,
                axes(lafigurecourante)
	            handi=plot(real(eig(a(ind,ind))),imag(eig(a(ind,ind))),'m*','LineWidth',2);
	            index=[index;[ind]];
	            handler=[handler;handi];
	            if length(ind)==2,
	               handler=[handler;handi];
	            end;
	     end
    elseif KK==4, 
        axes(lafigurecourante);
        ind=nouveau_point(a);
	    if ~isempty(ind),
           indice=find(index==ind(1));
	       delete(handler(indice));
	       indice=find(index~=ind(1));
	       if length(ind)==2,
	             indice=find((index~=ind(1))&(index~=ind(2)));
	       end;
	       if ~isempty(handler) handler=handler(indice);end;
           if ~isempty(index) index=index(indice);end;
	    end
    end;
end;
% On efface les marqueurs
delete(handler)
% On retourne � la pleine �chelle
zoom out

% =============================================
% fonction interne
% =============================================

function indo=nouveau_point(a)
% rajoute un mode a la selection

indo=[];
V=eig(a);
taille=size(V,1);
reV=real(V);
imV=imag(V);

	liste = ginput(1);
	
	i=sqrt(-1);
			test1 = V-(liste(1)+i*liste(2))*ones(taille,1);
			abs_test1 = abs(test1);
			[mini1,ind] = min(abs_test1);
			indo=ind;
		if abs(imV(ind)) > 1e-6*abs(reV(ind))
			test2 = V-(liste(1)-i*liste(2))*ones(taille,1);
			abs_test2 = abs(test2);
			[mini2,ind2] = min(abs_test2);
			ind1 = min(ind,ind2);
			ind2 = max(ind,ind2);

			indo=[ind1;ind2];
			
		end;

% end of function nouveau_point


% =============================================
% fonction interne
% =============================================

function indo=nouveau_point_fr(a)
% rajoute un mode a la selection

indo=[];
V=eig(a);
taille=size(V,1);
w=zeros(taille,1);xi=zeros(taille,1);
for ii=1:taille,
    [w(ii),xi(ii)]=damp(V(ii));
end
wr=real(w.*sqrt(1-2*(xi.^2)));

	liste = ginput(1);
	
			test1 = wr-liste(1)*ones(taille,1);
			abs_test1 = abs(test1);
			[mini1,ind] = min(abs_test1);
            indo=find(wr==wr(ind));
            
% end of function nouveau_point_fr



% ==========================================
% fonction interne
% ==========================================
function message=redondance(ind,index,k)
% verifie que le mode n'a pas deja ete selectionne
%
message = 0;
if ~isempty(index),
xx = find(index==ind);
if ~isempty(xx), 
		if nargin == 2,
                disp('============================================ ');
		disp('This mode was already selected');
  		elseif nargin == 3,
                disp('============================================ ');
fprintf(['The ',num2str(k),'-th selected mode choisi was already selected\n']);
  		end;
  		message = 1;
end; 
end;
% end of function redondance


% ==========================================
% fonction interne
% ==========================================
function indo=nuage(a,index)
% rajoute à la sélection les mode souples du nuage sélectionné
% un tri est effectué pour ne pas rajouter des modes déjà
% présents dans INDEX

indo=[];
V=eig(a);
taille=size(V,1);
reV=real(V);
imV=imag(V);
OK=0;
i=sqrt(-1);

% disp('le nuage se pr�sente sous la forme rectangulaire ')
% disp(['il ne doit pas traverser l''axe r�el (impossibilit� d''y  ' ,...
%     'inclure des modes r�els ).'])
% disp('pour d�finir ce nuage, cliquez en deux sommets antidiagonaux')
% disp('  ') 

%while OK~=1,
  
  waitforbuttonpress;
  point1 = get(gca,'CurrentPoint');
  finalRect=rbbox;
  point2 = get(gca,'CurrentPoint');    % button up detected
  point1 = point1(1,1:2);              % extract x and y
  point2 = point2(1,1:2);
  xmin=min([point1(1,1),point2(1,1)]);
  ymin=min([point1(1,2),point2(1,2)]);
  xmax=max([point1(1,1),point2(1,1)]);
  ymax=max([point1(1,2),point2(1,2)]);
% liste = ginput(2);
%   xmin=min([liste(1,1),liste(2,1)]);
%   ymin=min([liste(1,2),liste(2,2)]);
%   xmax=max([liste(1,1),liste(2,1)]);
%   ymax=max([liste(1,2),liste(2,2)]);
  
%   if sign(ymin)*sign(ymax) ~=1,
%     disp('le nuage ne doit pas traverser l''axe r�el, red�finissez le ')
%   else,
%     OK=1;
%   end;
%end;

indico1=find((real(V)>xmin) & (real(V)<xmax) & (imag(V)>ymin) & (imag(V)<ymax));	
indico2=find((real(V)>xmin) & (real(V)<xmax) & (-imag(V)>ymin) & (-imag(V)<ymax));	

indico=sort([indico1;indico2]);
indico=unique(indico);
li=length(indico);

if (li~=0) & (~isempty(index)),
  asuppr=[];
  for tt=1:li,
    indii=find(index==indico(tt));
    if ~isempty(indii),
      asuppr=[asuppr;indico(tt)];
    end;
  end;

  if ~isempty(asuppr),
    indico=indexsub(indico,asuppr);
    indico=indico';
  end;

end;

indo=indico;

%end of function NUAGE

function index=indexsub(ind1,ind2)
%INDEX=INDEXSUB(INDX1,INDX2) 
%   INDEX coorespond au vecteur INDX1 privé des composantes qui sont 
%   dans le vecteur INDX2. 

% D. Alazard /01/01/95
% Copyright (c) 1993-2000 ONERA/DCSD, All Rights Reserved.

index=[];
ind=1;
if ~isempty(ind2);
	for ii=1:length(ind1),
 	    if isempty(find(ind2==ind1(ii))),
     	        index(ind)=ind1(ii);
     	        ind=ind+1;
  	    end;
	end;
else
	index=ind1;
end;
