function [bid]=rsqlocus2(PLANT,M,P)
% RSQLOCUS2(PLANT,M,P)
%       Trace les 2 lieux du carré des racines découplés
%       (synthèse H2) associés à la forme standard PLANT:
%               |A   |B1  |B2 |
%       PLANT = |C1  |D11 |D12|
%               |C2  |D21 |D22| ,
%       M: Nombre de commandes,
%       P: Nombre de mesures,
%       - Lieu relatif à l'équation de Riccati de commande  
%       - Lieu relatif à l'équation de Riccati d'obsversation
%
%       Voir aussi RSQLOCUS 


% D. Alazard 01/94
% Copyright (c) 1993-2000 ONERA/DCSD, All Rights Reserved.
[A,B,C,D]=test_natss(PLANT);
mtot=size(B,2);
ptot=size(C,1);
B1=B(:,1:mtot-M);  B2=B(:,mtot-M+1:mtot);
C1=C(1:ptot-P,:);  C2=C(ptot-P+1:ptot,:);
D11=D(1:ptot-P,1:mtot-M);      D12=D(1:ptot-P,mtot-M+1:mtot);
D21=D(ptot-P+1:ptot,1:mtot-M); D22=D(ptot-P+1:ptot,mtot-M+1:mtot);

%
subplot(2,1,1);
[n,m]=size(B2);
QQ=[C1 D12]'*[C1 D12];
Q=QQ(1:n,1:n);
N=QQ(1:n,n+1:n+m);
R=QQ(n+1:n+m,n+1:n+m);
rsqlocus(A,B2,Q,R,N);
%hold on
%rsqlocus(A,B2,Q,R,N,1);
title('State feedback dynamic')
[kx,sx,ex]=lqr(A,B2,Q,R,N);
%
subplot(2,1,2);
[p,n]=size(C2);
QQ=[B1;D21]*[B1;D21]';
Q=QQ(1:n,1:n);
N=QQ(1:n,n+1:n+p);
R=QQ(n+1:n+p,n+1:n+p);
rsqlocus(A',C2',Q,R,N);
hold on
%rsqlocus(A',C2',Q,R,N,1);
%
clc
rep=input('Tracé de la dynamique du correcteur H2 (1/0) ? ');
if rep==1,
% [lx,px,fx]=lqe(A,eye(size(A)),C2,Q,R,N);
 lx=lqr(A',C2',Q,R,N);lx=lx';
 rlocusp(A-lx*C2,B2-lx*D22,kx,0*ones(m,m));
 %rlocusp(A-lx*C2,B2-lx*D22,kx,0*ones(m,m),1);
 title('Kalman filter dynamic and controller dynamic via state feedback');
else,
 title('Kalman filter dynamic');
end;
portrait
