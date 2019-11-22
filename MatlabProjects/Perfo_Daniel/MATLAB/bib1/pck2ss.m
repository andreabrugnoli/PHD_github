function [sys]=pck2ss(pack,T)
% [SYS]=PCK2SS(PACK)  permet de convertir un système continu
% représenté par une matrice système PACK (créée par PCK) 
% en objet LTI décrit par une représentation d'état SYS.
%				                            		
% [SYS]=PCK2SS(PACK,T)  permet de convertir un système échantillonné
% à la cadence T et représenté par une matrice système PACK 
% (créée par PCK) en objet LTI décrit par une représentation d'état SYS.
%				                            		
% Voir aussi: SS2PCK, PCK et UNPCK

% S. Delonnoy 01/99
% Copyright (c) 1993-2000 ONERA/DCSD, All Rights Reserved.

[ass,bss,css,dss]=unpck(pack);
if nargin==1,
   sys=ss(ass,bss,css,dss);
else
   sys=ss(ass,bss,css,dss,T);
end;
